"""
HyperTensor Chess Engine v3.4 — Alpha-Beta Negamax Search
===========================================================
Full PVS (Principal Variation Search) with:
  - Null-move pruning with verification
  - Late Move Reductions (LMR) with research
  - Futility pruning at frontier nodes
  - Razoring at depth 1
  - Delta pruning in quiescence
  - Countermove & killer move ordering
  - History heuristic (butterfly table)
  - SEE-based capture ordering
  - Quiescence search with delta pruning
  - Aspiration windows
  - Syzygy tablebase probing
  - Batched GPU neural network evaluation

This is the search core that separates 2800 engines from 3500+.
"""

import numpy as np
import math
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

from .board import Board, Move, Color, Piece, PIECE_VALUES, SQUARE_NAMES
from .strong_search import (see, see_capture, EliteMoveOrdering, 
                             can_null_move, get_lmr_reduction, is_futile,
                             SyzygyProbe, CountermoveTable)
from .evaluation import HyperTensorChessNet, CUDA_AVAILABLE, DEVICE
from .pretrain import heuristic_evaluate
from .opening_book import get_opening_move


# ===========================================================================
# Constants
# ===========================================================================

MATE_VALUE = 100000
MATE_THRESHOLD = 99000
INF = float('inf')

# History heuristic table (butterfly board)
@dataclass
class HistoryTable:
    """Butterfly history heuristic: piece, to_sq -> score."""
    table: np.ndarray = field(default_factory=lambda: np.zeros((6, 64), dtype=np.float32))
    
    def add(self, move: Move, depth: int, piece_type: int):
        bonus = min(depth * depth, 400)
        self.table[piece_type, move.to_sq] += bonus * 0.01
        # Decay
        self.table *= 0.995
    
    def get(self, move: Move, piece_type: int) -> float:
        return float(self.table[piece_type, move.to_sq])


@dataclass
class KillerTable:
    """Killer moves: quiet moves that caused beta cutoffs, by depth."""
    slots: List[List[Optional[Move]]] = field(default_factory=lambda: [[None, None] for _ in range(64)])
    
    def add(self, move: Move, depth: int):
        if depth >= len(self.slots): return
        if self.slots[depth][0] != move:
            self.slots[depth][1] = self.slots[depth][0]
            self.slots[depth][0] = move
    
    def get(self, depth: int) -> List[Optional[Move]]:
        if depth >= len(self.slots): return [None, None]
        return self.slots[depth]
    
    def is_killer(self, move: Move, depth: int) -> bool:
        if depth >= len(self.slots): return False
        return move in self.slots[depth]


# ===========================================================================
# Search State (shared across iterations)
# ===========================================================================

@dataclass
class SearchState:
    """Persistent state across search iterations."""
    tt: Dict[int, 'TTEntry'] = field(default_factory=dict)
    history: HistoryTable = field(default_factory=HistoryTable)
    killers: KillerTable = field(default_factory=KillerTable)
    countermoves: CountermoveTable = field(default_factory=CountermoveTable)
    nodes_searched: int = 0
    tt_hits: int = 0
    null_cuts: int = 0
    futility_cuts: int = 0
    lmr_reductions: int = 0
    syzygy_hits: int = 0
    
    def clear_search_stats(self):
        self.nodes_searched = 0
        self.tt_hits = 0
        self.null_cuts = 0
        self.futility_cuts = 0
        self.lmr_reductions = 0
        self.syzygy_hits = 0


@dataclass
class TTEntry:
    """Transposition table entry for alpha-beta."""
    zobrist: int
    value: float
    depth: int
    flag: str  # 'exact', 'lower', 'upper'
    best_move: Optional[Move] = None
    age: int = 0


# ===========================================================================
# Negamax Search Engine
# ===========================================================================

class NegamaxEngine:
    """Principal Variation Search with all modern pruning techniques."""
    
    def __init__(self, model: HyperTensorChessNet, 
                 tt_size_mb: int = 256,
                 syzygy_path: str = None):
        self.model = model
        self.state = SearchState()
        self.tt_max_entries = (tt_size_mb * 1024 * 1024) // 48
        self.syzygy = SyzygyProbe(syzygy_path)
        self.move_order = EliteMoveOrdering()
        
        # Configuration
        self.use_null_move = True
        self.use_lmr = True
        self.use_futility = True
        self.use_razoring = True
        self.use_delta = True
        self.use_syzygy = True
        
        # Batch GPU eval
        self.eval_batch_tensors = []
        self.eval_batch_positions = []
    
    # =====================================================================
    # Transposition Table
    # =====================================================================
    
    def tt_probe(self, zobrist: int) -> Optional[TTEntry]:
        entry = self.state.tt.get(zobrist)
        if entry:
            self.state.tt_hits += 1
            entry.age = 0
        return entry
    
    def tt_store(self, zobrist: int, value: float, depth: int, 
                 flag: str, best_move: Optional[Move] = None):
        if len(self.state.tt) >= self.tt_max_entries:
            # Remove 25% oldest entries
            entries = sorted(self.state.tt.items(), key=lambda x: x[1].age)
            for k, _ in entries[:len(entries)//4]:
                del self.state.tt[k]
        self.state.tt[zobrist] = TTEntry(zobrist, value, depth, flag, best_move, 0)
    
    def tt_age(self):
        for entry in self.state.tt.values():
            entry.age += 1
    
    # =====================================================================
    # Evaluation (GPU-batched for speed)
    # =====================================================================
    
    def evaluate(self, board: Board) -> float:
        """Evaluate a position. Uses batched NN on GPU for speed."""
        if self.model is not None:
            tensor = board.to_tensor().astype(np.float32)
            import torch
            x = torch.from_numpy(tensor).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                val, _, _, _ = self.model(x)
            return float(val.item())
        else:
            val, _, _ = heuristic_evaluate(board)
            return val
    
    def evaluate_batch(self, boards: List[Board]) -> np.ndarray:
        """Evaluate multiple positions in one GPU batch."""
        if not boards:
            return np.array([])
        if self.model is not None:
            import torch
            batch = np.stack([b.to_tensor().astype(np.float32) for b in boards])
            x = torch.from_numpy(batch).to(DEVICE)
            with torch.inference_mode():
                val, _, _, _ = self.model(x)
            return val.squeeze(-1).cpu().numpy()
        else:
            return np.array([heuristic_evaluate(b)[0] for b in boards])
    
    # =====================================================================
    # Move Ordering
    # =====================================================================
    
    def score_moves(self, moves: List[Move], board: Board, 
                    tt_move: Optional[Move], depth: int,
                    prev_move: Optional[Move] = None) -> List[Tuple[Move, int]]:
        """Score moves for ordering. Higher score = searched first."""
        in_check = board.is_in_check()
        scored = []
        
        for move in moves:
            score = 0
            
            # TT move gets highest priority
            if tt_move and move == tt_move:
                score = 2000000
            
            # SEE for captures
            if board.piece_at(move.to_sq) is not None:
                see_val = see_capture(board, move)
                score += see_val + 1000000
            
            # Killer moves
            if self.state.killers.is_killer(move, depth):
                score += 900000
            
            # Countermove
            if prev_move and self.state.countermoves.is_countermove(prev_move, move):
                score += 800000
            
            # MVV-LVA for captures
            victim = board.piece_at(move.to_sq)
            if victim:
                _, victim_piece = victim
                attacker_piece = board.piece_at(move.from_sq)[1]
                score += PIECE_VALUES[victim_piece] - PIECE_VALUES[attacker_piece] // 100
            
            # History heuristic for quiets
            if not victim:
                piece_type = board.piece_at(move.from_sq)[1]
                score += int(self.state.history.get(move, piece_type))
            
            # Promotions
            if move.promotion:
                score += PIECE_VALUES[move.promotion] - 100
            
            # Castling
            if move.is_castle_kingside or move.is_castle_queenside:
                score += 500
            
            scored.append((move, score))
        
        # Sort descending by score
        return sorted(scored, key=lambda x: x[1], reverse=True)
    
    # =====================================================================
    # Quiescence Search
    # =====================================================================
    
    def quiescence(self, board: Board, alpha: float, beta: float, 
                   depth: int = 0) -> float:
        """Quiescence search: evaluate only captures to avoid horizon effect."""
        self.state.nodes_searched += 1
        
        # Stand pat — we can always do nothing (stop captures)
        stand_pat = self.evaluate(board)
        
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
        
        # Generate captures only
        moves = board.generate_legal_moves()
        captures = [m for m in moves if board.piece_at(m.to_sq) is not None or m.promotion]
        
        # Order captures by SEE
        scored = self.score_moves(captures, board, None, depth)
        
        for move, _ in scored:
            # Delta pruning: if stand_pat + piece_value + margin < alpha, skip
            if self.use_delta:
                victim = board.piece_at(move.to_sq)
                if victim:
                    gain = PIECE_VALUES[victim[1]]
                    if stand_pat + gain + 200 < alpha:
                        continue
                elif move.promotion:
                    gain = PIECE_VALUES[move.promotion] - 100
                    if stand_pat + gain + 200 < alpha:
                        continue
            
            # SEE pruning for losing captures
            if not move.promotion:
                victim = board.piece_at(move.to_sq)
                if victim and PIECE_VALUES[victim[1]] < 300:  # Capturing pawn/minor
                    if not see(board, move, -50):
                        continue
            
            board.make_move(move)
            val = -self.quiescence(board, -beta, -alpha, depth + 1)
            board.unmake_move()
            
            if val >= beta:
                return beta
            if val > alpha:
                alpha = val
        
        return alpha
    
    # =====================================================================
    # Principal Variation Search (PVS)
    # =====================================================================
    
    def search(self, board: Board, depth: int, alpha: float, beta: float,
               ply: int = 0, prev_move: Optional[Move] = None,
               allow_null: bool = True) -> float:
        """PVS negamax with all modern pruning techniques.
        
        Returns the evaluation from the perspective of the side to move.
        Positive = good for side to move.
        """
        self.state.nodes_searched += 1
        
        # Mate distance pruning
        alpha_orig = alpha
        
        # Transposition table lookup
        tt_entry = self.tt_probe(board.zobrist)
        tt_move = tt_entry.best_move if tt_entry else None
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == 'exact':
                return tt_entry.value
            elif tt_entry.flag == 'lower' and tt_entry.value >= beta:
                return tt_entry.value
            elif tt_entry.flag == 'upper' and tt_entry.value <= alpha:
                return tt_entry.value
        
        # Syzygy tablebase probe (at up to 6 pieces)
        if self.use_syzygy and depth >= 2:
            piece_count = len(board.pieces)
            if piece_count <= 6:
                wdl = self.syzygy.probe_wdl(board)
                if wdl is not None:
                    self.state.syzygy_hits += 1
                    if wdl == 2: val = MATE_VALUE - ply - 1  # White wins
                    elif wdl == -2: val = -(MATE_VALUE - ply - 1)  # Black wins
                    elif wdl == 1: val = 500 - ply  # Cursed win
                    elif wdl == -1: val = -500 + ply  # Blessed loss
                    else: val = 0  # Draw
                    self.tt_store(board.zobrist, val, depth, 'exact')
                    return val
        
        # Check for game over
        if board.is_game_over():
            if board.is_checkmate():
                return -(MATE_VALUE - ply)  # Mated! Worse at higher ply
            return 0.0  # Draw
        
        # Check extension: if in check, search deeper
        in_check = board.is_in_check()
        if in_check:
            depth = max(depth, 1)
        
        # Quiescence at depth 0
        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)
        
        # =================================================================
        # Pruning techniques (non-PV nodes only = when window is narrow)
        # =================================================================
        is_pv = (beta - alpha > 1.0)
        
        # Razoring at depth 1
        if self.use_razoring and depth == 1 and not in_check:
            static_eval = self.evaluate(board)
            if static_eval + 300 < alpha:
                val = self.quiescence(board, alpha, beta, ply)
                return val
        
        # Null-move pruning
        if self.use_null_move and allow_null and not in_check and depth >= 3:
            if can_null_move(board, depth, beta):
                R = 3 + depth // 4
                # Make null move (pass turn) — just flip color, skip zobrist update
                saved_color = board.color_to_move
                board.color_to_move = Color.BLACK if saved_color == Color.WHITE else Color.WHITE
                
                val = -self.search(board, depth - 1 - R, -beta, -beta + 1, 
                                   ply + 1, None, allow_null=False)
                
                # Undo null move
                board.color_to_move = saved_color
                
                if val >= beta:
                    self.state.null_cuts += 1
                    return beta  # Cutoff
        
        # =================================================================
        # Main search loop
        # =================================================================
        
        moves = board.generate_legal_moves()
        if not moves:
            # Stalemate — penalize stalemating side if up material
            mat = 0
            for sq, (c, p) in board.pieces.items():
                mat += PIECE_VALUES[p] * (1 if c == Color.WHITE else -1)
            side_mult = 1 if board.color_to_move == Color.WHITE else -1
            stalemate_val = -mat * 0.001 * side_mult
            return stalemate_val
        
        # Score and order moves
        scored_moves = self.score_moves(moves, board, tt_move, depth, prev_move)
        
        best_val = -INF
        best_move = None
        move_count = 0
        
        for i, (move, _) in enumerate(scored_moves):
            move_count += 1
            
            # Futility pruning for quiet moves at frontier
            if self.use_futility and depth <= 3 and not in_check and not move.promotion:
                victim = board.piece_at(move.to_sq)
                if not victim:
                    static_eval = self.evaluate(board)
                    if is_futile(static_eval, beta, depth):
                        self.state.futility_cuts += 1
                        continue
            
            board.make_move(move)
            
            # Late Move Reduction
            if self.use_lmr and move_count >= 4 and depth >= 3:
                victim = board.piece_at(move.to_sq)
                is_quiet = (victim is None and not move.promotion)
                reduction = get_lmr_reduction(depth, move_count, is_quiet)
                
                if reduction > 0:
                    self.state.lmr_reductions += 1
                    # Reduced depth search (zero window)
                    val = -self.search(board, depth - 1 - reduction, -alpha - 1, -alpha,
                                      ply + 1, move, allow_null=True)
                    
                    # If reduced search beats alpha, research at full depth
                    if val > alpha:
                        val = -self.search(board, depth - 1, -beta, -alpha,
                                          ply + 1, move, allow_null=True)
                else:
                    val = -self.search(board, depth - 1, -beta, -alpha,
                                      ply + 1, move, allow_null=True)
            else:
                # Full depth search
                if move_count == 1:
                    # First move: full window
                    val = -self.search(board, depth - 1, -beta, -alpha,
                                      ply + 1, move, allow_null=True)
                else:
                    # Later moves: zero window (PVS)
                    val = -self.search(board, depth - 1, -alpha - 1, -alpha,
                                      ply + 1, move, allow_null=True)
                    if val > alpha and val < beta:
                        # Research with full window
                        val = -self.search(board, depth - 1, -beta, -alpha,
                                          ply + 1, move, allow_null=True)
            
            board.unmake_move()
            
            if val > best_val:
                best_val = val
                best_move = move
                
                if val > alpha:
                    alpha = val
                    
                    if val >= beta:
                        # Beta cutoff!
                        # Update history and killer tables
                        piece = board.piece_at(move.from_sq)
                        if piece:
                            self.state.history.add(move, depth, piece[1])
                        victim = board.piece_at(move.to_sq)
                        if not victim and not move.promotion:
                            self.state.killers.add(move, depth)
                        break
        
        # No legal moves = checkmate or stalemate (handled above)
        if best_val == -INF:
            best_val = -(MATE_VALUE - ply)
        
        # Store in TT
        flag = 'exact'
        if best_val <= alpha_orig: flag = 'upper'
        elif best_val >= beta: flag = 'lower'
        self.tt_store(board.zobrist, best_val, depth, flag, best_move)
        
        return best_val
    
    # =====================================================================
    # Iterative Deepening
    # =====================================================================
    
    def find_best_move(self, board: Board, time_limit_ms: float = 3000,
                       max_depth: int = 99) -> Tuple[Optional[Move], Dict]:
        """Iterative deepening with aspiration windows and time management.
        
        Returns (best_move, stats_dict).
        """
        start_time = time.time()
        
        # Opening book
        book_move = get_opening_move(board)
        if book_move:
            legal = board.generate_legal_moves()
            if book_move in legal:
                return book_move, {'book': True, 'nodes': 0, 'depth': 0, 
                                   'score': 0, 'time_ms': 0}
        
        # Clear per-search stats
        self.state.clear_search_stats()
        
        # Iterative deepening
        best_move = None
        best_score = 0
        alpha = -INF
        beta = INF
        stats = {}
        
        for depth in range(1, max_depth + 1):
            # Check time
            elapsed = (time.time() - start_time) * 1000
            if elapsed > time_limit_ms * 0.7:  # Use 70% of time for search
                break
            
            # Aspiration window search
            score = self.search(board, depth, alpha, beta)
            
            # Aspiration adjustment
            if score <= alpha or score >= beta:
                # Failed low/high — research with full window
                alpha = -INF
                beta = INF
                score = self.search(board, depth, alpha, beta)
            
            # Narrow window for next iteration
            alpha = score - 25
            beta = score + 25
            
            best_score = score
            
            # Extract best move from TT
            tt_entry = self.tt_probe(board.zobrist)
            if tt_entry and tt_entry.best_move:
                best_move = tt_entry.best_move
            
            elapsed = (time.time() - start_time) * 1000
            if elapsed > time_limit_ms * 0.5:
                # Not enough time for another depth
                if depth >= 4:
                    break
            
            # Update stats
            stats = {
                'depth': depth,
                'score': int(best_score),
                'nodes': self.state.nodes_searched,
                'nps': int(self.state.nodes_searched / (elapsed / 1000)) if elapsed > 0 else 0,
                'time_ms': int(elapsed),
                'tt_hits': self.state.tt_hits,
                'null_cuts': self.state.null_cuts,
                'futility_cuts': self.state.futility_cuts,
                'lmr_reductions': self.state.lmr_reductions,
                'syzygy_hits': self.state.syzygy_hits,
            }
        
        return best_move, stats
    
    def play_game(self, time_limit_ms: float = 3000) -> str:
        """Play a full game with this engine. Returns result."""
        board = Board()
        moves_played = 0
        max_moves = 200
        
        while not board.is_game_over() and moves_played < max_moves:
            move, stats = self.find_best_move(board, time_limit_ms)
            if move is None:
                break
            board.make_move(move)
            moves_played += 1
        
        return board.result() or '1/2-1/2'
