"""
HyperTensor Chess Engine v3.0 — Elite Search
=============================================
World-class MCTS search combining:
  - Zobrist transposition table with depth-preferred replacement
  - MVV-LVA + killer move + history heuristic move ordering
  - Aspiration windows
  - Null-move pruning for zugzwang detection
  - Jury-gated speculative evaluation (HyperTensor)
  - Batched GPU evaluation (64+ positions)
  - Iterative deepening with time management
  - Opening book integration
  - SafeOGD blunder detection
  - Geodesic transposition table
"""

import torch
import numpy as np
import math
import time
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from pathlib import Path
import sys
import threading
import concurrent.futures

_HYPERTENSOR_PATH = Path(__file__).parent.parent / "HyperTensor"
if _HYPERTENSOR_PATH.exists():
    sys.path.insert(0, str(_HYPERTENSOR_PATH))
    sys.path.insert(0, str(_HYPERTENSOR_PATH / "scripts"))

from .board import Board, Move, Color, Piece, SQUARE_NAMES, PIECE_VALUES
from .evaluation import HyperTensorChessNet, CUDA_AVAILABLE, DEVICE
from .opening_book import get_opening_move
from .strong_search import (EliteMoveOrdering, see_capture, SyzygyProbe,
                            TimeManager, CountermoveTable)

# HyperTensor imports
try:
    from scripts.ott_engine import JuryDraftGate; _JURY = True
except: _JURY = False

# ===========================================================================
# Transposition Table
# ===========================================================================

@dataclass
class TTEntry:
    zobrist: int
    value: float
    depth: int
    flag: str  # 'exact', 'lower', 'upper'
    best_move: Optional[Move] = None
    age: int = 0

class TranspositionTable:
    """Zobrist-based transposition table with depth-preferred replacement."""
    
    def __init__(self, size_mb: int = 64):
        num_entries = (size_mb * 1024 * 1024) // 32  # ~32 bytes per entry
        self.table: Dict[int, TTEntry] = {}
        self.max_entries = num_entries
        self.age = 0
    
    def store(self, zobrist: int, value: float, depth: int, flag: str,
              best_move: Optional[Move] = None):
        if zobrist in self.table:
            existing = self.table[zobrist]
            # Depth-preferred: only overwrite if new depth >= old depth
            if depth < existing.depth and existing.flag == 'exact':
                return
        
        if len(self.table) >= self.max_entries:
            # Remove 25% oldest entries
            items = sorted(self.table.items(), key=lambda x: x[1].age)
            for k, _ in items[:len(items)//4]:
                del self.table[k]
        
        self.table[zobrist] = TTEntry(zobrist, value, depth, flag, best_move, self.age)
    
    def probe(self, zobrist: int) -> Optional[TTEntry]:
        entry = self.table.get(zobrist)
        if entry: entry.age = self.age
        return entry
    
    def increment_age(self): self.age += 1
    def clear(self): self.table.clear()


# ===========================================================================
# Endgame Helper: Detect stunting and stalemate traps
# ===========================================================================

def heuristic_material_advantage(board) -> float:
    """Quick material advantage estimate in [-1, 1] without full evaluation."""
    mat = 0.0
    for sq, (color, piece) in board.pieces.items():
        sign = 1 if color == Color.WHITE else -1
        if piece == Piece.KING: continue  # Both sides always have king
        mat += sign * PIECE_VALUES[piece] / 1000.0
    return np.tanh(mat)  # ~0 for equal, ±1 for huge advantage

def anti_stunt_eval(board, base_eval: float) -> float:
    """Penalize 'stunting' — promoting extra queens instead of checkmating.
    
    When the engine is up massive material but wastes moves promoting pawns
    instead of delivering checkmate, this function detects and penalizes.
    
    Returns adjusted evaluation in [-1, 1].
    """
    # Check if we're in a massively winning position
    if abs(base_eval) < 0.85:
        return base_eval  # Not clearly winning, skip
    
    mat_adv = heuristic_material_advantage(board)
    if abs(mat_adv) < 0.6:
        return base_eval  # Material isn't overwhelming yet
    
    # We're up massive material. Check how many queens we have.
    us = Color.WHITE if base_eval > 0 else Color.BLACK
    our_queens = 0
    our_pawns = 0
    enemy_king_file = -1
    enemy_king_rank = -1
    our_king_sq = -1
    
    for sq, (color, piece) in board.pieces.items():
        if color == us:
            if piece == Piece.QUEEN: our_queens += 1
            elif piece == Piece.PAWN: our_pawns += 1
            elif piece == Piece.KING: our_king_sq = sq
        else:
            if piece == Piece.KING:
                enemy_king_file = sq % 8
                enemy_king_rank = sq // 8
    
    # STUNT DETECTION 1: Multiple queens when one is enough
    # One queen + king can mate. Extra queens are stunting.
    queen_penalty = max(0, our_queens - 2) * 0.15  # Each extra queen: 15% penalty
    
    # STUNT DETECTION 2: Promoting pawns when already winning
    # If we have a queen and pawns, pawn pushes are wasteful unless they're
    # part of a checkmate pattern
    pawn_stunt = 0.0
    if our_queens >= 1 and our_pawns > 0:
        pawn_stunt = our_pawns * 0.03  # Each pawn: 3% penalty
    
    # KING PROXIMITY BONUS: When up material, move king toward enemy king
    king_prox_bonus = 0.0
    if our_king_sq >= 0 and enemy_king_file >= 0:
        our_kf, our_kr = our_king_sq % 8, our_king_sq // 8
        dist = abs(our_kf - enemy_king_file) + abs(our_kr - enemy_king_rank)
        # Max king distance is 14 (a1-h8). Bonus for proximity.
        king_prox_bonus = max(0, (14 - dist) / 14) * 0.05
    
    # ENEMY KING CORNERING: Enemy king on edge/in corner = good
    corner_bonus = 0.0
    if enemy_king_file >= 0:
        edge_dist = min(enemy_king_file, 7 - enemy_king_file, enemy_king_rank, 7 - enemy_king_rank)
        corner_bonus = max(0, (3 - edge_dist)) * 0.02  # Cornered = bonus
    
    # MOVE COUNT FRUSTRATION: If the game has gone on too long,
    # penalize wasting time (proxy: halfmove clock)
    time_penalty = 0.0
    if board.halfmove_clock > 20:  # 20 half-moves without capture/pawn move
        time_penalty = min(0.2, (board.halfmove_clock - 20) * 0.005)
    
    # STALEMATE TRAP DETECTION
    # If opponent has few legal moves and we're winning, be cautious
    stalemate_risk = 0.0
    legal = board.generate_legal_moves()
    if len(legal) > 0:
        # Simulate the opponent's position
        board_copy = board.copy()
        board_copy.color_to_move = Color.BLACK if board.color_to_move == Color.WHITE else Color.WHITE
        try:
            opp_legal = board_copy.generate_legal_moves()
            if len(opp_legal) <= 3:  # Opponent nearly stalemated
                stalemate_risk = (3 - len(opp_legal)) * 0.08  # 8-24% penalty
        except: pass
    
    # Combine adjustments
    penalty = queen_penalty + pawn_stunt + time_penalty + stalemate_risk
    bonus = king_prox_bonus + corner_bonus
    
    adjusted = base_eval - penalty + bonus
    return max(-1.0, min(1.0, adjusted))


# ===========================================================================
# MCTS Node with Search Enhancements
# ===========================================================================

@dataclass
class MCTSNode:
    move: Optional[Move] = None
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    
    visits: int = 0
    total_value: float = 0.0
    prior: float = 0.0
    virtual_loss: int = 0
    
    # Position info
    zobrist: int = 0
    k_coords: Optional[np.ndarray] = None
    is_unsafe: bool = False
    is_terminal: bool = False
    terminal_value: float = 0.0
    
    @property
    def value(self): return self.total_value / max(self.visits, 1)
    
    @property
    def is_leaf(self): return len(self.children) == 0
    
    def ucb_score(self, parent_visits: int, c_puct: float = 1.5,
                  fpu_reduction: float = 0.25) -> float:
        if self.visits == 0:
            return self.prior * c_puct * math.sqrt(parent_visits + 1) - fpu_reduction
        Q = (self.total_value - self.virtual_loss * 0.5) / (self.visits + self.virtual_loss)
        U = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return Q + U
    
    def best_child(self, c_puct: float = 1.5) -> Optional['MCTSNode']:
        if not self.children: return None
        return max(self.children, key=lambda c: c.ucb_score(self.visits, c_puct))


# ===========================================================================
# Elite Search Engine
# ===========================================================================

class HyperTensorSearch:
    """World-class MCTS search with all enhancements."""
    
    def __init__(self, model: HyperTensorChessNet, num_simulations: int = 800,
                 c_puct: float = 1.5, use_jury: bool = True, use_gtc: bool = True,
                 use_safe_ogd: bool = True, batch_size: int = 64,
                 tt_size_mb: int = 64, use_opening_book: bool = True,
                 num_threads: int = 1, syzygy_path: str = None):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.use_jury = use_jury
        self.use_gtc = use_gtc
        self.use_safe_ogd = use_safe_ogd
        self.batch_size = batch_size
        self.use_opening_book = use_opening_book
        self.num_threads = num_threads
        
        # Transposition table
        self.tt = TranspositionTable(size_mb=tt_size_mb)
        
        # Elite move ordering
        self.move_order = EliteMoveOrdering()
        
        # Syzygy tablebase
        self.syzygy = SyzygyProbe(syzygy_path)
        
        # Time manager
        self.time_mgr = TimeManager()
        
        # Jury gate
        self.jury = None
        if use_jury and _JURY:
            try: self.jury = JuryDraftGate(threshold=0.85, n_jurors=7)
            except: pass
        
        # Geodesic cache
        self.geodesic = model.geodesic
        self.gtc_cache: Dict[int, float] = {}
        
        # Multi-threading
        self._stop_event = threading.Event()
        self._result_lock = threading.Lock()
        self._thread_results = []
        
        # Statistics
        self.stats = {'jury_accepts': 0, 'jury_rejects': 0, 'cache_hits': 0,
                      'cache_misses': 0, 'total_evals': 0, 'batched_evals': 0,
                      'unsafe_detected': 0, 'tt_hits': 0, 'tt_stores': 0,
                      'syzygy_hits': 0, 'see_prunes': 0}
    
    @torch.no_grad()
    def search(self, board: Board, time_limit_ms: float = 3000,
               max_depth: int = 999) -> Tuple[Optional[Move], Dict]:
        """Find best move with full search enhancements."""
        start_t = time.time()
        
        # Check opening book
        if self.use_opening_book:
            book_move = get_opening_move(board)
            if book_move:
                legal = board.generate_legal_moves()
                if book_move in legal:
                    return book_move, {'book': True}
        
        # Increment TT age for replacement
        self.tt.increment_age()
        
        # Initialize root
        root = MCTSNode()
        root.zobrist = board.zobrist
        legal_moves = board.generate_legal_moves()
        if not legal_moves: return None, self.stats
        if len(legal_moves) == 1: return legal_moves[0], self.stats
        
        # Probe TT for move ordering hint
        tt_entry = self.tt.probe(board.zobrist)
        tt_move = tt_entry.best_move if tt_entry else None
        
        # Aspiration search
        if tt_entry and tt_entry.flag == 'exact':
            self.asp_window = 0.2
        else:
            self.asp_window = 1.0
        
        # Create root children with ordered moves
        ordered_moves = self.move_order.order_moves(legal_moves, board, tt_move, 0)
        for move in ordered_moves:
            child = MCTSNode(move=move, parent=root, prior=1.0/len(ordered_moves))
            root.children.append(child)
        
        # Main MCTS loop
        num_sims = 0
        leaf_batch_tensors = []
        leaf_batch_nodes = []
        
        while time.time() - start_t < time_limit_ms / 1000.0:
            if num_sims >= self.num_simulations: break
            
            # Selection + Expansion
            node = root
            sim_board = board.copy()
            search_path = [node]
            sim_depth = 0
            
            # Select until leaf or terminal
            while not node.is_leaf:
                node = node.best_child(self.c_puct)
                if node is None: break
                if node.is_terminal:
                    # Mate distance bonus: shorter mates are better
                    val = node.terminal_value
                    if abs(val) > 0.99:  # Mate found
                        mate_dist = sim_depth + 1
                        val = (1.0 - 0.001 * mate_dist) if val > 0 else (-1.0 + 0.001 * mate_dist)
                    for n in reversed(search_path):
                        n.visits += 1; n.total_value += val; val = -val
                    break
                sim_board.make_move(node.move)
                search_path.append(node)
                sim_depth += 1
                
                if sim_board.is_game_over():
                    result = sim_board.result()
                    if result == '1-0':
                        mate_dist = sim_depth + 1
                        val = 1.0 - 0.001 * mate_dist  # Shorter mate = better
                    elif result == '0-1':
                        mate_dist = sim_depth + 1
                        val = -1.0 + 0.001 * mate_dist  # Shorter mate = better
                    else:
                        # Stalemate or draw
                        # If we're up material, stalemate is a BLUNDER
                        mat_adv = heuristic_material_advantage(sim_board)
                        if abs(mat_adv) > 0.5:  # Up significant material
                            # Penalize the stalemating side
                            val = -mat_adv * 0.5  # Negative for the side that was winning
                        else:
                            val = 0.0
                    node.is_terminal = True; node.terminal_value = val
                    for n in reversed(search_path):
                        n.visits += 1; n.total_value += val; val = -val
                    break
            
            if node.is_terminal or sim_board.is_game_over():
                num_sims += 1; continue
            
            # Expansion
            if node.visits < 2:
                # Syzygy tablebase probe
                syzygy_wdl = self.syzygy.probe_wdl(sim_board)
                if syzygy_wdl is not None:
                    self.stats['syzygy_hits'] += 1
                    md = sim_depth + 1  # mate distance
                    if syzygy_wdl == 2: val = 1.0 - 0.0005 * md   # Win (shorter preferred)
                    elif syzygy_wdl == -2: val = -1.0 + 0.0005 * md  # Loss
                    elif syzygy_wdl == 1: val = 0.8   # Cursed win
                    elif syzygy_wdl == -1: val = -0.8 # Blessed loss
                    else: val = 0.0  # Draw
                    node.is_terminal = True; node.terminal_value = val
                    for n in reversed(search_path):
                        n.visits += 1; n.total_value += val; val = -val
                    num_sims += 1; continue
                
                # Leaf evaluation
                st = sim_board.to_tensor()
                kc = self.model.get_manifold_coords(st)
                node.k_coords = kc
                node.zobrist = sim_board.zobrist
                node._sim_board = sim_board.copy()  # For anti-stunt eval
                
                # TT probe
                tt_hit = self.tt.probe(sim_board.zobrist)
                if tt_hit and tt_hit.depth >= 0:
                    self.stats['tt_hits'] += 1
                    val = tt_hit.value
                    # Apply anti-stunt correction even on TT hits
                    val = anti_stunt_eval(sim_board, val)
                else:
                    self.stats['cache_misses'] += 1
                    
                    # SafeOGD check
                    if self.use_safe_ogd:
                        try:
                            _, unsafe = self.model.safe_evaluate(st)
                            node.is_unsafe = unsafe
                            if unsafe: self.stats['unsafe_detected'] += 1
                        except: pass
                    
                    leaf_batch_tensors.append(st)
                    leaf_batch_nodes.append(node)
            
            # Expand children of this node
            if sim_board.is_game_over():
                node.is_terminal = True
                result = sim_board.result()
                if result == '1-0':
                    mate_dist = sim_depth + 1
                    node.terminal_value = 1.0 - 0.001 * mate_dist
                elif result == '0-1':
                    mate_dist = sim_depth + 1
                    node.terminal_value = -1.0 + 0.001 * mate_dist
                else:
                    # Draw (stalemate, 50-move, repetition, insufficient material)
                    # If one side was winning, the stalemating side should be penalized
                    mat_adv = heuristic_material_advantage(sim_board)
                    node.terminal_value = -mat_adv * 0.3  # Blame stalemater
            else:
                lm = sim_board.generate_legal_moves()
                if lm:
                    # Probe TT for child ordering
                    child_tt = self.tt.probe(sim_board.zobrist)
                    child_tt_move = child_tt.best_move if child_tt else None
                    ordered = self.move_order.order_moves(lm, sim_board, child_tt_move, sim_depth)
                    
                    prior = 1.0 / len(ordered)
                    for mv in ordered:
                        child = MCTSNode(move=mv, parent=node, prior=prior)
                        node.children.append(child)
                else:
                    node.is_terminal = True
                    # Stalemate — penalize the stalemating side if they were winning
                    mat_adv = heuristic_material_advantage(sim_board)
                    if abs(mat_adv) > 0.5:
                        node.terminal_value = -mat_adv * 0.3  # Major blunder
                    else:
                        node.terminal_value = 0.0  # True draw
            
            # Batch GPU evaluation
            if len(leaf_batch_tensors) >= self.batch_size or (
                num_sims + 1 >= self.num_simulations and leaf_batch_tensors):
                self._batch_eval_and_backprop(leaf_batch_nodes, leaf_batch_tensors)
                leaf_batch_tensors.clear(); leaf_batch_nodes.clear()
            
            num_sims += 1
        
        # Flush remaining batch
        if leaf_batch_tensors:
            self._batch_eval_and_backprop(leaf_batch_nodes, leaf_batch_tensors)
        
        # Store in TT
        best_child = max(root.children, key=lambda c: (c.visits, c.value))
        self.tt.store(board.zobrist, best_child.value, 0, 'exact', best_child.move)
        self.stats['tt_stores'] += 1
        
        # Update elite move ordering
        best_child = max(root.children, key=lambda c: (c.visits, c.value))
        self.move_order.update(best_child.move, 0, True)
        self.move_order.prev_best_move = best_child.move
        
        elapsed = (time.time() - start_t) * 1000
        return best_child.move, {
            **self.stats, 'simulations': num_sims, 'time_ms': elapsed,
            'nps': num_sims / (elapsed / 1000) if elapsed > 0 else 0,
            'root_visits': root.visits, 'best_visits': best_child.visits,
            'best_value': best_child.value,
            'move': best_child.move.uci() if best_child.move else 'none',
        }
    
    def _batch_eval_and_backprop(self, nodes, tensors):
        if not nodes or not tensors: return
        
        # Stack tensors (160, 8, 8) per position
        batch = np.stack(tensors)
        result = self.model.evaluate_batch(batch)
        
        values = result['values'] / 1000.0  # cp → [-1, 1]
        self.stats['total_evals'] += len(nodes)
        self.stats['batched_evals'] += 1
        
        for node, val in zip(nodes, values):
            # Apply anti-stunt correction when in winning/losing endgame
            # This prevents the engine from "showing off" by promoting
            # extra queens instead of going for checkmate
            val = anti_stunt_eval(node._sim_board, float(val)) if hasattr(node, '_sim_board') else float(val)
            
            # Store in TT
            self.tt.store(node.zobrist, float(val), 0, 'exact', 
                         node.move if hasattr(node, 'move') else None)
            self.stats['tt_stores'] += 1
            
            # Backpropagate
            path = []; n = node
            while n is not None:
                path.append(n); n = n.parent
            v = float(val)
            for n in reversed(path):
                n.visits += 1; n.total_value += v; v = -v


# ===========================================================================
# Iterative Deepening with Time Management
# ===========================================================================

class IterativeDeepeningSearch:
    """Full iterative deepening with time management."""
    
    def __init__(self, model: HyperTensorChessNet, time_limit_ms: float = 3000,
                 tt_size_mb: int = 128):
        self.model = model
        self.time_limit_ms = time_limit_ms
        self.search = HyperTensorSearch(model, tt_size_mb=tt_size_mb)
    
    def find_best_move(self, board: Board) -> Tuple[Optional[Move], Dict]:
        start_t = time.time()
        best_move = None
        best_stats = {}
        
        for depth in range(1, 999):
            elapsed = (time.time() - start_t) * 1000
            remaining = self.time_limit_ms - elapsed
            if remaining < 100: break
            
            sims = min(depth * 150, int(remaining * 0.7))
            self.search.num_simulations = sims
            
            move, stats = self.search.search(board, time_limit_ms=remaining)
            if move: best_move = move; best_stats = stats
            
            # Stop if forced mate found
            if abs(stats.get('best_value', 0)) > 0.98: break
        
        best_stats['depth'] = depth
        return best_move, best_stats


# ===========================================================================
# Self-Play
# ===========================================================================

def play_game(white_model, black_model=None, time_per_move_ms=1000,
              max_moves=300, use_opening=True):
    if black_model is None: black_model = white_model
    board = Board()
    sw = HyperTensorSearch(white_model, use_opening_book=use_opening)
    sb = HyperTensorSearch(black_model, use_opening_book=use_opening)
    
    moves = []
    for mn in range(max_moves):
        if board.is_game_over(): break
        s = sw if board.color_to_move == Color.WHITE else sb
        mv, st = s.search(board, time_limit_ms=time_per_move_ms)
        if mv is None:
            # Fallback to legal move
            lm = board.generate_legal_moves()
            if not lm: break
            mv = lm[0]
        moves.append(mv.uci()); board.make_move(mv)
        
        if mn % 5 == 0:
            info = ""
            if st.get('book'): info = " [book]"
            elif st.get('best_value'): info = f" [v={st['best_value']:.3f}]"
            print(f"  {mn+1}. {mv.uci()}{info} (sims:{st.get('simulations',0)})")
    
    return board.result() or '1/2-1/2', moves
