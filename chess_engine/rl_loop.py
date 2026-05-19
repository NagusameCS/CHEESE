"""
HyperTensor Chess — Fast Self-Play RL Launcher
================================================
Quick self-play training loop that runs alongside datagen/auto_train.
Uses the current model to play games against itself, records outcomes,
and trains to predict game results. This breaks the teacher ceiling.

Usage:
  python chess_engine/rl_loop.py --games 200 --model models/sf_autopilot_best.pt
"""

import torch
import torch.nn.functional as F
import numpy as np
import time, os, sys, math, random, argparse
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_engine.board import Board, Move, Color, Piece, STARTING_FEN, PIECE_VALUES
from chess_engine.evaluation import create_model, create_optimizer, count_parameters, CUDA_AVAILABLE, DEVICE
from chess_engine.pretrain import heuristic_evaluate


# ===========================================================================
# Fast Self-Play (heuristic eval for speed, model for final position scoring)
# ===========================================================================

class FastSelfPlay:
    """Play games quickly using heuristic eval for moves, model for scoring."""
    
    def __init__(self, model=None):
        self.model = model
    
    def play_game(self, max_moves: int = 200) -> List[Dict]:
        """Play a fast self-play game. Returns labeled positions."""
        board = Board()
        positions = []
        
        for move_count in range(max_moves):
            # Game over checks
            if board.is_checkmate():
                outcome = -1.0 if board.color_to_move == Color.WHITE else 1.0
                return self._label(positions, outcome)
            if board.is_stalemate():
                return self._label(positions, 0.0)
            # Insufficient material check (simple heuristic)
            piece_count = len(board.pieces)
            if piece_count <= 3 and not any(
                board.piece_at(sq) and board.piece_at(sq)[1] in (Piece.QUEEN, Piece.ROOK, Piece.PAWN)
                for sq in range(64)):
                return self._label(positions, 0.0)
            if board.halfmove_clock >= 100:
                return self._label(positions, 0.0)
            
            # Record position (skip early opening moves)
            if move_count >= 6 and len(positions) < 60:
                positions.append({
                    'tensor': board.to_tensor().astype(np.float32),
                    'side': board.color_to_move,
                })
            
            # Select move: use model eval if available, else heuristic
            move = self._select_move(board)
            if move is None:
                return self._label(positions, 0.0)
            
            board.make_move(move)
        
        return self._label(positions, 0.0)
    
    def _select_move(self, board: Board) -> Move:
        """Select a move using model evaluation on candidate positions."""
        moves = list(board.generate_legal_moves())
        if not moves:
            return None
        
        # For opening moves, use variety
        if len(board.pieces) > 28 and random.random() < 0.3:
            # Pick a reasonable developing move
            good = [m for m in moves if m.to_sq in (27, 28, 35, 36, 18, 21, 42, 45)  # center
                    or (board.piece_at(m.from_sq) and board.piece_at(m.from_sq)[1] > 1)]  # minor piece
            if good:
                return random.choice(good)
        
        if self.model is not None and len(moves) <= 40:
            # Evaluate all moves with model (batch)
            boards = []
            for m in moves:
                bc = board.copy()
                bc.make_move(m)
                boards.append(bc)
            
            tensors = np.stack([b.to_tensor().astype(np.float32) for b in boards])
            x = torch.from_numpy(tensors).to(DEVICE)
            with torch.inference_mode():
                vals, _, _, _ = self.model(x)
            
            scores = vals.squeeze(-1).cpu().numpy()
            
            # From perspective of side to move
            if board.color_to_move == Color.BLACK:
                scores = -scores
            
            # Temperature-based selection
            temperature = 0.5
            scores = scores - scores.max()
            probs = np.exp(scores / temperature)
            probs = probs / probs.sum()
            
            idx = np.random.choice(len(moves), p=probs)
            return moves[idx]
        
        # Fallback: heuristic selection
        best_move = moves[0]
        best_score = -99999
        for m in moves[:min(20, len(moves))]:
            bc = board.copy()
            bc.make_move(m)
            score, _, _ = heuristic_evaluate(bc)
            if board.color_to_move == Color.BLACK:
                score = -score
            if score > best_score:
                best_score = score
                best_move = m
        
        return best_move
    
    def _label(self, positions: List[Dict], outcome: float) -> List[Dict]:
        for p in positions:
            p['outcome'] = outcome if p['side'] == Color.WHITE else -outcome
        return positions


# ===========================================================================
# RL Trainer
# ===========================================================================

class RLLoop:
    """Self-play RL: play games, train on outcomes, repeat."""
    
    def __init__(self, model_path: str = None, model_size: str = 'xl',
                 games_per_cycle: int = 100, epochs_per_cycle: int = 5,
                 batch_size: int = 256, lr: float = 1e-4):
        self.games_per_cycle = games_per_cycle
        self.epochs_per_cycle = epochs_per_cycle
        self.batch_size = batch_size
        
        # Model
        from chess_engine.industrial_train import get_model_config
        mc = get_model_config(model_size)
        self.model = create_model(**mc).to(DEVICE)
        
        if model_path and Path(model_path).exists():
            ck = torch.load(model_path, map_location=DEVICE, weights_only=True)
            md = self.model.state_dict()
            pt = {k: v for k, v in ck.get('model_state_dict', ck).items() 
                  if k in md and md[k].shape == v.shape}
            md.update(pt)
            self.model.load_state_dict(md, strict=False)
            print(f"Loaded {len(pt)}/{len(md)} params")
        
        n = count_parameters(self.model)[1]
        print(f"Model: {n:,} params on {DEVICE}")
        
        self.optimizer = create_optimizer(self.model, lr=lr)
        self.player = FastSelfPlay(self.model)
        
        # Buffer
        self.buffer_tensors = []
        self.buffer_outcomes = []
        self.max_buffer = 50000
        
        # Stats
        self.cycle = 0
        self.total_games = 0
    
    def generate_games(self, n_games: int):
        """Play self-play games in parallel."""
        print(f"Playing {n_games} self-play games...")
        t0 = time.time()
        
        n_workers = min(4, n_games)
        games_per_worker = n_games // n_workers
        
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(self._play_games, games_per_worker) for _ in range(n_workers)]
            
            for i, f in enumerate(as_completed(futures)):
                positions = f.result()
                for p in positions:
                    self.buffer_tensors.append(p['tensor'])
                    self.buffer_outcomes.append(p['outcome'])
                
                if (i + 1) % max(1, n_workers // 2) == 0:
                    print(f"  {i+1}/{n_workers} workers done, "
                          f"buffer: {len(self.buffer_tensors)}")
        
        # Trim buffer
        if len(self.buffer_tensors) > self.max_buffer:
            keep = self.max_buffer
            self.buffer_tensors = self.buffer_tensors[-keep:]
            self.buffer_outcomes = self.buffer_outcomes[-keep:]
        
        self.total_games += n_games
        elapsed = time.time() - t0
        print(f"  {n_games} games in {elapsed:.0f}s, "
              f"buffer: {len(self.buffer_tensors)} positions")
    
    def _play_games(self, n: int) -> List[Dict]:
        results = []
        for _ in range(n):
            results.extend(self.player.play_game())
        return results
    
    def train_cycle(self):
        """One complete cycle: generate + train."""
        self.cycle += 1
        print(f"\n{'='*50}")
        print(f"CYCLE {self.cycle}")
        print(f"{'='*50}")
        
        # Generate
        self.generate_games(self.games_per_cycle)
        
        if len(self.buffer_tensors) < 100:
            print("Not enough data to train")
            return
        
        # Train
        print(f"Training on {len(self.buffer_tensors)} positions...")
        t0 = time.time()
        
        n = len(self.buffer_tensors)
        X = torch.from_numpy(np.stack(self.buffer_tensors)).float().to(DEVICE)
        y = torch.from_numpy(np.array(self.buffer_outcomes, dtype=np.float32)).to(DEVICE).unsqueeze(1)
        
        for epoch in range(self.epochs_per_cycle):
            self.model.train()
            perm = torch.randperm(n, device=DEVICE)
            losses = []
            
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb, yb = X[idx], y[idx]
                
                self.optimizer.zero_grad()
                vp, _, wdl_p, _ = self.model(xb)
                
                # MSE on value + CE on WDL
                v_loss = F.mse_loss(vp, yb)
                
                # Convert outcome to WDL
                wdl_target = torch.zeros(len(xb), 3, device=DEVICE)
                wdl_target[:, 0] = torch.clamp(yb.squeeze(), 0, 1)   # win
                wdl_target[:, 2] = torch.clamp(-yb.squeeze(), 0, 1)  # loss
                wdl_target[:, 1] = 1 - wdl_target[:, 0] - wdl_target[:, 2]  # draw
                w_loss = F.cross_entropy(wdl_p, wdl_target)
                
                loss = v_loss + 0.5 * w_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                self.optimizer.step()
                losses.append(loss.item())
            
            avg_loss = np.mean(losses)
            if (epoch + 1) % max(1, self.epochs_per_cycle // 2) == 0:
                print(f"  Epoch {epoch+1}/{self.epochs_per_cycle}: loss={avg_loss:.4f}")
        
        elapsed = time.time() - t0
        print(f"  Training: {elapsed:.0f}s, avg loss: {np.mean(losses):.4f}")
        
        # Save
        ckpt = Path('models/rl_loop_best.pt')
        ckpt.parent.mkdir(exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'cycle': self.cycle,
            'total_games': self.total_games,
        }, ckpt)
        print(f"  Saved: {ckpt}")
    
    def run(self, cycles: int = 20):
        print(f"\nSELF-PLAY RL LOOP")
        print(f"  Games per cycle: {self.games_per_cycle}")
        print(f"  Cycles: {cycles}")
        print(f"  Total games target: {self.games_per_cycle * cycles}")
        print(f"  Device: {DEVICE}")
        
        for _ in range(cycles):
            self.train_cycle()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--games', type=int, default=100)
    p.add_argument('--cycles', type=int, default=20)
    p.add_argument('--model', default='models/sf_autopilot_best.pt')
    p.add_argument('--model-size', default='xl')
    p.add_argument('--batch-size', type=int, default=256)
    args = p.parse_args()
    
    loop = RLLoop(
        model_path=args.model,
        model_size=args.model_size,
        games_per_cycle=args.games,
        batch_size=args.batch_size,
    )
    loop.run(cycles=args.cycles)
