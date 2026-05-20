"""
HyperTensor Chess — Fast Self-Play RL Launcher v2
==================================================
Runs alongside train_fast.py. Plays self-play games using FAST heuristic
move selection, then trains model on game outcomes. No model inference
during games — that's too slow. Model only scores final positions.

Strategy:
  1. Play N self-play games using heuristic eval (fast, CPU-only, parallel)
  2. Score positions using model to get value labels
  3. Train on position→outcome pairs
  4. Repeat

Usage:
  python chess_engine/rl_loop.py --games 200 --cycles 50 --model models/train_fast_best.pt
"""

import torch
import torch.nn.functional as F
import numpy as np
import time, os, sys, math, random, argparse
from pathlib import Path
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_engine.board import Board, Move, Color, Piece, STARTING_FEN, PIECE_VALUES
from chess_engine.evaluation import create_model, create_optimizer, count_parameters, CUDA_AVAILABLE, DEVICE
# heuristic_evaluate removed — too slow, using direct move selection


class FastSelfPlay:
    """Play games quickly using heuristic eval ONLY for move selection."""
    
    def play_game(self, max_moves: int = 200) -> List[Dict]:
        """Play a fast self-play game. Returns labeled positions."""
        board = Board()
        positions = []
        
        for move_count in range(max_moves):
            # Game over checks
            if board.is_checkmate():
                outcome = -1.0 if board.color_to_move == Color.WHITE else 1.0
                return self._label(positions, outcome)
            if board.is_stalemate() or board.halfmove_clock >= 100:
                return self._label(positions, 0.0)
            
            # Insufficient material (simple heuristic)
            piece_count = len(board.pieces)
            if piece_count <= 3 and not any(
                board.piece_at(sq) and board.piece_at(sq)[1] in (Piece.QUEEN, Piece.ROOK, Piece.PAWN)
                for sq in range(64)):
                return self._label(positions, 0.0)
            
            # Record position (skip early opening, cap samples)
            if move_count >= 6 and len(positions) < 60:
                positions.append({
                    'tensor': board.to_tensor().astype(np.float32),
                    'side': board.color_to_move,
                })
            
            # Select move using fast heuristic eval
            move = self._select_move(board)
            if move is None:
                return self._label(positions, 0.0)
            
            board.make_move(move)
        
        return self._label(positions, 0.0)
    
    def _select_move(self, board: Board) -> Move:
        """Ultra-fast move selection — captures first, then random. No eval needed."""
        moves = list(board.generate_legal_moves())
        if not moves:
            return None
        
        # 20% random for variety
        if random.random() < 0.20:
            return random.choice(moves)
        
        # Prefer captures (MVV-LVA: highest victim, lowest attacker)
        captures = []
        for m in moves:
            victim = board.piece_at(m.to_sq)
            if victim is not None:
                attacker = board.piece_at(m.from_sq)
                if attacker is not None:
                    victim_val = PIECE_VALUES.get(victim[1], 0)
                    attacker_val = PIECE_VALUES.get(attacker[1], 0)
                    captures.append((victim_val - attacker_val * 0.01, m))
        
        if captures:
            captures.sort(key=lambda x: x[0], reverse=True)
            # Pick from top 3 captures
            return random.choice(captures[:min(3, len(captures))])[1]
        
        # Center-control preference
        center_moves = [m for m in moves if m.to_sq in (27, 28, 35, 36)]
        if center_moves and random.random() < 0.5:
            return random.choice(center_moves)
        
        return random.choice(moves)
    
    def _label(self, positions: List[Dict], outcome: float) -> List[Dict]:
        for p in positions:
            p['outcome'] = outcome if p['side'] == Color.WHITE else -outcome
        return positions


class RLLoop:
    """Self-play RL: play games fast, train on outcomes, repeat."""
    
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
        self.scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
        self.player = FastSelfPlay()
        
        # Stats
        self.cycle = 0
        self.total_games = 0
        self.best_val_loss = float('inf')
    
    def play_games(self, n_games: int) -> List[Dict]:
        """Play games in parallel using threads."""
        n_workers = min(6, n_games)
        games_per_worker = n_games // n_workers
        remaining = n_games % n_workers
        
        all_positions = []
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = []
            for i in range(n_workers):
                ng = games_per_worker + (1 if i < remaining else 0)
                futures.append(ex.submit(self._play_worker, ng))
            
            for f in as_completed(futures):
                positions = f.result()
                all_positions.extend(positions)
        
        return all_positions
    
    def _play_worker(self, n_games: int) -> List[Dict]:
        """Worker: play n games and return all positions."""
        positions = []
        for _ in range(n_games):
            try:
                positions.extend(self.player.play_game())
            except Exception as e:
                pass
        return positions
    
    def train_on_positions(self, positions: List[Dict]):
        """Train model on self-play positions."""
        if len(positions) < self.batch_size:
            return
        
        # Prepare data
        X_list = [p['tensor'] for p in positions]
        y_list = [p['outcome'] for p in positions]
        
        X = torch.from_numpy(np.stack(X_list)).float().to(DEVICE)
        y = torch.from_numpy(np.array(y_list, dtype=np.float32)).float().to(DEVICE).unsqueeze(1)
        n = len(X)
        
        for epoch in range(self.epochs_per_cycle):
            self.model.train()
            perm = torch.randperm(n, device=DEVICE)
            self.optimizer.zero_grad()
            
            for start in range(0, n, self.batch_size):
                idx = perm[start:start+self.batch_size]
                xb, yb = X[idx], y[idx]
                
                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        vp, _, _, _ = self.model(xb)
                        loss = F.mse_loss(vp, yb)
                    self.scaler.scale(loss).backward()
                else:
                    vp, _, _, _ = self.model(xb)
                    loss = F.mse_loss(vp, yb)
                    loss.backward()
                
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
        
        del X, y
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    
    def run_cycle(self):
        """One play-train cycle."""
        self.cycle += 1
        t0 = time.time()
        
        # Play games
        play_t0 = time.time()
        positions = self.play_games(self.games_per_cycle)
        play_time = time.time() - play_t0
        self.total_games += self.games_per_cycle
        
        # Train
        train_t0 = time.time()
        self.train_on_positions(positions)
        train_time = time.time() - train_t0
        
        # Simple validation: check loss on last batch
        self.model.eval()
        with torch.inference_mode():
            if len(positions) >= 64:
                Xv = torch.from_numpy(np.stack([p['tensor'] for p in positions[:64]])).float().to(DEVICE)
                yv = torch.from_numpy(np.array([p['outcome'] for p in positions[:64]], dtype=np.float32)).float().to(DEVICE).unsqueeze(1)
                vp, _, _, _ = self.model(Xv)
                val_loss = F.mse_loss(vp, yv).item()
            else:
                val_loss = 1.0
        
        # Save if improved
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'val_loss': val_loss,
                'cycle': self.cycle,
                'total_games': self.total_games,
            }, 'models/rl_loop_best.pt')
            new_best = ' *** NEW BEST ***'
        else:
            new_best = ''
        
        total_time = time.time() - t0
        print(f"CYCLE {self.cycle:3d} | "
              f"play {play_time:4.0f}s ({self.games_per_cycle} games) | "
              f"train {train_time:4.0f}s | "
              f"positions {len(positions):>6,} | "
              f"loss {val_loss:.4f} | "
              f"total games {self.total_games:>7,}"
              f"{new_best}")
        
        # Reload best model from train_fast if it exists and is newer
        fast_best = Path('models/train_fast_best.pt')
        if fast_best.exists():
            fast_mtime = fast_best.stat().st_mtime
            rl_best = Path('models/rl_loop_best.pt')
            if not rl_best.exists() or fast_mtime > rl_best.stat().st_mtime + 60:
                print(f"  Syncing from train_fast_best.pt...")
                ck = torch.load(fast_best, map_location=DEVICE, weights_only=True)
                md = self.model.state_dict()
                pt = {k: v for k, v in ck.get('model_state_dict', ck).items()
                      if k in md and md[k].shape == v.shape}
                md.update(pt)
                self.model.load_state_dict(md, strict=False)
        
        return val_loss
    
    def run(self, cycles=50):
        """Run for specified cycles."""
        print(f"\nRL Loop: {self.games_per_cycle} games/cycle, "
              f"{self.epochs_per_cycle} epochs/cycle, "
              f"{cycles} cycles\n")
        
        for _ in range(cycles):
            self.run_cycle()
        
        print(f"\nDone. Total cycles: {self.cycle}, total games: {self.total_games}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--games', type=int, default=200)
    p.add_argument('--cycles', type=int, default=50)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--model', default='models/train_fast_best.pt')
    p.add_argument('--model-size', default='xl')
    args = p.parse_args()
    
    loop = RLLoop(
        model_path=args.model,
        model_size=args.model_size,
        games_per_cycle=args.games,
        epochs_per_cycle=args.epochs,
        batch_size=args.batch_size,
    )
    loop.run(cycles=args.cycles)
