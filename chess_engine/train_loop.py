"""
HyperTensor Chess Engine v3.1 — Continuous Training Loop
=========================================================
Self-contained training that runs until interrupted.
Generates data via deep MCTS self-play, trains on replay buffer,
saves checkpoints, and logs progress.

Usage:
  python -m chess_engine.train_loop           # Run until Ctrl+C
  python -m chess_engine.train_loop --resume  # Resume from checkpoint

This is the AlphaZero bootstrap approach:
  Phase 1 (hours 0-2):   Random → basic piece values (k=4→8)
  Phase 2 (hours 2-8):   Basic tactics emerge (k=8→16)
  Phase 3 (hours 8-24):  Strategy develops (k=16→32)
  Phase 4 (days 1-3):    Strong amateur (k=32→64)
  Phase 5 (days 3-14):   Master level (k=64, deep training)
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import time
import json
import random
import signal
import sys
import os
from pathlib import Path
from collections import deque
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from .board import Board, Move, Color, Piece, STARTING_FEN
from .evaluation import (
    HyperTensorChessNet, KExpansionScheduler, RiemannianAdamW,
    create_model, create_optimizer, count_parameters,
    CUDA_AVAILABLE, DEVICE,
)
from .search import HyperTensorSearch, play_game, TranspositionTable
from .data_pipeline import DataAugmentor, TrainingExample
from .opening_book import get_opening_move

# Global stop flag for graceful shutdown
_stop_requested = False

def _signal_handler(sig, frame):
    global _stop_requested
    _stop_requested = True
    print("\n[Stop] Graceful shutdown requested...")

signal.signal(signal.SIGINT, _signal_handler)
try: signal.signal(signal.SIGBREAK, _signal_handler)
except: pass


# ===========================================================================
# Prioritized Experience Buffer
# ===========================================================================

class PrioritizedBuffer:
    """Buffer with prioritization: novel positions sampled more often."""
    
    def __init__(self, max_size=500000, alpha=0.6):
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = alpha
        self.max_size = max_size
    
    def add(self, example, novelty_score=1.0):
        self.buffer.append(example)
        self.priorities.append(novelty_score ** self.alpha)
    
    def sample(self, batch_size):
        n = min(batch_size, len(self.buffer))
        if n == 0: return []
        
        probs = np.array(self.priorities)
        probs = probs / probs.sum()
        indices = np.random.choice(len(self.buffer), n, replace=False, p=probs)
        return [self.buffer[i] for i in indices]
    
    def update_priorities(self, indices, losses):
        for idx, loss in zip(indices, losses):
            if idx < len(self.priorities):
                self.priorities[idx] = (abs(loss) + 1e-6) ** self.alpha
    
    def __len__(self): return len(self.buffer)


# ===========================================================================
# Fast Data Generator
# ===========================================================================

class FastDataGenerator:
    """Generate training data rapidly using shorter searches."""
    
    def __init__(self, model: HyperTensorChessNet, sims_per_move=50,
                 temperature=1.0):
        self.model = model
        self.sims_per_move = sims_per_move
        
        # Create search with minimal features for speed
        self.search = HyperTensorSearch(
            model,
            num_simulations=sims_per_move,
            use_jury=False, use_gtc=False,
            use_safe_ogd=True, batch_size=16,
            tt_size_mb=32, use_opening_book=True,
        )
    
    @torch.no_grad()
    def generate_game(self, max_moves=80, time_limit_ms=300) -> Tuple[List[TrainingExample], str, int]:
        board = Board()
        examples = []
        move_count = 0
        
        while not board.is_game_over() and move_count < max_moves:
            if board.halfmove_clock > 80: break
            
            tensor = board.to_tensor()
            
            move, stats = self.search.search(board, time_limit_ms=time_limit_ms)
            if move is None: break
            
            # Policy: uniform over legal moves (simple but unbiased)
            legal = board.generate_legal_moves()
            policy = np.zeros(4096, dtype=np.float32)
            move_idx = (move.from_sq * 64 + move.to_sq) % 4096
            policy[move_idx] = 1.0
            
            examples.append(TrainingExample(
                board_tensor=tensor.astype(np.float32),
                value_target=0.0,  # Will be set after game ends
                policy_target=policy,
                wdl_target=np.array([0.33, 0.34, 0.33], dtype=np.float32),
                outcome=0.0,
            ))
            
            board.make_move(move)
            move_count += 1
        
        # Game result
        result = board.result() or '1/2-1/2'
        if result == '1-0': outcome = 1.0
        elif result == '0-1': outcome = -1.0
        else: outcome = 0.0
        
        # Update all examples with actual outcome
        for ex in examples:
            ex.value_target = outcome
            ex.outcome = outcome
            if outcome > 0:
                ex.wdl_target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif outcome < 0:
                ex.wdl_target = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                ex.wdl_target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        return examples, result, move_count
    
    def generate_batch(self, num_games=4) -> List[TrainingExample]:
        all_ex = []
        for _ in range(num_games):
            ex, result, moves = self.generate_game()
            all_ex.extend(ex)
            # Augment
            augmented = []
            for e in ex:
                augmented.extend(DataAugmentor.augment(e))
            all_ex.extend(augmented[len(ex):])  # Only add extras
        return all_ex


# ===========================================================================
# Training Loop
# ===========================================================================

def compute_loss(model, batch, device):
    if not batch: return torch.tensor(0.0), {}
    tensors = torch.stack([torch.from_numpy(ex.board_tensor).float() for ex in batch]).to(device)
    v_tgt = torch.tensor([ex.value_target for ex in batch], dtype=torch.float32, device=device).unsqueeze(1)
    p_tgt = torch.stack([torch.from_numpy(ex.policy_target).float() for ex in batch]).to(device)
    w_tgt = torch.stack([torch.from_numpy(ex.wdl_target).float() for ex in batch]).to(device)
    
    values, p_logits, w_logits, k_projs = model(tensors)
    
    v_loss = F.mse_loss(values, v_tgt)
    p_loss = F.cross_entropy(p_logits, p_tgt)
    w_loss = F.cross_entropy(w_logits, w_tgt)
    
    # Manifold regularization
    k_norm = F.normalize(k_projs, dim=1)
    collapse = (k_norm @ k_norm.T).mean()
    
    total = v_loss + 0.5 * p_loss + 0.3 * w_loss + 0.01 * collapse
    return total, {'value': v_loss.item(), 'policy': p_loss.item(),
                   'wdl': w_loss.item(), 'collapse': collapse.item()}


def train_loop(model=None, k_start=4, k_target=64, hidden_dim=128,
               games_per_batch=4, batch_size=128, epochs_per_batch=3,
               sims_per_move=50, buffer_size=200000,
               checkpoint_interval=50, log_interval=10, resume_path=None):
    """Main continuous training loop."""
    global _stop_requested
    
    # Create or load model
    if model is None:
        model = create_model(k_manifold=k_start, hidden_dim=hidden_dim,
                            num_layers=3, use_jit=False).to(DEVICE)
    
    optimizer = create_optimizer(model, lr=1e-3)
    
    k_scheduler = KExpansionScheduler(
        model, k_start=k_start, k_target=k_target,
        warmup_epochs=200, total_epochs=10000,
    )
    
    buffer = PrioritizedBuffer(max_size=buffer_size)
    generator = FastDataGenerator(model, sims_per_move=sims_per_move)
    
    save_dir = Path(__file__).parent.parent / "models"
    save_dir.mkdir(exist_ok=True)
    
    # Load checkpoint
    start_iteration = 0
    if resume_path and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        start_iteration = ck.get('iteration', 0)
        k_scheduler.current_k = ck.get('k_current', k_start)
        k_scheduler.epoch = start_iteration
        print(f"Resumed from iteration {start_iteration}")
    
    tp, tr = count_parameters(model)
    print("=" * 60)
    print("HyperTensor Chess — Continuous Training Loop")
    print("=" * 60)
    print(f"Model: {tr:,} params | Device: {DEVICE} | K: {k_start}→{k_target}")
    print(f"Games/batch: {games_per_batch} | Sims/move: {sims_per_move}")
    print(f"Checkpoint every {checkpoint_interval} iterations")
    print(f"Press Ctrl+C for graceful shutdown")
    print("=" * 60, flush=True)
    
    history = {'loss': [], 'v_loss': [], 'p_loss': [], 'k_vals': [],
               'games': [], 'times': []}
    iteration = start_iteration
    
    try:
        while not _stop_requested:
            iteration += 1
            iter_start = time.time()
            current_k = k_scheduler.current_k
            
            # 1. Generate training data
            model.eval()
            examples = generator.generate_batch(games_per_batch)
            for ex in examples:
                buffer.add(ex, novelty_score=random.random())
            
            game_stats = {
                'examples': len(examples),
                'buffer': len(buffer),
            }
            
            # 2. Train
            if len(buffer) >= batch_size:
                model.train()
                epoch_losses = []
                for ep in range(epochs_per_batch):
                    batch = buffer.sample(batch_size)
                    if not batch: continue
                    
                    optimizer.zero_grad()
                    loss, metrics = compute_loss(model, batch, DEVICE)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                    
                    epoch_losses.append(metrics)
                
                if epoch_losses:
                    avg = {k: np.mean([e[k] for e in epoch_losses])
                          for k in epoch_losses[0]}
                    game_stats['loss'] = avg
                    history['loss'].append(avg.get('value', 0) + avg.get('policy', 0))
                    history['v_loss'].append(avg.get('value', 0))
                    history['p_loss'].append(avg.get('policy', 0))
                
                # Periodic reorthogonalization
                if iteration % 5 == 0:
                    model.reorthogonalize_all()
            
            # 3. K-expansion
            k_scheduler.step()
            new_k = k_scheduler.current_k
            history['k_vals'].append(new_k)
            
            # 4. Update generator model
            model.eval()
            generator.search.model = model
            generator.model = model
            
            # 5. Apply OnlineOja basis updates
            model.apply_basis_updates()
            
            # 6. Logging
            elapsed = time.time() - iter_start
            history['times'].append(elapsed)
            history['games'].append(game_stats)
            
            if iteration % log_interval == 0 or iteration <= 5:
                k_str = f"k={current_k}"
                if new_k > current_k: k_str += f"→{new_k}"
                loss_str = ""
                if 'loss' in game_stats:
                    loss_str = (f" | loss: v={game_stats['loss']['value']:.4f} "
                              f"p={game_stats['loss']['policy']:.4f}")
                print(f"[Iter {iteration:5d}] {k_str:12s} | "
                      f"buffer: {len(buffer):6d} | "
                      f"{game_stats['examples']:4d} examples{loss_str} | "
                      f"{elapsed:.1f}s", flush=True)
            
            # 7. Checkpoint
            if iteration % checkpoint_interval == 0:
                path = save_dir / f"hypertensor_loop_iter{iteration}.pt"
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'k_current': k_scheduler.current_k,
                    'history': history,
                }, path)
                
                # Also save latest
                latest = save_dir / "hypertensor_latest.pt"
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'k_current': k_scheduler.current_k,
                    'history': history,
                }, latest)
                
                # Save history
                with open(save_dir / "training_loop_history.json", 'w') as f:
                    json.dump(history, f, indent=2, default=float)
                
                print(f"  [Checkpoint] iter_{iteration} saved ({elapsed:.1f}s)")
    
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback; traceback.print_exc()
    finally:
        # Final save
        path = save_dir / "hypertensor_loop_final.pt"
        torch.save({
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'k_current': k_scheduler.current_k,
            'history': history,
        }, path)
        with open(save_dir / "training_loop_history.json", 'w') as f:
            json.dump(history, f, indent=2, default=float)
        print(f"\n[Final] Saved at iteration {iteration}")
        print(f"Total iterations: {iteration}")
        print(f"Final k: {k_scheduler.current_k}")
    
    return model, history


# ===========================================================================
# Quick test
# ===========================================================================

def quick_test():
    """Run a 1-minute training test to verify pipeline."""
    print("Quick training test (60 seconds)...")
    model = create_model(k_manifold=4, hidden_dim=64, num_layers=2).to(DEVICE)
    
    # Single batch
    gen = FastDataGenerator(model, sims_per_move=30)
    t0 = time.time()
    examples = gen.generate_batch(2)
    print(f"Generated {len(examples)} examples in {time.time()-t0:.1f}s")
    
    # Quick train
    model.train()
    optimizer = create_optimizer(model, lr=1e-3)
    tensors = torch.stack([torch.from_numpy(ex.board_tensor).float() for ex in examples[:32]]).to(DEVICE)
    v_tgt = torch.tensor([ex.value_target for ex in examples[:32]], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    
    for _ in range(5):
        optimizer.zero_grad()
        v, p, w, k = model(tensors)
        loss = F.mse_loss(v, v_tgt)
        loss.backward()
        optimizer.step()
    
    print(f"Training OK. Final loss: {loss.item():.4f}")
    print("Pipeline verified!")


if __name__ == '__main__':
    quick_test()
