"""
HyperTensor Chess Engine — Industrial-Scale Training Pipeline
==============================================================
Target: 4000 Elo
Strategy:
  - Parallel Stockfish data generation (N workers)
  - Massive model (10-20M params)
  - Gradient accumulation for large effective batch sizes
  - Cosine annealing + warmup
  - Automatic mixed precision (AMP)
  - Continuous 24/7 training with checkpoint merging
  - Knowledge distillation from Stockfish depth 14-18

Hardware: RTX 4070 Laptop (8.6GB VRAM), Ryzen 9 7940HS (16 threads)

Usage:
  python chess_engine/industrial_train.py
  python chess_engine/industrial_train.py --workers 6 --model-size large
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
import math
import threading
import queue
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_engine.stockfish_train import StockfishEvaluator, generate_positions
from chess_engine.evaluation import create_model, create_optimizer, count_parameters, CUDA_AVAILABLE, DEVICE
from chess_engine.board import Board
from chess_engine.pretrain import heuristic_evaluate

# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class TrainConfig:
    # Model (up to 40M params for A10G 24GB)
    model_size: str = 'xl'  # 'small'→'xxxl': 0.5M→40M params
    
    # Data generation (progressive: 14→18→22 as model improves)
    num_workers: int = 4         # Parallel Stockfish evaluators
    sf_depth: int = 18           # Stockfish search depth (14=~2800, 18=~3200, 22=~3500)
    positions_per_batch: int = 20000
    max_positions: int = 10_000_000
    
    # Training
    epochs_per_batch: int = 5
    batch_size: int = 256
    grad_accum_steps: int = 2
    learning_rate: float = 3e-4  # Peak learning rate
    min_lr: float = 1e-6         # Minimum LR (cosine schedule)
    warmup_epochs: int = 2       # LR warmup epochs
    weight_decay: float = 0.01   # Weight decay
    
    # System
    use_amp: bool = True         # Automatic mixed precision
    save_every: int = 1          # Save checkpoint every N batches
    val_every: int = 1           # Validate every N batches
    checkpoint_dir: str = 'models/industrial'
    best_path: str = 'models/sf_autopilot_best.pt'
    
    # Data
    data_dir: str = 'models/industrial_data'
    resume: bool = True          # Resume from checkpoint


def get_model_config(size: str) -> dict:
    """Model architecture by size. Scales up to 40M params for A10G."""
    configs = {
        'small':  {'k_manifold': 32,  'hidden_dim': 128,  'num_layers': 3},   # ~0.5M
        'medium': {'k_manifold': 48,  'hidden_dim': 256,  'num_layers': 5},   # ~2M
        'large':  {'k_manifold': 64,  'hidden_dim': 384,  'num_layers': 6},   # ~5M
        'xl':     {'k_manifold': 96,  'hidden_dim': 512,  'num_layers': 8},   # ~4.2M (actual: ~12M with FT)
        'xxl':    {'k_manifold': 128, 'hidden_dim': 768,  'num_layers': 10},  # ~20M
        'xxxl':   {'k_manifold': 160, 'hidden_dim': 1024, 'num_layers': 12},  # ~40M (A10G max)
    }
    return configs.get(size, configs['xl'])


# ===========================================================================
# Parallel Data Generator
# ===========================================================================

class ParallelDataGenerator:
    """Generate Stockfish-evaluated positions using multiple workers."""
    
    def __init__(self, config: TrainConfig):
        self.config = config
        self.workers = []
        self.position_queue = queue.Queue(maxsize=1000)
        self.result_queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.total_generated = 0
        
        # Data storage
        self.data_dir = Path(config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Stats
        self.positions_per_sec = 0
        self.last_report = time.time()
        self.last_count = 0
    
    def _worker(self, worker_id: int):
        """Worker thread: generate positions and evaluate with Stockfish."""
        sf = None
        try:
            sf = StockfishEvaluator(depth=self.config.sf_depth)
            pos_count = 0
            
            while not self.stop_flag.is_set():
                # Generate a batch of positions
                try:
                    positions = generate_positions(100)  # 100 positions at a time
                except Exception:
                    continue
                
                tensors = []
                values = []
                
                for board in positions:
                    if self.stop_flag.is_set():
                        break
                    try:
                        tensors.append(board.to_tensor().astype(np.float32))
                        r = sf.evaluate(board)
                        values.append(r['value'])
                        pos_count += 1
                    except Exception:
                        # Restart Stockfish on error
                        try:
                            sf.close()
                        except:
                            pass
                        try:
                            sf = StockfishEvaluator(depth=self.config.sf_depth)
                        except:
                            pass
                        continue
                    
                    # Periodic SF restart
                    if pos_count % 2000 == 0 and pos_count > 0:
                        try:
                            sf.close()
                        except:
                            pass
                        sf = StockfishEvaluator(depth=self.config.sf_depth)
                
                # Send batch to main thread
                if tensors:
                    self.result_queue.put((tensors, values))
        except Exception as e:
            print(f'  Worker {worker_id} error: {e}')
        finally:
            if sf:
                try:
                    sf.close()
                except:
                    pass
    
    def start(self, num_workers: int = None):
        """Start worker threads."""
        if num_workers is None:
            num_workers = self.config.num_workers
        
        print(f'Starting {num_workers} parallel Stockfish workers (depth {self.config.sf_depth})...')
        
        for i in range(num_workers):
            t = threading.Thread(target=self._worker, args=(i,), daemon=True)
            t.start()
            self.workers.append(t)
    
    def stop(self):
        """Stop all workers."""
        self.stop_flag.set()
        for t in self.workers:
            t.join(timeout=5)
    
    def collect_batch(self, target_positions: int) -> Tuple[np.ndarray, np.ndarray]:
        """Collect a batch of evaluated positions."""
        all_tensors = []
        all_values = []
        collected = 0
        
        while collected < target_positions:
            try:
                tensors, values = self.result_queue.get(timeout=60)
                all_tensors.extend(tensors)
                all_values.extend(values)
                collected += len(tensors)
                self.total_generated += len(tensors)
                
                # Progress report
                if self.total_generated - self.last_count >= 5000:
                    elapsed = time.time() - self.last_report
                    rate = (self.total_generated - self.last_count) / elapsed if elapsed > 0 else 0
                    self.positions_per_sec = rate
                    print(f'  Generated {self.total_generated:,} total '
                          f'({rate:.0f} pos/s across {len(self.workers)} workers)')
                    self.last_report = time.time()
                    self.last_count = self.total_generated
            except queue.Empty:
                continue
        
        return np.stack(all_tensors[:target_positions]), np.array(all_values[:target_positions], dtype=np.float32)


# ===========================================================================
# Training Loop with Cosine Annealing
# ===========================================================================

class IndustrialTrainer:
    """Industrial-scale training with all modern techniques."""
    
    def __init__(self, config: TrainConfig = None):
        self.config = config or TrainConfig()
        self.device = DEVICE
        
        # Create model
        mc = get_model_config(self.config.model_size)
        print(f'Creating {self.config.model_size} model: {mc}')
        self.model = create_model(**mc).to(self.device)
        n_params = count_parameters(self.model)[1]
        print(f'Model: {n_params:,} parameters')
        
        # Optimizer with weight decay
        self.optimizer = create_optimizer(
            self.model, lr=self.config.learning_rate, wd=self.config.weight_decay
        )
        
        # AMP scaler
        self.scaler = torch.amp.GradScaler('cuda') if self.config.use_amp and CUDA_AVAILABLE else None
        
        # Learning rate scheduler (cosine with warmup)
        self.current_step = 0
        self.best_val_loss = float('inf')
        
        # Data generator
        self.generator = ParallelDataGenerator(self.config)
        
        # Validation set
        self.val_boards = [
            Board(),
            Board('rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 1'),
            Board('r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1'),
            Board('r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1'),
            Board('8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1'),  # KQvK
            Board('8/8/8/4k3/8/8/4R3/4K3 w - - 0 1'),  # KRvK
        ]
        
        # Compute val targets
        print('Computing Stockfish val targets (depth 18)...')
        sf = StockfishEvaluator(depth=18)
        self.val_targets = []
        for b in self.val_boards:
            r = sf.evaluate(b)
            self.val_targets.append(r['value'])
        sf.close()
        
        self.val_tensors = torch.stack([
            torch.from_numpy(b.to_tensor()).float() for b in self.val_boards
        ]).to(self.device)
        self.val_targets_t = torch.tensor(
            self.val_targets, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        
        # Checkpointing
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Resume
        if self.config.resume:
            self._resume_checkpoint()
    
    def _resume_checkpoint(self):
        """Resume from latest checkpoint."""
        best = Path(self.config.best_path)
        if best.exists():
            try:
                ck = torch.load(best, map_location=self.device, weights_only=True)
                model_dict = self.model.state_dict()
                pretrained = {k: v for k, v in ck['model_state_dict'].items()
                             if k in model_dict and model_dict[k].shape == v.shape}
                model_dict.update(pretrained)
                self.model.load_state_dict(model_dict, strict=False)
                self.best_val_loss = ck.get('val_loss', float('inf'))
                print(f'Resumed: loaded {len(pretrained)}/{len(model_dict)} params, '
                      f'best val_loss={self.best_val_loss:.6f}')
            except Exception as e:
                print(f'Could not resume: {e}')
    
    def get_lr(self) -> float:
        """Cosine learning rate schedule with warmup."""
        if self.current_step < self.config.warmup_epochs:
            # Linear warmup
            return self.config.learning_rate * (self.current_step + 1) / self.config.warmup_epochs
        
        # Cosine decay
        progress = (self.current_step - self.config.warmup_epochs) / max(1, 100 - self.config.warmup_epochs)
        progress = min(1.0, progress)
        return self.config.min_lr + 0.5 * (self.config.learning_rate - self.config.min_lr) * (1 + math.cos(math.pi * progress))
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Single training step with optional gradient accumulation."""
        self.model.train()
        
        if self.scaler is not None:
            with torch.amp.autocast('cuda'):
                val_pred, _, _, _ = self.model(x)
                loss = F.mse_loss(val_pred, y)
            loss = loss / self.config.grad_accum_steps
            self.scaler.scale(loss).backward()
        else:
            val_pred, _, _, _ = self.model(x)
            loss = F.mse_loss(val_pred, y)
            loss = loss / self.config.grad_accum_steps
            loss.backward()
        
        return loss.item() * self.config.grad_accum_steps
    
    def validate(self) -> Dict:
        """Validate on fixed set."""
        self.model.eval()
        with torch.inference_mode():
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    val_pred, _, _, _ = self.model(self.val_tensors)
            else:
                val_pred, _, _, _ = self.model(self.val_tensors)
            val_loss = F.mse_loss(val_pred, self.val_targets_t).item()
        
        # Per-position errors
        errors = []
        for i in range(len(self.val_boards)):
            pred_cp = float(torch.tanh(val_pred[i] * 3).item() * 1000)
            true_cp = float(self.val_targets[i] * 1000)
            errors.append(abs(pred_cp - true_cp))
        
        return {
            'val_loss': val_loss,
            'avg_cp_error': np.mean(errors),
            'per_position': errors,
        }
    
    def train(self):
        """Main training loop."""
        print(f'\n{"="*60}')
        print(f'INDUSTRIAL TRAINING PIPELINE')
        print(f'Target: 4000 Elo')
        print(f'Model: {self.config.model_size} ({count_parameters(self.model)[1]:,} params)')
        print(f'Workers: {self.config.num_workers} × Stockfish depth {self.config.sf_depth}')
        print(f'Batch size: {self.config.batch_size} × {self.config.grad_accum_steps} grad accum')
        print(f'Device: {self.device}')
        print(f'{"="*60}\n')
        
        # Start data generation
        self.generator.start()
        
        batch_num = 0
        total_positions = 0
        
        try:
            while total_positions < self.config.max_positions:
                batch_num += 1
                t_start = time.time()
                
                # Collect training data
                print(f'\n--- Batch {batch_num} ---')
                print(f'Collecting {self.config.positions_per_batch:,} positions...')
                
                X_np, y_np = self.generator.collect_batch(self.config.positions_per_batch)
                total_positions += len(X_np)
                
                # Convert to GPU tensors
                X = torch.from_numpy(X_np).float().to(self.device)
                y = torch.from_numpy(y_np).float().to(self.device).unsqueeze(1)
                
                collect_time = time.time() - t_start
                print(f'Collected {len(X_np):,} positions in {collect_time:.0f}s '
                      f'(total: {total_positions:,})')
                
                # Save data batch
                data_path = self.checkpoint_dir / f'data_batch_{batch_num:04d}.npz'
                np.savez_compressed(data_path, tensors=X_np[:5000], values=y_np[:5000])
                
                # Train
                print(f'Training {self.config.epochs_per_batch} epochs...')
                train_start = time.time()
                
                n = len(X)
                for epoch in range(self.config.epochs_per_batch):
                    # Update LR
                    lr = self.get_lr()
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    
                    # Shuffle
                    perm = torch.randperm(n, device=self.device)
                    epoch_losses = []
                    self.optimizer.zero_grad()
                    
                    for i, start in enumerate(range(0, n, self.config.batch_size)):
                        idx = perm[start:start + self.config.batch_size]
                        xb, yb = X[idx], y[idx]
                        
                        loss_val = self.train_step(xb, yb)
                        epoch_losses.append(loss_val)
                        
                        # Gradient accumulation step
                        if (i + 1) % self.config.grad_accum_steps == 0:
                            if self.scaler is not None:
                                self.scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                                self.optimizer.step()
                            self.optimizer.zero_grad()
                    
                    avg_loss = np.mean(epoch_losses)
                    self.current_step += 1
                    
                    if (epoch + 1) % max(1, self.config.epochs_per_batch // 2) == 0:
                        print(f'  Epoch {epoch+1}/{self.config.epochs_per_batch}: '
                              f'loss={avg_loss:.6f}, lr={lr:.2e}')
                
                train_time = time.time() - train_start
                
                # Validate
                val_results = self.validate()
                val_loss = val_results['val_loss']
                avg_error = val_results['avg_cp_error']
                
                # Elo estimate
                eval_elo = max(1800, min(3500, int(2800 - 900 * math.sqrt(max(val_loss, 0.0001)))))
                play_elo = eval_elo + 200
                
                print(f'\n--- Results ---')
                print(f'Train loss: {avg_loss:.6f}')
                print(f'Val loss:   {val_loss:.6f}')
                print(f'Avg cp err: {avg_error:.0f} cp')
                print(f'Est. Elo:   eval={eval_elo}  play={play_elo}')
                print(f'Positions:  {total_positions:,}')
                print(f'Rate:       {self.generator.positions_per_sec:.0f} pos/s')
                
                # Save if best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'batch_num': batch_num,
                        'total_positions': total_positions,
                        'model_config': get_model_config(self.config.model_size),
                    }, self.config.best_path)
                    print(f'>>> NEW BEST MODEL (val_loss={val_loss:.6f})')
                
                # Periodic checkpoint
                if batch_num % self.config.save_every == 0:
                    ckpt_path = self.checkpoint_dir / f'checkpoint_{batch_num:04d}.pt'
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'batch_num': batch_num,
                        'total_positions': total_positions,
                    }, ckpt_path)
                
                # Clean up
                del X, y, X_np, y_np
                if CUDA_AVAILABLE:
                    torch.cuda.empty_cache()
                
                batch_time = time.time() - t_start
                print(f'Batch time: {batch_time:.0f}s\n')
        
        except KeyboardInterrupt:
            print('\nTraining interrupted. Saving...')
        finally:
            self.generator.stop()
            # Final save
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': self.best_val_loss,
                'batch_num': batch_num,
                'total_positions': total_positions,
                'model_config': get_model_config(self.config.model_size),
            }, self.config.best_path)
            print(f'\nTraining complete. {total_positions:,} positions.')
            print(f'Best val_loss: {self.best_val_loss:.6f}')
            print(f'Best model: {self.config.best_path}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model-size', default='xl', choices=['large','xl','xxl'])
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--sf-depth', type=int, default=14)
    p.add_argument('--positions-per-batch', type=int, default=20000)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--no-resume', action='store_true')
    args = p.parse_args()
    
    config = TrainConfig(
        model_size=args.model_size,
        num_workers=args.workers,
        sf_depth=args.sf_depth,
        positions_per_batch=args.positions_per_batch,
        batch_size=args.batch_size,
        resume=not args.no_resume,
    )
    
    trainer = IndustrialTrainer(config)
    trainer.train()
