"""
HyperTensor Chess — Automated Training Launcher
=================================================
Watches for data shards from datagen.py and automatically launches
training when enough data is available.

Keeps GPU at 95%+ utilization by:
  1. Waiting for first shard (100K positions)
  2. Launching training on available shards
  3. Training continues while datagen produces more shards
  4. New shards are picked up automatically

Usage:
  python chess_engine/auto_train.py --data-dir data/ --model-size xl
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import sys
import math
import threading
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_engine.evaluation import create_model, create_optimizer, count_parameters, CUDA_AVAILABLE, DEVICE
from chess_engine.board import Board
from chess_engine.stockfish_train import StockfishEvaluator


@dataclass
class AutoTrainConfig:
    data_dir: str = 'data'
    model_size: str = 'xl'
    batch_size: int = 512
    epochs_per_shard: int = 3
    learning_rate: float = 3e-4
    min_lr: float = 1e-6
    warmup_steps: int = 100
    total_steps: int = 5000
    weight_decay: float = 1e-4
    use_amp: bool = True
    checkpoint_dir: str = 'models/auto_train'
    best_path: str = 'models/auto_train_best.pt'
    resume: bool = True
    pretrained: str = 'models/sf_autopilot_best.pt'
    val_sf_depth: int = 18


class AutoTrainer:
    """Watches for data and trains continuously."""
    
    def __init__(self, config: AutoTrainConfig = None):
        self.config = config or AutoTrainConfig()
        self.device = DEVICE
        
        # Model
        from chess_engine.industrial_train import get_model_config
        mc = get_model_config(self.config.model_size)
        print(f"Creating {self.config.model_size} model: {mc}")
        self.model = create_model(**mc).to(self.device)
        n = count_parameters(self.model)[1]
        print(f"Model: {n:,} params")
        
        # Load pretrained
        pt_path = Path(self.config.pretrained)
        if pt_path.exists():
            self._load_pretrained(pt_path)
        
        # Optimizer
        self.optimizer = create_optimizer(
            self.model, lr=self.config.learning_rate, wd=self.config.weight_decay
        )
        
        # AMP
        self.scaler = torch.amp.GradScaler('cuda') if self.config.use_amp and CUDA_AVAILABLE else None
        
        # State
        self.step = 0
        self.best_val_loss = float('inf')
        self.seen_shards = set()
        
        # Validation
        self._build_validation()
        
        # Checkpoint
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def _load_pretrained(self, path: Path):
        try:
            ck = torch.load(path, map_location=self.device, weights_only=True)
            model_dict = self.model.state_dict()
            pretrained = {k: v for k, v in ck['model_state_dict'].items()
                         if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained)
            self.model.load_state_dict(model_dict, strict=False)
            print(f"Loaded {len(pretrained)}/{len(model_dict)} params from {path.name}")
        except Exception as e:
            print(f"Pretrained load: {e}")
    
    def _build_validation(self):
        boards = [
            Board(),
            Board('rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6'),
            Board('r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 b kq - 3 5'),
            Board('8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1'),
            Board('8/8/8/4k3/8/8/4R3/4K3 w - - 0 1'),
        ]
        print(f"Computing validation targets (SF depth {self.config.val_sf_depth})...")
        sf = StockfishEvaluator(depth=self.config.val_sf_depth)
        self.val_targets = []
        for b in boards:
            r = sf.evaluate(b)
            self.val_targets.append(r['value'])
        sf.close()
        
        self.val_tensors = torch.stack([
            torch.from_numpy(b.to_tensor()).float() for b in boards
        ]).to(self.device)
        self.val_targets_t = torch.tensor(
            self.val_targets, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        print(f"Validation: {len(boards)} positions ready")
    
    def get_lr(self):
        if self.step < self.config.warmup_steps:
            return self.config.learning_rate * (self.step + 1) / self.config.warmup_steps
        progress = (self.step - self.config.warmup_steps) / max(1, self.config.total_steps - self.config.warmup_steps)
        progress = min(1.0, progress)
        return self.config.min_lr + 0.5 * (self.config.learning_rate - self.config.min_lr) * (1 + math.cos(math.pi * progress))
    
    def get_available_shards(self) -> List[Path]:
        """Find all .npz shards in data directory."""
        data_dir = Path(self.config.data_dir)
        shards = sorted(data_dir.glob('shard_*.npz'))
        return [s for s in shards if s.name not in self.seen_shards]
    
    def load_shard(self, path: Path) -> tuple:
        """Load a data shard. Handles corrupted files gracefully."""
        try:
            data = np.load(path)
            X = torch.from_numpy(data['tensors']).float().to(self.device)
            y = torch.from_numpy(data['values']).float().to(self.device).unsqueeze(1)
            return X, y
        except Exception as e:
            print(f"  Corrupted shard {path.name}: {e} — skipping")
            # Rename bad shard so it's not retried
            bad_path = path.with_suffix('.bad')
            path.rename(bad_path)
            return None, None
    
    def validate(self) -> float:
        self.model.eval()
        with torch.inference_mode():
            val_pred, _, _, _ = self.model(self.val_tensors)
            val_loss = F.mse_loss(val_pred, self.val_targets_t).item()
        
        errors = []
        for i in range(len(self.val_tensors)):
            pred_cp = float(torch.tanh(val_pred[i] * 3).item() * 1000)
            true_cp = float(self.val_targets[i] * 1000)
            errors.append(abs(pred_cp - true_cp))
        
        print(f"  Val loss: {val_loss:.6f} | Avg cp err: {np.mean(errors):.0f}")
        return val_loss
    
    def train_on_shard(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 3):
        """Train on a single shard."""
        n = len(X)
        
        for epoch in range(epochs):
            self.model.train()
            
            # Update LR
            lr = self.get_lr()
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
            
            perm = torch.randperm(n, device=self.device)
            losses = []
            self.optimizer.zero_grad()
            
            for i, start in enumerate(range(0, n, self.config.batch_size)):
                idx = perm[start:start + self.config.batch_size]
                xb, yb = X[idx], y[idx]
                
                if self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        vp, _, _, _ = self.model(xb)
                        loss = F.mse_loss(vp, yb)
                    self.scaler.scale(loss).backward()
                else:
                    vp, _, _, _ = self.model(xb)
                    loss = F.mse_loss(vp, yb)
                    loss.backward()
                
                if (i + 1) % 1 == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.step += 1
                
                losses.append(loss.item())
            
            avg_loss = np.mean(losses)
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} lr={lr:.2e}")
        
        # Validate
        val_loss = self.validate()
        
        # Save if best
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'step': self.step,
            }, self.config.best_path)
            print(f"  >>> NEW BEST (val_loss={val_loss:.6f})")
        
        return val_loss
    
    def run(self):
        """Main loop: watch for shards, train continuously."""
        print(f"\n{'='*60}")
        print("AUTO TRAINER — Watching for data shards...")
        print(f"Data dir: {self.config.data_dir}")
        print(f"Device: {self.device}")
        print(f"AMP: {self.config.use_amp}")
        print(f"{'='*60}\n")
        
        last_check = time.time()
        
        while True:
            shards = self.get_available_shards()
            
            if shards:
                for shard_path in shards:
                    print(f"\n--- Training on {shard_path.name} ---")
                    t0 = time.time()
                    
                    X, y = self.load_shard(shard_path)
                    if X is None:
                        continue  # Skip corrupted shard
                    self.train_on_shard(X, y, self.config.epochs_per_shard)
                    
                    self.seen_shards.add(shard_path.name)
                    elapsed = time.time() - t0
                    print(f"  Shard time: {elapsed:.0f}s")
                    
                    # Free memory
                    del X, y
                    if CUDA_AVAILABLE:
                        torch.cuda.empty_cache()
            else:
                # No new shards — wait and check again
                time.sleep(10)
                
                # Periodic status
                if time.time() - last_check > 60:
                    n_shards = len(list(Path(self.config.data_dir).glob('shard_*.npz')))
                    print(f"  Waiting for data... ({n_shards} shards available, "
                          f"best val_loss={self.best_val_loss:.6f})")
                    last_check = time.time()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data')
    p.add_argument('--model-size', default='xl')
    p.add_argument('--batch-size', type=int, default=512)
    args = p.parse_args()
    
    config = AutoTrainConfig(data_dir=args.data_dir, model_size=args.model_size, batch_size=args.batch_size)
    trainer = AutoTrainer(config)
    trainer.run()
