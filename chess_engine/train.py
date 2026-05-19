"""
HyperTensor Chess Engine v3.0 — Training Pipeline
===================================================
GPU-accelerated training with:
  - Data generation via deep MCTS self-play
  - Data augmentation (symmetries)
  - Supervised pretraining + RL fine-tuning
  - K-expansion curriculum learning
  - AxiomGauge pre-optimization
  - OnlineOja basis adaptation
  - Mixed precision (AMP)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, json, random
from pathlib import Path
from collections import deque
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from .board import Board, Move, Color, Piece, STARTING_FEN
from .evaluation import (HyperTensorChessNet, KExpansionScheduler,
    RiemannianAdamW, create_model, create_optimizer, count_parameters,
    CUDA_AVAILABLE, DEVICE)
from .search import HyperTensorSearch, play_game
from .data_pipeline import DataGenerator, DataAugmentor
from .opening_book import get_opening_move

@dataclass
class TrainingExample:
    board_tensor: np.ndarray; value_target: float
    policy_target: np.ndarray; wdl_target: np.ndarray; outcome: float

class ExperienceBuffer:
    def __init__(self, max_size=500000): self.buffer = deque(maxlen=max_size)
    def add(self, ex): self.buffer.append(ex)
    def sample(self, n):
        n = min(n, len(self.buffer))
        return random.sample(self.buffer, n)
    def __len__(self): return len(self.buffer)
    def add_examples(self, exs):
        for ex in exs: self.buffer.append(ex)

def compute_loss(model, batch, device):
    tensors = torch.stack([torch.from_numpy(ex.board_tensor).float() for ex in batch]).to(device)
    v_tgt = torch.tensor([ex.value_target for ex in batch], dtype=torch.float32, device=device).unsqueeze(1)
    p_tgt = torch.stack([torch.from_numpy(ex.policy_target).float() for ex in batch]).to(device)
    w_tgt = torch.stack([torch.from_numpy(ex.wdl_target).float() for ex in batch]).to(device)
    values, p_logits, w_logits, k_projs = model(tensors)
    v_loss = F.mse_loss(values, v_tgt)
    p_loss = F.cross_entropy(p_logits, p_tgt)
    w_loss = F.cross_entropy(w_logits, w_tgt)
    collapse = F.normalize(k_projs, dim=1) @ F.normalize(k_projs, dim=1).T; collapse = collapse.mean()
    total = v_loss + 0.3 * p_loss + 0.2 * w_loss + 0.01 * collapse
    return total, {'total': total.item(), 'value': v_loss.item(),
                   'policy': p_loss.item(), 'wdl': w_loss.item()}

class HyperTensorTrainer:
    def __init__(self, k_start=4, k_target=64, hidden_dim=256, lr=1e-3,
                 batch_size=256, buffer_size=500000, games_per_iter=20,
                 epochs_per_iter=5, total_iterations=100, warmup_epochs=20,
                 device='cuda', use_amp=True, data_sims=200):
        self.device = torch.device(device if CUDA_AVAILABLE else 'cpu')
        self.use_amp = use_amp and CUDA_AVAILABLE
        self.batch_size = batch_size
        self.model = create_model(k_manifold=k_start, hidden_dim=hidden_dim,
                                  use_jit=CUDA_AVAILABLE).to(self.device)
        self.optimizer = create_optimizer(self.model, lr=lr)
        self.k_scheduler = KExpansionScheduler(self.model, k_start, k_target, warmup_epochs, total_iterations)
        self.buffer = ExperienceBuffer(max_size=buffer_size)
        self.data_gen = DataGenerator(self.model, sims_per_move=data_sims)
        self.games_per_iter = games_per_iter
        self.epochs_per_iter = epochs_per_iter
        self.total_iterations = total_iterations
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.history = {'loss':[],'v_loss':[],'p_loss':[],'w_loss':[],'k_vals':[],'game_stats':[]}
        self.save_dir = Path(__file__).parent.parent / "models"; self.save_dir.mkdir(exist_ok=True)

    def train(self):
        print("="*60); print("HyperTensor Chess v3.0 — Elite Training"); print("="*60)
        tp,tr=count_parameters(self.model)
        print(f"Model: {tr:,} params | Device: {self.device} | AMP: {self.use_amp}")
        print(f"K: {self.k_scheduler.k_start}→{self.k_scheduler.k_target}")
        if hasattr(self.model,'apply_axiom_gauge'):
            ok=self.model.apply_axiom_gauge(30)
            print(f"AxiomGauge: {'OK' if ok else 'skip'}")
        for it in range(self.total_iterations):
            it0=time.time(); ck=self.k_scheduler.current_k
            print(f"\n--- Iter {it+1}/{self.total_iterations} (k={ck}) ---")
            self.model.eval()
            examples = self.data_gen.generate_dataset(self.games_per_iter)
            # generate_dataset returns List[TrainingExample]; compute stats
            stats = {'1-0': 0, '0-1': 0, '1/2-1/2': 0, 'games': self.games_per_iter}
            for ex in examples:
                if ex.outcome > 0: stats['1-0'] += 1
                elif ex.outcome < 0: stats['0-1'] += 1
                else: stats['1/2-1/2'] += 1
            # Normalize (rough)
            for k in ['1-0','0-1','1/2-1/2']: stats[k] = min(stats[k], 99)
            for ex in examples:
                augmented=DataAugmentor.augment(ex)
                for ae in augmented: self.buffer.add(ae)
            print(f"  Games W/D/L: {stats.get('1-0',0)}/{stats.get('1/2-1/2',0)}/{stats.get('0-1',0)} | Buffer: {len(self.buffer)}")
            self.history['game_stats'].append(stats)
            if len(self.buffer)>=self.batch_size:
                print(f"  Training ({self.epochs_per_iter} epochs)...")
                self.model.train()
                for ep in range(self.epochs_per_iter):
                    b=self.buffer.sample(self.batch_size)
                    self.optimizer.zero_grad()
                    if self.use_amp:
                        with torch.amp.autocast('cuda'):
                            loss,metrics=compute_loss(self.model,b,self.device)
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
                        self.scaler.step(self.optimizer); self.scaler.update()
                    else:
                        loss,metrics=compute_loss(self.model,b,self.device)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
                        self.optimizer.step()
                    if ep%5==0: self.model.reorthogonalize_all()
                avg={k:np.mean([metrics[k]]) for k in metrics}
                self.history['loss'].append(avg['total'])
                print(f"  Loss: {avg['total']:.4f} v={avg['value']:.4f} p={avg['policy']:.4f}")
            self.model.apply_basis_updates()
            self.k_scheduler.step()
            nk=self.k_scheduler.current_k
            if nk>ck: print(f"  K-expand: {ck}→{nk}")
            self.history['k_vals'].append(nk)
            self.model.eval(); self.data_gen.search.model=self.model
            print(f"  Time: {time.time()-it0:.0f}s")
            if (it+1)%10==0: self.save_checkpoint(it+1)
        self.save_checkpoint('final'); self.save_history()

    def save_checkpoint(self, tag):
        p=self.save_dir/f"hypertensor_chess_v3_{tag}.pt"
        torch.save({'model_state_dict':self.model.state_dict(),
                    'optimizer_state_dict':self.optimizer.state_dict(),
                    'k_current':self.k_scheduler.current_k,
                    'history':self.history}, p)
        print(f"  Saved: {p}")

    def save_history(self):
        with open(self.save_dir/"training_history_v3.json",'w') as f:
            json.dump(self.history,f,indent=2,default=float)

    def load_checkpoint(self,p):
        ck=torch.load(p,map_location=self.device)
        self.model.load_state_dict(ck['model_state_dict'])
        self.optimizer.load_state_dict(ck['optimizer_state_dict'])
        self.k_scheduler.current_k=ck.get('k_current',self.k_scheduler.k_target)
        if 'history' in ck: self.history=ck['history']

def quick_demo():
    print("HyperTensor Chess v3.0 — Quick Demo"); print("="*50)
    m=create_model(k_manifold=8,hidden_dim=64,num_layers=2)
    t,tr=count_parameters(m)
    print(f"Device: {DEVICE} | CUDA: {CUDA_AVAILABLE} | JIT: {m.jit_enabled}")
    print(f"Model: {tr:,} params")
    batch=np.zeros((32,160,8,8),dtype=np.float32)
    r=m.evaluate_batch(batch)
    print(f"Batch eval (32): {r['time_ms']:.1f}ms | {r['pps']:.0f} pos/s")
    print("\nSelf-play game...")
    result,moves=play_game(m,m,time_per_move_ms=500,max_moves=40)
    print(f"Result: {result} | Moves: {len(moves)}")
    from .evaluation import ChessNativeLinear
    nl=ChessNativeLinear(256,8)
    print(f"NativeLinear savings: {nl.param_savings*100:.1f}%")
    print("Demo complete!")

if __name__=='__main__': quick_demo()
