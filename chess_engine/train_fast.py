"""
HyperTensor Chess — Fast Training Loop v3
==========================================
GPU-saturating trainer: accumulates positions, trains HARD each cycle.
Generates fresh positions with Stockfish evaluation, adds to growing
dataset, then trains for many epochs on ALL data.

Strategy:
  1. Generate N random positions via self-play (fast)
  2. Evaluate with Stockfish depth 10
  3. Add to accumulated dataset (up to 200K positions)
  4. Train on GPU for MANY epochs (50+) — actually uses the GPU
  5. Repeat

Usage:
  python chess_engine/train_fast.py --model-size xl --batch-size 1024 --epochs 50 --hours 24
"""

import torch, numpy as np, time, os, sys, math, random, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_engine.board import Board, Move, Color, Piece, PIECE_VALUES
from chess_engine.evaluation import create_model, create_optimizer, count_parameters, DEVICE
from chess_engine.stockfish_train import StockfishEvaluator
import torch.nn.functional as F


class FastTrainer:
    def __init__(self, model_size='xl', batch_size=1024, sf_depth=10,
                 positions_per_cycle=10000, epochs_per_cycle=50,
                 lr=3e-4, pretrained=None, max_dataset=200000):
        self.batch_size = batch_size
        self.sf_depth = sf_depth
        self.positions_per_cycle = positions_per_cycle
        self.epochs_per_cycle = epochs_per_cycle
        self.max_dataset = max_dataset
        
        # Model
        from chess_engine.industrial_train import get_model_config
        mc = get_model_config(model_size)
        self.model = create_model(**mc).to(DEVICE)
        n = count_parameters(self.model)[1]
        print(f"Model: {n:,} params on {DEVICE}")
        
        # Load pretrained
        if pretrained and Path(pretrained).exists():
            ck = torch.load(pretrained, map_location=DEVICE, weights_only=True)
            md = self.model.state_dict()
            pt = {k: v for k, v in ck.get('model_state_dict', ck).items() 
                  if k in md and md[k].shape == v.shape}
            md.update(pt)
            self.model.load_state_dict(md, strict=False)
            print(f"Loaded {len(pt)}/{len(md)} params")
        
        self.optimizer = create_optimizer(self.model, lr=lr)
        self.scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
        
        # Accumulated dataset
        self.all_X = []  # list of np arrays
        self.all_y = []
        self.dataset_size = 0
        
        # Validation positions (fixed, evaluated once at SF depth 18)
        self.val_boards = [
            Board(),
            Board('rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6'),
            Board('r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 b kq - 3 5'),
            Board('8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1'),
        ]
        
        print(f"Getting SF depth-18 validation targets...")
        sf = StockfishEvaluator(depth=18)
        self.val_targets = []
        for b in self.val_boards:
            r = sf.evaluate(b)
            self.val_targets.append(r['value'])
        sf.close()
        
        self.val_X = torch.stack([
            torch.from_numpy(b.to_tensor()).float() for b in self.val_boards
        ]).to(DEVICE)
        self.val_y = torch.tensor(self.val_targets, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        
        self.cycle = 0
        self.best_val_loss = float('inf')
        self.total_positions = 0
        self.train_time_total = 0.0
        
        Path('models').mkdir(exist_ok=True)
    
    def generate_positions(self, n):
        """Generate diverse positions by playing random moves."""
        boards = []
        for _ in range(n):
            board = Board()
            for _ in range(random.randint(4, 40)):
                moves = list(board.generate_legal_moves())
                if not moves:
                    break
                board.make_move(random.choice(moves))
            boards.append(board)
        return boards
    
    def evaluate_with_sf(self, boards):
        """Batch evaluate positions with Stockfish."""
        sf = StockfishEvaluator(depth=self.sf_depth)
        tensors, values = [], []
        for i, b in enumerate(boards):
            try:
                tensors.append(b.to_tensor().astype(np.float32))
                r = sf.evaluate(b)
                values.append(r['value'])
            except:
                continue
        sf.close()
        return np.stack(tensors), np.array(values, dtype=np.float32)
    
    def train_on_dataset(self):
        """Train for many epochs on the entire accumulated dataset."""
        if self.dataset_size == 0:
            return
        
        X = torch.from_numpy(np.concatenate(self.all_X, axis=0)).float().to(DEVICE)
        y = torch.from_numpy(np.concatenate(self.all_y, axis=0)).float().to(DEVICE).unsqueeze(1)
        n = len(X)
        
        # Always train hard — GPU is idle otherwise
        epochs = self.epochs_per_cycle
        
        for epoch in range(epochs):
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
    
    def validate(self):
        """Validate on fixed set."""
        self.model.eval()
        with torch.inference_mode():
            vp, _, _, _ = self.model(self.val_X)
            val_loss = F.mse_loss(vp, self.val_y).item()
        
        errors = [abs(float(torch.tanh(vp[i]*3))*1000 - self.val_targets[i]*1000) 
                  for i in range(len(self.val_boards))]
        return val_loss, np.mean(errors)
    
    def run_cycle(self):
        """One generate-evaluate-train cycle."""
        self.cycle += 1
        t0 = time.time()
        
        # Generate + Evaluate with SF (CPU-bound)
        boards = self.generate_positions(self.positions_per_cycle)
        X_np, y_np = self.evaluate_with_sf(boards)
        n_new = len(X_np)
        gen_time = time.time() - t0
        self.total_positions += n_new
        
        # Add to accumulated dataset
        self.all_X.append(X_np)
        self.all_y.append(y_np)
        self.dataset_size += n_new
        
        # Trim old data if exceeding max
        while self.dataset_size > self.max_dataset and len(self.all_X) > 1:
            old_n = len(self.all_X[0])
            self.all_X.pop(0)
            self.all_y.pop(0)
            self.dataset_size -= old_n
        
        # Train HARD on GPU
        train_t0 = time.time()
        self.train_on_dataset()
        train_time = time.time() - train_t0
        self.train_time_total += train_time
        
        # Validate
        val_loss, avg_err = self.validate()
        eval_elo = max(1500, int(3000 - 1500 * math.sqrt(max(val_loss, 0.0001))))
        
        # Save if best
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'val_loss': val_loss,
                'cycle': self.cycle,
                'total_positions': self.total_positions,
            }, 'models/train_fast_best.pt')
            new_best = ' *** NEW BEST ***'
        else:
            new_best = ''
        
        total_time = time.time() - t0
        gpu_pct = (train_time / total_time * 100) if total_time > 0 else 0
        print(f"Cycle {self.cycle:3d} | "
              f"gen {gen_time:4.0f}s | train {train_time:4.0f}s ({gpu_pct:.0f}% GPU) | "
              f"loss {val_loss:.4f} | cp err {avg_err:4.0f} | "
              f"elo ~{eval_elo} | dataset {self.dataset_size:>7,} | "
              f"total {self.total_positions:>9,} pos"
              f"{new_best}")
        
        return val_loss
    
    def run(self, hours=24):
        """Run for specified hours."""
        deadline = time.time() + hours * 3600
        print(f"\nRunning for {hours}h (until {time.strftime('%H:%M', time.localtime(deadline))})")
        print(f"SF depth: {self.sf_depth}, positions/cycle: {self.positions_per_cycle}")
        print(f"Epochs/cycle: up to {self.epochs_per_cycle}, max dataset: {self.max_dataset:,}")
        print(f"Device: {DEVICE}\n")
        
        while time.time() < deadline:
            self.run_cycle()
        
        print(f"\nDone. Total cycles: {self.cycle}, best val_loss: {self.best_val_loss:.4f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model-size', default='xl')
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--sf-depth', type=int, default=10)
    p.add_argument('--positions', type=int, default=10000)
    p.add_argument('--epochs', type=int, default=50, help='Max epochs per cycle')
    p.add_argument('--hours', type=float, default=24)
    p.add_argument('--pretrained', default=None)
    p.add_argument('--max-dataset', type=int, default=200000)
    args = p.parse_args()
    
    trainer = FastTrainer(
        model_size=args.model_size,
        batch_size=args.batch_size,
        sf_depth=args.sf_depth,
        positions_per_cycle=args.positions,
        epochs_per_cycle=args.epochs,
        pretrained=args.pretrained,
        max_dataset=args.max_dataset,
    )
    trainer.run(hours=args.hours)
