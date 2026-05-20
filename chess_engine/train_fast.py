"""
HyperTensor Chess — Fast Training Loop v2
==========================================
Simple, robust training that actually works. Generates positions with
Stockfish and trains immediately. No complex sharding, no workers, no
race conditions. Just works.

Strategy:
  1. Generate N random positions via self-play (fast)
  2. Evaluate with Stockfish depth 10 (reasonable accuracy, fast)
  3. Train on GPU immediately
  4. Repeat

Usage:
  python chess_engine/train_fast.py --model-size xl --batch-size 512 --hours 24
"""

import torch, numpy as np, time, os, sys, math, random, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_engine.board import Board, Move, Color, Piece, PIECE_VALUES
from chess_engine.evaluation import create_model, create_optimizer, count_parameters, DEVICE
from chess_engine.stockfish_train import StockfishEvaluator
import torch.nn.functional as F


class FastTrainer:
    def __init__(self, model_size='xl', batch_size=512, sf_depth=10,
                 positions_per_cycle=5000, lr=3e-4, pretrained=None):
        self.batch_size = batch_size
        self.sf_depth = sf_depth
        self.positions_per_cycle = positions_per_cycle
        
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
        
        # Validation positions
        self.val_boards = [
            Board(),
            Board('rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6'),
            Board('r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 b kq - 3 5'),
            Board('8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1'),
        ]
        
        # Get SF targets
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
        
        Path('models').mkdir(exist_ok=True)
    
    def generate_positions(self, n):
        """Generate diverse positions by playing random moves."""
        boards = []
        for _ in range(n):
            board = Board()
            # Play 4-40 random moves to reach diverse positions
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
    
    def train_on_batch(self, X_np, y_np, epochs=3):
        """Train on a batch of positions."""
        n = len(X_np)
        X = torch.from_numpy(X_np).float().to(DEVICE)
        y = torch.from_numpy(y_np).float().to(DEVICE).unsqueeze(1)
        
        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(n, device=DEVICE)
            losses = []
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
                
                self.scaler.unscale_(self.optimizer) if self.scaler else None
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                self.scaler.step(self.optimizer) if self.scaler else self.optimizer.step()
                self.scaler.update() if self.scaler else None
                self.optimizer.zero_grad()
                losses.append(loss.item())
        
        del X, y
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        
        return np.mean(losses)
    
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
        
        # Generate
        boards = self.generate_positions(self.positions_per_cycle)
        
        # Evaluate with SF
        X_np, y_np = self.evaluate_with_sf(boards)
        self.total_positions += len(X_np)
        gen_time = time.time() - t0
        
        # Train
        train_t0 = time.time()
        train_loss = self.train_on_batch(X_np, y_np)
        train_time = time.time() - train_t0
        
        # Validate
        val_loss, avg_err = self.validate()
        
        # Elo estimate
        eval_elo = max(1500, int(3000 - 1500 * math.sqrt(max(val_loss, 0.0001))))
        
        # Save if best
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'val_loss': val_loss,
                'cycle': self.cycle,
            }, 'models/train_fast_best.pt')
            new_best = ' *** NEW BEST ***'
        else:
            new_best = ''
        
        total_time = time.time() - t0
        print(f"Cycle {self.cycle:3d} | "
              f"gen {gen_time:4.0f}s | train {train_time:3.0f}s | "
              f"loss {val_loss:.4f} | cp err {avg_err:4.0f} | "
              f"elo ~{eval_elo} | total {self.total_positions:>8,} pos"
              f"{new_best}")
        
        return val_loss
    
    def run(self, hours=24):
        """Run for specified hours."""
        deadline = time.time() + hours * 3600
        print(f"\nRunning for {hours}h (until {time.strftime('%H:%M', time.localtime(deadline))})")
        print(f"SF depth: {self.sf_depth}, positions/cycle: {self.positions_per_cycle}")
        print(f"Device: {DEVICE}\n")
        
        while time.time() < deadline:
            self.run_cycle()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model-size', default='xl')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--sf-depth', type=int, default=10)
    p.add_argument('--positions', type=int, default=5000)
    p.add_argument('--hours', type=float, default=24)
    p.add_argument('--pretrained', default=None)
    args = p.parse_args()
    
    trainer = FastTrainer(
        model_size=args.model_size,
        batch_size=args.batch_size,
        sf_depth=args.sf_depth,
        positions_per_cycle=args.positions,
        pretrained=args.pretrained,
    )
    trainer.run(hours=args.hours)
