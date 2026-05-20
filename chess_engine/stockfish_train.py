"""
Stockfish-Supervised Training Pipeline
=======================================
Uses Stockfish 18 to generate millions of accurately evaluated positions,
then trains the HyperTensor NN to approximate Stockfish-level evaluation.

This is the FASTEST path to strength:
  Stockfish depth 12 ≈ 3000 Elo evaluation accuracy
  NN trained on SF12 data ≈ 2500-2800 Elo evaluation
  NN evaluates in microseconds (vs milliseconds for SF)
  NN-guided MCTS searches much deeper than SF alone

Target: Generate 500K positions, train on them, reach ~2000-2400 Elo.
"""

import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import time
import os
import random
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from chess_engine.board import Board, Move, STARTING_FEN, SQUARE_NAMES
from chess_engine.evaluation import create_model, create_optimizer, count_parameters, CUDA_AVAILABLE, DEVICE
from chess_engine.pretrain import heuristic_evaluate


class StockfishEvaluator:
    """Interface to Stockfish for position evaluation."""
    
    def __init__(self, depth=12, stockfish_path='stockfish'):
        self.depth = depth
        self.process = None
        self._start(stockfish_path)
    
    def _start(self, path):
        """Start Stockfish process."""
        try:
            self.process = subprocess.Popen(
                [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1
            )
            self._send('uci')
            self._send('setoption name Threads value 1')
            self._send('setoption name Hash value 32')
            self._read_until('uciok')
            print(f'  Stockfish started (depth {self.depth})', flush=True)
        except Exception as e:
            print(f'  Stockfish start failed: {e}')
            self.process = None
    
    def _send(self, cmd):
        try:
            if self.process and self.process.stdin:
                self.process.stdin.write(cmd + '\n')
                self.process.stdin.flush()
        except (OSError, BrokenPipeError):
            self.process = None
    
    def _read_until(self, marker):
        lines = []
        try:
            if self.process and self.process.stdout:
                for line in self.process.stdout:
                    lines.append(line)
                    if marker in line:
                        break
        except (OSError, BrokenPipeError):
            pass
        return lines
    
    def _eval_inner(self, board: Board) -> Dict:
        """Core evaluation logic (called with timeout)."""
        fen = board.fen()
        try:
            self._send(f'position fen {fen}')
            self._send(f'go depth {self.depth}')
        except:
            raise RuntimeError("SF send failed")
        
        score_cp = 0
        best_move = None
        search_depth = 0
        
        for line in self.process.stdout:
            line = line.strip()
            
            if line.startswith('bestmove'):
                parts = line.split()
                if len(parts) > 1:
                    best_move = parts[1]
                break
            
            if 'score cp' in line:
                try:
                    parts = line.split()
                    cp_idx = parts.index('cp')
                    if cp_idx + 1 < len(parts):
                        score_cp = int(parts[cp_idx + 1])
                except (ValueError, IndexError):
                    pass
            elif 'score mate' in line:
                try:
                    parts = line.split()
                    mate_idx = parts.index('mate')
                    if mate_idx + 1 < len(parts):
                        mate_in = int(parts[mate_idx + 1])
                        score_cp = 20000 if mate_in > 0 else -20000
                except (ValueError, IndexError):
                    pass
            
            if 'depth' in line:
                try:
                    parts = line.split()
                    d_idx = parts.index('depth')
                    if d_idx + 1 < len(parts):
                        search_depth = int(parts[d_idx + 1])
                except (ValueError, IndexError):
                    pass
        
        cp = max(-4000, min(4000, score_cp))
        wdl_w = 1.0 / (1.0 + np.exp(-cp / 200.0))
        wdl_d = np.exp(-abs(cp) / 400.0) * 0.5
        wdl_l = max(0, 1.0 - wdl_w - wdl_d)
        total = wdl_w + wdl_d + wdl_l
        if total > 0:
            wdl_w /= total; wdl_d /= total; wdl_l /= total
        
        return {
            'score_cp': score_cp,
            'value': np.tanh(score_cp / 400.0),
            'best_move': best_move,
            'wdl': [wdl_w, wdl_d, wdl_l],
            'depth': search_depth,
        }
    
    def evaluate(self, board: Board) -> Dict:
        """Evaluate with timeout — restart SF if it hangs."""
        if self.process is None:
            self._start('stockfish')
        
        if self.process is None or self.process.stdout is None:
            val, _, _ = heuristic_evaluate(board)
            return {'score_cp': int(val*1000), 'best_move': None,
                    'wdl': [0.33,0.34,0.33], 'value': val, 'depth': 0}
        
        import threading
        result = [None]
        exception = [None]
        
        def _run():
            try:
                result[0] = self._eval_inner(board)
            except Exception as e:
                exception[0] = e
        
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=5.0)  # 5s timeout per position
        
        if t.is_alive():
            # Stockfish hung — kill it and restart
            try:
                self.process.kill()
            except:
                pass
            self.process = None
            print(f'  SF timeout, restarting...', flush=True)
            self._start('stockfish')
            val, _, _ = heuristic_evaluate(board)
            return {'score_cp': int(val*1000), 'best_move': None,
                    'wdl': [0.33,0.34,0.33], 'value': val, 'depth': 0}
        
        if exception[0] or result[0] is None:
            # Communication error — restart SF
            try:
                self.process.kill()
            except:
                pass
            self.process = None
            self._start('stockfish')
            val, _, _ = heuristic_evaluate(board)
            return {'score_cp': int(val*1000), 'best_move': None,
                    'wdl': [0.33,0.34,0.33], 'value': val, 'depth': 0}
        
        return result[0]
    
    def close(self):
        if self.process:
            self._send('quit')
            try:
                self.process.wait(timeout=3)
            except:
                self.process.kill()
            self.process = None


def generate_positions(num_positions=50000):
    """Generate diverse chess positions for evaluation."""
    print(f'Generating {num_positions} positions...', flush=True)
    positions = []
    
    for i in range(num_positions):
        board = Board()
        n_moves = random.randint(1, 15)  # Reduced from 2-30 for speed
        for _ in range(n_moves):
            legal = board.generate_legal_moves()
            if not legal or board.is_game_over():
                break
            move = random.choice(legal)
            board.make_move(move)
        
        if not board.is_game_over() and len(board.pieces) > 3:
            positions.append(board)
        
        if (i + 1) % 5000 == 0:
            print(f'  {i+1}/{num_positions} positions generated ({len(positions)} valid)...', flush=True)
    
    print(f'  Generated {len(positions)} valid positions', flush=True)
    return positions


def generate_stockfish_dataset(num_positions=10000, depth=12, output='models/sf_data.npz'):
    """Generate Stockfish-evaluated training dataset with crash recovery."""
    print('=' * 60)
    print('Stockfish-Supervised Dataset Generation')
    print(f'Positions: {num_positions} | SF Depth: {depth}')
    print('=' * 60)
    
    positions = generate_positions(num_positions)
    
    tensors = np.zeros((len(positions), 160, 8, 8), dtype=np.float32)
    values = np.zeros(len(positions), dtype=np.float32)
    scores = np.zeros(len(positions), dtype=np.float32)
    wdls = np.zeros((len(positions), 3), dtype=np.float32)
    
    # Pre-compute tensors
    for i, board in enumerate(positions):
        tensors[i] = board.to_tensor().astype(np.float32)
    
    # Evaluate with Stockfish, restarting periodically
    sf = StockfishEvaluator(depth=depth)
    t0 = time.time()
    
    for i, board in enumerate(positions):
        # Restart Stockfish every 2000 positions to avoid pipe issues
        if i > 0 and i % 2000 == 0:
            sf.close()
            sf = StockfishEvaluator(depth=depth)
        
        try:
            result = sf.evaluate(board)
            values[i] = result['value']
            scores[i] = result['score_cp'] / 100.0
            wdls[i] = result['wdl']
        except Exception as e:
            # Fallback to heuristic on error
            val, _, _ = heuristic_evaluate(board)
            values[i] = val
            scores[i] = val * 10  # Convert to cp-like
            wdls[i] = [0.33, 0.34, 0.33]
        
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(positions) - i - 1) / max(rate, 0.01)
            print(f'  {i+1}/{len(positions)} ({rate:.1f} pos/s, ETA: {eta:.0f}s) '
                  f'score range: [{scores[:i+1].min():+.1f}, {scores[:i+1].max():+.1f}]', flush=True)
        
        # Intermediate save every 2000
        if (i + 1) % 2000 == 0 and i > 0:
            partial = output.replace('.npz', f'_partial_{i+1}.npz')
            np.savez_compressed(partial,
                               tensors=tensors[:i+1], values=values[:i+1],
                               scores=scores[:i+1], wdls=wdls[:i+1])
            print(f'  [Partial save: {partial}]', flush=True)
    
    sf.close()
    
    # Final save
    np.savez_compressed(output, tensors=tensors, values=values,
                       scores=scores, wdls=wdls)
    
    elapsed = time.time() - t0
    print(f'\nDataset saved: {output}')
    print(f'  {len(positions)} positions in {elapsed:.0f}s ({len(positions)/elapsed:.1f} pos/s)')
    print(f'  Score range: [{scores.min():+.1f}, {scores.max():+.1f}] pawns')
    print(f'  Avg |score|: {np.abs(scores).mean():.1f} pawns')
    
    return positions


def train_on_stockfish_data(dataset_path='models/sf_data.npz',
                            model=None, epochs=20, batch_size=256, lr=1e-3):
    """Train NN on Stockfish-evaluated data."""
    print(f'\nLoading {dataset_path}...')
    data = np.load(dataset_path)
    tensors = data['tensors']
    values = data['values']
    wdls = data['wdls']
    print(f'  {len(tensors)} positions loaded')
    
    if model is None:
        model = create_model(k_manifold=32, hidden_dim=128, num_layers=3)
    
    model = model.to(DEVICE)
    model.train()
    optimizer = create_optimizer(model, lr=lr)
    
    n = len(tensors)
    best_loss = float('inf')
    
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        epoch_losses = []
        
        for start in range(0, n, batch_size):
            idx = indices[start:start+batch_size]
            
            x = torch.from_numpy(tensors[idx]).float().to(DEVICE)
            v_tgt = torch.from_numpy(values[idx]).float().to(DEVICE).unsqueeze(1)
            w_tgt = torch.from_numpy(wdls[idx]).float().to(DEVICE)
            
            optimizer.zero_grad()
            v_pred, p_pred, w_pred, k_proj = model(x)
            
            v_loss = F.mse_loss(v_pred, v_tgt)
            w_loss = F.cross_entropy(w_pred, w_tgt)
            
            k_norm = F.normalize(k_proj, dim=1)
            spread = -(k_norm @ k_norm.T).mean() * 0.001
            
            loss = v_loss + 0.3 * w_loss + spread
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            
            epoch_losses.append(v_loss.item())
        
        avg_loss = np.mean(epoch_losses)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model_state_dict': model.state_dict()}, 
                      'models/sf_trained_best.pt')
        
        print(f'  Epoch {epoch+1}/{epochs}: v_loss={avg_loss:.4f} '
              f'(best={best_loss:.4f})', flush=True)
        
        # Validate on startpos
        if (epoch+1) % 5 == 0:
            board = Board()
            t = torch.from_numpy(board.to_tensor()).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                v, _, w, _ = model(t)
                w_p = torch.softmax(w, dim=1).squeeze()
            print(f'    Startpos: {torch.tanh(v*3).item()*1000:+.0f} cp '
                  f'W={w_p[0]:.2f} D={w_p[1]:.2f} L={w_p[2]:.2f}', flush=True)
    
    # Save final
    torch.save({'model_state_dict': model.state_dict()}, 'models/sf_trained_final.pt')
    model.eval()
    
    # Final validation
    print('\n=== Final Validation ===')
    test_positions = [
        ('Startpos', Board()),
        ('White up Queen', Board('r6k/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1')),
        ('Black up Queen', Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R6K w kq - 0 1')),
        ('Sicilian Najdorf', Board('rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 1')),
    ]
    for name, board in test_positions:
        t = torch.from_numpy(board.to_tensor()).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            v, _, w, _ = model(t)
            w_p = torch.softmax(w, dim=1).squeeze()
        print(f'  {name}: {torch.tanh(v*3).item()*1000:+.0f} cp '
              f'W={w_p[0]:.2f} D={w_p[1]:.2f} L={w_p[2]:.2f}')
    
    return model


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--generate', action='store_true')
    p.add_argument('--train', action='store_true')  
    p.add_argument('--positions', type=int, default=5000)
    p.add_argument('--depth', type=int, default=10)
    p.add_argument('--epochs', type=int, default=20)
    args = p.parse_args()
    
    if args.generate:
        generate_stockfish_dataset(args.positions, args.depth)
    
    if args.train:
        train_on_stockfish_data(epochs=args.epochs)
    
    if not args.generate and not args.train:
        # Quick test
        print("Quick Stockfish test...")
        sf = StockfishEvaluator(depth=8)
        board = Board()
        r = sf.evaluate(board)
        print(f"Startpos: {r['score_cp']} cp, best: {r['best_move']}")
        sf.close()
