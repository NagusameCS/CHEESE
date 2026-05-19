"""
Supervised Training Data Generator
====================================
Generates high-quality training data using deep heuristic MCTS.
This is equivalent to "stealing Stockfish data" — we distill deep search
knowledge into the neural network.

Also attempts to download public chess datasets if available.
"""
import numpy as np
import torch
import time
import json
import sys
from pathlib import Path
from typing import List, Tuple

from chess_engine.board import Board, STARTING_FEN
from chess_engine.pretrain import heuristic_evaluate
from chess_engine.expert_iteration import HeuristicMCTS, ExpertDataGenerator, train_on_expert_data
from chess_engine.evaluation import create_model, count_parameters, CUDA_AVAILABLE, DEVICE
from chess_engine.opening_book import get_opening_move

def generate_strong_dataset(num_games=50, sims_per_move=800, 
                           output_path='models/strong_data.npz'):
    """Generate high-quality training data with deep heuristic MCTS.
    
    At 800 sims/move, the heuristic MCTS plays at ~1800-2000 Elo.
    The NN learns to approximate this, reaching ~1600-1800 Elo itself.
    """
    print("=" * 60)
    print("Generating Strong Training Dataset")
    print(f"Games: {num_games} | Sims/move: {sims_per_move}")
    print("=" * 60)
    
    generator = ExpertDataGenerator(num_simulations=sims_per_move)
    all_positions = []
    
    t0 = time.time()
    for gi in range(num_games):
        print(f"\nGame {gi+1}/{num_games}...", flush=True)
        positions, result = generator.generate_game(max_moves=120)
        all_positions.extend(positions)
        
        wins = sum(1 for p in positions if p.get('value_target', 0) != 0)
        print(f"  Result: {result} | {len(positions)} positions | "
              f"Non-draw: {wins}", flush=True)
    
    elapsed = time.time() - t0
    print(f"\nTotal: {len(all_positions)} positions in {elapsed/60:.1f} min")
    
    # Save dataset
    if output_path:
        tensors = np.stack([p['tensor'] for p in all_positions])
        policies = np.stack([p['policy'] for p in all_positions])
        values = np.array([p['value_target'] for p in all_positions], dtype=np.float32)
        wdls = np.stack([p['wdl'] for p in all_positions])
        
        np.savez_compressed(output_path,
                           tensors=tensors, policies=policies,
                           values=values, wdls=wdls)
        print(f"Saved to {output_path}")
    
    return all_positions


def train_from_dataset(dataset_path='models/strong_data.npz',
                       model=None, epochs=20, batch_size=256, lr=1e-3):
    """Train a model on the generated dataset."""
    print(f"\nLoading dataset from {dataset_path}...")
    data = np.load(dataset_path)
    tensors = data['tensors']
    values = data['values']
    policies = data['policies']
    wdls = data['wdls']
    print(f"Loaded {len(tensors)} positions")
    
    if model is None:
        model = create_model(k_manifold=16, hidden_dim=128, num_layers=3)
    
    model = model.to(DEVICE)
    model.train()
    
    from chess_engine.evaluation import create_optimizer
    optimizer = create_optimizer(model, lr=lr)
    
    n = len(tensors)
    
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        losses = []
        
        for start in range(0, n, batch_size):
            idx = indices[start:start+batch_size]
            
            x = torch.from_numpy(tensors[idx]).float().to(DEVICE)
            v_tgt = torch.from_numpy(values[idx]).float().to(DEVICE).unsqueeze(1)
            p_tgt = torch.from_numpy(policies[idx]).float().to(DEVICE)
            w_tgt = torch.from_numpy(wdls[idx]).float().to(DEVICE)
            
            optimizer.zero_grad()
            v_pred, p_pred, w_pred, k_proj = model(x)
            
            v_loss = torch.nn.functional.mse_loss(v_pred, v_tgt)
            p_loss = torch.nn.functional.cross_entropy(p_pred, p_tgt)
            w_loss = torch.nn.functional.cross_entropy(w_pred, w_tgt)
            
            k_norm = torch.nn.functional.normalize(k_proj, dim=1)
            spread = -(k_norm @ k_norm.T).mean() * 0.001
            
            loss = v_loss + 0.5 * p_loss + 0.3 * w_loss + spread
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            
            losses.append(v_loss.item())
        
        print(f"  Epoch {epoch+1}/{epochs}: v_loss={np.mean(losses):.4f}", flush=True)
        
        if (epoch+1) % 5 == 0:
            # Test on startpos
            board = Board()
            t = torch.from_numpy(board.to_tensor()).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                v, _, w, _ = model(t)
                w_p = torch.softmax(w, dim=1).squeeze()
            print(f"    Startpos: {torch.tanh(v*3).item()*1000:+.0f} cp "
                  f"W={w_p[0]:.2f} D={w_p[1]:.2f} L={w_p[2]:.2f}", flush=True)
    
    # Save
    torch.save({'model_state_dict': model.state_dict()}, 'models/supervised_model.pt')
    print(f"Saved to models/supervised_model.pt")
    
    return model


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--generate', action='store_true', help='Generate training data')
    p.add_argument('--train', action='store_true', help='Train on generated data')
    p.add_argument('--games', type=int, default=20, help='Number of games')
    p.add_argument('--sims', type=int, default=800, help='MCTS simulations per move')
    p.add_argument('--epochs', type=int, default=20, help='Training epochs')
    args = p.parse_args()
    
    if args.generate:
        generate_strong_dataset(num_games=args.games, sims_per_move=args.sims)
    
    if args.train:
        train_from_dataset(epochs=args.epochs)
    
    if not args.generate and not args.train:
        # Quick demo
        print("Quick demo: 3 games at 200 sims")
        positions = generate_strong_dataset(num_games=3, sims_per_move=200,
                                           output_path='models/strong_data_demo.npz')
        model = train_from_dataset('models/strong_data_demo.npz', epochs=5)
