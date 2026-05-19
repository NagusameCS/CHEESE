"""
HyperTensor Chess — Self-Play Reinforcement Learning Pipeline
==============================================================
AlphaZero-style self-play training loop. The model plays games against
itself, learns from the outcomes, and iteratively improves beyond any
fixed-depth teacher.

This is THE mechanism that let AlphaZero/Leela beat Stockfish:
  1. Generate self-play games using current model + MCTS/search
  2. Record (position, outcome) pairs
  3. Train model to predict outcomes (WDL) with cross-entropy loss
  4. New model plays against old, better one is kept
  5. Repeat → model strength compounds over iterations

Architecture:
  SelfPlayWorker: Plays games using NegamaxEngine + current model
  Arena: Pits new model vs old model (rating comparison)
  RLTrainer: Orchestrates the RL loop

Usage:
  python chess_engine/self_play.py --games 1000 --model-size xl --iterations 10
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
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_engine.board import Board, Move, Color, STARTING_FEN
from chess_engine.evaluation import create_model, create_optimizer, count_parameters, CUDA_AVAILABLE, DEVICE
from chess_engine.negamax import NegamaxEngine, SearchState
from chess_engine.pretrain import heuristic_evaluate


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class SelfPlayConfig:
    # Model
    model_size: str = 'xl'
    model_path: str = 'models/sf_autopilot_best.pt'
    
    # Self-play
    games_per_iteration: int = 500
    search_depth: int = 4         # Depth for self-play (fast, many games)
    search_nodes: int = 1000      # Max nodes per move
    temperature: float = 1.0      # Move selection temperature
    temperature_decay: float = 0.95  # Decay per move
    resignation_threshold: float = -0.85  # Resign if eval drops below this
    
    # Training
    epochs_per_iteration: int = 10
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    wdl_loss_weight: float = 1.0   # Primary loss for RL training
    policy_loss_weight: float = 0.5
    
    # Arena (model comparison)
    arena_games: int = 100
    arena_depth: int = 6
    acceptance_threshold: float = 0.55  # New model must win >55% to be kept
    
    # Data
    positions_per_game: int = 40    # Sample N positions per game for training
    replay_buffer_size: int = 500_000  # Max positions in training buffer
    
    # Saving
    checkpoint_dir: str = 'models/self_play'
    best_path: str = 'models/self_play_best.pt'
    save_every: int = 5
    
    # Misc
    num_workers: int = 4           # Parallel game workers (for CPU games)
    use_gpu_search: bool = False   # Use GPU for evaluation during search


# ===========================================================================
# Self-Play Game Worker
# ===========================================================================

class SelfPlayWorker:
    """Plays a single self-play game and records training data."""
    
    def __init__(self, model, config: SelfPlayConfig, worker_id: int = 0):
        self.model = model
        self.config = config
        self.worker_id = worker_id
        self.engine = None  # Created per game
    
    def play_game(self) -> List[Dict]:
        """
        Play one self-play game.
        
        Returns:
            List of dicts: {board_tensor, outcome, policy_target}
            outcome ∈ {1.0 (white win), 0.0 (draw), -1.0 (black win)}
        """
        board = Board()
        positions = []
        move_count = 0
        temperature = self.config.temperature
        
        # Create engine for this game
        self.engine = NegamaxEngine(
            self.model,
            tt_size_mb=16,
            max_depth=self.config.search_depth,
            max_nodes=self.config.search_nodes,
        )
        
        # Play until game ends or max moves
        while move_count < 300:
            # Check game state
            if board.is_checkmate():
                # Current player lost
                outcome = -1.0 if board.color_to_move == Color.WHITE else 1.0
                return self._label_positions(positions, outcome)
            
            if board.is_stalemate() or board.is_insufficient_material():
                return self._label_positions(positions, 0.0)
            
            if board.halfmove_clock >= 100:
                return self._label_positions(positions, 0.0)
            
            # Record position (before move)
            if move_count >= 4 and len(positions) < self.config.positions_per_game:
                positions.append({
                    'board': board.copy(),
                    'side_to_move': board.color_to_move,
                })
            
            # Search for best move
            result = self.engine.find_best_move(board, time_limit=1.0)
            best_move = result.get('best_move')
            move_scores = result.get('move_scores', [])
            
            if best_move is None:
                # No legal moves
                if board.is_checkmate():
                    outcome = -1.0 if board.color_to_move == Color.WHITE else 1.0
                else:
                    outcome = 0.0
                return self._label_positions(positions, outcome)
            
            # Temperature-based move selection
            if move_count < 15:  # Opening: more exploration
                temperature = self.config.temperature * (self.config.temperature_decay ** max(0, move_count - 4))
                move = self._select_move_with_temperature(best_move, move_scores, temperature)
            else:
                move = best_move
            
            # Check resignation
            if result.get('score', 0) < self.config.resignation_threshold and move_count > 20:
                outcome = -1.0 if board.color_to_move == Color.WHITE else 1.0
                return self._label_positions(positions, outcome)
            
            board.push(move)
            move_count += 1
        
        # Draw by move limit
        return self._label_positions(positions, 0.0)
    
    def _select_move_with_temperature(self, best_move, move_scores, temperature):
        """Select move using temperature-based sampling."""
        if not move_scores or temperature < 0.01:
            return best_move
        
        # Convert scores to probabilities
        moves = [ms[0] for ms in move_scores[:10]]  # Top 10 moves
        scores = np.array([ms[1] for ms in move_scores[:10]], dtype=np.float64)
        
        # Softmax with temperature
        scores = scores - scores.max()  # Stability
        probs = np.exp(scores / max(temperature, 0.01))
        probs = probs / probs.sum()
        
        # Sample
        chosen_idx = np.random.choice(len(moves), p=probs)
        return moves[chosen_idx]
    
    def _label_positions(self, positions: List[Dict], outcome: float) -> List[Dict]:
        """Label positions with game outcome."""
        for pos in positions:
            if pos['side_to_move'] == Color.WHITE:
                pos['outcome'] = outcome
            else:
                pos['outcome'] = -outcome
        
        return positions


# ===========================================================================
# Replay Buffer
# ===========================================================================

class ReplayBuffer:
    """Circular buffer of training positions from self-play games."""
    
    def __init__(self, max_size: int = 500_000):
        self.max_size = max_size
        self.positions: List[Dict] = []
    
    def add(self, positions: List[Dict]):
        """Add positions from a game."""
        self.positions.extend(positions)
        # Trim
        if len(self.positions) > self.max_size:
            self.positions = self.positions[-self.max_size:]
    
    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of positions for training."""
        if len(self.positions) < n:
            n = len(self.positions)
        
        batch = random.sample(self.positions, n)
        
        tensors = np.stack([p['board'].to_tensor().astype(np.float32) for p in batch])
        outcomes = np.array([p['outcome'] for p in batch], dtype=np.float32)
        
        # Convert outcomes to WDL targets
        # outcome 1.0 → [1,0,0], 0.0 → [0,1,0], -1.0 → [0,0,1]
        wdls = np.zeros((n, 3), dtype=np.float32)
        wdls[:, 0] = np.maximum(outcomes, 0)           # win%
        wdls[:, 2] = np.maximum(-outcomes, 0)          # loss%
        wdls[:, 1] = 1.0 - wdls[:, 0] - wdls[:, 2]    # draw%
        
        return tensors, outcomes, wdls
    
    def __len__(self):
        return len(self.positions)


# ===========================================================================
# Arena: Model Comparison
# ===========================================================================

class Arena:
    """Pits new model vs old model to measure strength improvement."""
    
    def __init__(self, config: SelfPlayConfig):
        self.config = config
    
    def play_match(self, new_model, old_model, num_games: int = 100) -> Dict:
        """
        Play new_model vs old_model.
        
        Returns:
            {new_wins, old_wins, draws, elo_diff}
        """
        new_wins = 0
        old_wins = 0
        draws = 0
        
        for i in range(num_games):
            # Alternate colors
            if i % 2 == 0:
                white_model, black_model = new_model, old_model
            else:
                white_model, black_model = old_model, new_model
            
            result = self._play_game(white_model, black_model)
            
            if result == 1.0:  # White wins
                if i % 2 == 0:
                    new_wins += 1
                else:
                    old_wins += 1
            elif result == -1.0:  # Black wins
                if i % 2 == 0:
                    old_wins += 1
                else:
                    new_wins += 1
            else:
                draws += 1
        
        # Calculate Elo difference
        total = new_wins + old_wins + draws
        if total > 0 and new_wins + old_wins > 0:
            win_rate = (new_wins + draws * 0.5) / total
            elo_diff = -400 * math.log10(1.0 / max(win_rate, 0.001) - 1.0)
        else:
            elo_diff = 0.0
        
        return {
            'new_wins': new_wins,
            'old_wins': old_wins,
            'draws': draws,
            'elo_diff': elo_diff,
            'new_win_rate': (new_wins + draws * 0.5) / max(total, 1),
        }
    
    def _play_game(self, white_model, black_model) -> float:
        """Play a single game between two models."""
        board = Board()
        
        # Create engines
        white_engine = NegamaxEngine(white_model, tt_size_mb=8,
                                     max_depth=self.config.arena_depth)
        black_engine = NegamaxEngine(black_model, tt_size_mb=8,
                                     max_depth=self.config.arena_depth)
        
        for _ in range(300):
            if board.is_checkmate():
                return -1.0 if board.color_to_move == Color.WHITE else 1.0
            if board.is_stalemate() or board.is_insufficient_material():
                return 0.0
            if board.halfmove_clock >= 100:
                return 0.0
            
            engine = white_engine if board.color_to_move == Color.WHITE else black_engine
            result = engine.find_best_move(board, time_limit=0.5)
            best_move = result.get('best_move')
            
            if best_move is None:
                return 0.0
            
            board.push(best_move)
        
        return 0.0


# ===========================================================================
# RL Trainer
# ===========================================================================

class SelfPlayTrainer:
    """Orchestrates the self-play RL training loop."""
    
    def __init__(self, config: SelfPlayConfig = None):
        self.config = config or SelfPlayConfig()
        self.device = DEVICE
        
        # Create model
        from chess_engine.evaluation import get_model_config
        mc = get_model_config(self.config.model_size) if hasattr(
            __import__('chess_engine.evaluation', fromlist=['get_model_config']), 
            'get_model_config') else {'k_manifold': 96, 'hidden_dim': 512, 'num_layers': 8}
        self.model = create_model(**mc).to(self.device)
        
        # Load pretrained weights
        model_path = Path(self.config.model_path)
        if model_path.exists():
            self._load_weights(model_path)
        
        # Optimizer
        self.optimizer = create_optimizer(
            self.model, lr=self.config.learning_rate, wd=self.config.weight_decay
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=self.config.replay_buffer_size)
        
        # Arena
        self.arena = Arena(config)
        
        # Old model (for comparison)
        self.old_model = None
        
        # Checkpoint
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.iteration = 0
        self.best_elo = 0
    
    def _load_weights(self, path: Path):
        """Load pretrained weights with shape matching."""
        try:
            ck = torch.load(path, map_location=self.device, weights_only=True)
            model_dict = self.model.state_dict()
            pretrained = {k: v for k, v in ck['model_state_dict'].items()
                         if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained)
            self.model.load_state_dict(model_dict, strict=False)
            print(f'Loaded {len(pretrained)}/{len(model_dict)} params from {path}')
        except Exception as e:
            print(f'Weight load failed: {e}')
    
    def _save_old_model(self):
        """Clone current model as old model."""
        self.old_model = create_model()
        self.old_model.load_state_dict(self.model.state_dict())
        self.old_model.to(self.device)
        self.old_model.eval()
    
    def generate_games(self, num_games: int) -> int:
        """Generate self-play games in parallel."""
        print(f'Generating {num_games} self-play games...')
        
        total_positions = 0
        workers = []
        
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            for i in range(num_games):
                worker = SelfPlayWorker(self.model, self.config, worker_id=i % 4)
                futures.append(executor.submit(worker.play_game))
            
            for i, future in enumerate(as_completed(futures)):
                try:
                    positions = future.result()
                    self.replay_buffer.add(positions)
                    total_positions += len(positions)
                    
                    if (i + 1) % max(1, num_games // 5) == 0:
                        print(f'  {i+1}/{num_games} games complete, '
                              f'{total_positions} positions buffered')
                except Exception as e:
                    print(f'  Game {i} failed: {e}')
        
        print(f'Generated {total_positions} training positions from {num_games} games')
        return total_positions
    
    def train_step(self, X: torch.Tensor, y_value: torch.Tensor, 
                   y_wdl: torch.Tensor) -> Dict:
        """Multi-task training step."""
        self.model.train()
        
        val_pred, pol_pred, wdl_pred, _ = self.model(X)
        
        losses = {}
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Value loss (MSE on outcome)
        if self.config.value_loss_weight > 0:
            v_loss = F.mse_loss(val_pred, y_value.unsqueeze(1))
            total_loss = total_loss + self.config.value_loss_weight * v_loss
            losses['value'] = v_loss.item()
        
        # WDL loss (cross-entropy on game outcome)
        if self.config.wdl_loss_weight > 0:
            wdl_loss = F.cross_entropy(wdl_pred, y_wdl)
            total_loss = total_loss + self.config.wdl_loss_weight * wdl_loss
            losses['wdl'] = wdl_loss.item()
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
        self.optimizer.step()
        
        return {**losses, 'total': total_loss.item()}
    
    def train_iteration(self):
        """One complete RL iteration: generate games + train."""
        self.iteration += 1
        print(f'\n{"="*60}')
        print(f'ITERATION {self.iteration}')
        print(f'{"="*60}')
        
        # Phase 1: Generate self-play games
        t_start = time.time()
        n_positions = self.generate_games(self.config.games_per_iteration)
        gen_time = time.time() - t_start
        
        # Phase 2: Train on replay buffer
        print(f'\nTraining on {len(self.replay_buffer):,} buffered positions...')
        train_start = time.time()
        
        for epoch in range(self.config.epochs_per_iteration):
            X_np, y_val, y_wdl = self.replay_buffer.sample(
                min(20000, len(self.replay_buffer))
            )
            
            X = torch.from_numpy(X_np).float().to(self.device)
            yv = torch.from_numpy(y_val).float().to(self.device)
            yw = torch.from_numpy(y_wdl).float().to(self.device)
            
            n = len(X)
            indices = np.random.permutation(n)
            epoch_losses = []
            
            for start in range(0, n, self.config.batch_size):
                idx = indices[start:start + self.config.batch_size]
                loss_dict = self.train_step(X[idx], yv[idx], yw[idx])
                epoch_losses.append(loss_dict['total'])
            
            avg_loss = np.mean(epoch_losses)
            
            if (epoch + 1) % max(1, self.config.epochs_per_iteration // 3) == 0:
                print(f'  Epoch {epoch+1}/{self.config.epochs_per_iteration}: '
                      f'loss={avg_loss:.4f}')
        
        train_time = time.time() - train_start
        
        # Phase 3: Arena comparison
        if self.old_model is not None:
            print(f'\nArena: new model vs old model ({self.config.arena_games} games)...')
            arena_results = self.arena.play_match(
                self.model, self.old_model, self.config.arena_games
            )
            
            print(f'  New: {arena_results["new_wins"]}W '
                  f'Old: {arena_results["old_wins"]}W '
                  f'Draws: {arena_results["draws"]}')
            print(f'  Win rate: {arena_results["new_win_rate"]:.1%} '
                  f'(+{arena_results["elo_diff"]:+.0f} Elo)')
            
            # Accept or reject
            if arena_results['new_win_rate'] >= self.config.acceptance_threshold:
                print(f'  >>> NEW MODEL ACCEPTED (+{arena_results["elo_diff"]:+.0f} Elo)')
                self._save_old_model()
                self.best_elo += arena_results['elo_diff']
                
                # Save accepted model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'iteration': self.iteration,
                    'elo_gain': arena_results['elo_diff'],
                    'total_elo_gain': self.best_elo,
                }, self.config.best_path)
            else:
                print(f'  >>> MODEL REJECTED (reverting to old)')
                # Revert model weights
                self.model.load_state_dict(self.old_model.state_dict())
        else:
            # First iteration, accept by default
            print(f'\nFirst iteration — model accepted as baseline')
            self._save_old_model()
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'iteration': self.iteration,
                'total_elo_gain': 0,
            }, self.config.best_path)
        
        # Summary
        total_time = time.time() - t_start
        print(f'\n--- Iteration {self.iteration} Summary ---')
        print(f'Games: {self.config.games_per_iteration} ({gen_time:.0f}s generate)')
        print(f'Training: {train_time:.0f}s')
        print(f'Total Elo gained: +{self.best_elo:.0f}')
        print(f'Replay buffer: {len(self.replay_buffer):,} positions')
        print(f'Iteration time: {total_time:.0f}s')
        
        # Periodic checkpoint
        if self.iteration % self.config.save_every == 0:
            ckpt_path = self.checkpoint_dir / f'iter_{self.iteration:04d}.pt'
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'iteration': self.iteration,
                'total_elo_gain': self.best_elo,
            }, ckpt_path)
    
    def train(self, num_iterations: int = 10):
        """Full RL training loop."""
        print(f'{"="*60}')
        print(f'SELF-PLAY RL TRAINING')
        print(f'Model: {count_parameters(self.model)[1]:,} params')
        print(f'Games per iteration: {self.config.games_per_iteration}')
        print(f'Iterations: {num_iterations}')
        print(f'Device: {self.device}')
        print(f'{"="*60}')
        
        for _ in range(num_iterations):
            self.train_iteration()
        
        print(f'\nRL Training complete!')
        print(f'Total Elo gained: +{self.best_elo:.0f}')
        print(f'Best model: {self.config.best_path}')


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='HyperTensor Self-Play RL')
    parser.add_argument('--games', type=int, default=500,
                       help='Self-play games per iteration')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of RL iterations')
    parser.add_argument('--model-size', default='xl',
                       choices=['small','medium','large','xl','xxl'])
    parser.add_argument('--model-path', default='models/sf_autopilot_best.pt',
                       help='Path to pretrained model')
    parser.add_argument('--search-depth', type=int, default=4,
                       help='Search depth for self-play')
    parser.add_argument('--arena-games', type=int, default=100,
                       help='Games for model comparison')
    parser.add_argument('--workers', type=int, default=4,
                       help='Parallel game workers')
    parser.add_argument('--no-resume', action='store_true')
    
    args = parser.parse_args()
    
    config = SelfPlayConfig(
        model_size=args.model_size,
        model_path=args.model_path,
        games_per_iteration=args.games,
        search_depth=args.search_depth,
        arena_games=args.arena_games,
        num_workers=args.workers,
    )
    
    trainer = SelfPlayTrainer(config)
    trainer.train(num_iterations=args.iterations)
