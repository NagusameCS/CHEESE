"""
HyperTensor Chess Engine v3.0 — Data Pipeline
==============================================
Generates training data through:
1. Deep MCTS self-play (bootstrap from random weights)
2. Position augmentation (symmetries, color flip)
3. Game outcome smoothing
4. Export to HDF5/numpy for efficient GPU training
"""

import numpy as np
import torch
import time
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque

from .board import Board, Move, Color, Piece, STARTING_FEN
from .evaluation import HyperTensorChessNet, CUDA_AVAILABLE, DEVICE
from .search import HyperTensorSearch

# Try python-chess for data validation
try:
    import chess
    _CHESS_AVAIL = True
except ImportError:
    _CHESS_AVAIL = False


@dataclass
class TrainingExample:
    board_tensor: np.ndarray   # (160, 8, 8)
    value_target: float         # [-1, 1]
    policy_target: np.ndarray   # (4096,)
    wdl_target: np.ndarray      # (3,)
    outcome: float              # Actual game result


class DataGenerator:
    """Generate high-quality training data via deep MCTS self-play."""
    
    def __init__(self, model: HyperTensorChessNet, sims_per_move: int = 400,
                 temperature: float = 1.0, exploration_noise: float = 0.25,
                 dirichlet_alpha: float = 0.3):
        self.model = model
        self.sims_per_move = sims_per_move
        self.temperature = temperature
        self.exploration_noise = exploration_noise
        self.dirichlet_alpha = dirichlet_alpha
        
        self.search = HyperTensorSearch(
            model, num_simulations=sims_per_move,
            use_jury=False, use_gtc=False,  # Pure MCTS for data generation
            use_safe_ogd=True, batch_size=32,
        )
    
    def generate_game(self) -> Tuple[List[TrainingExample], str, List[str]]:
        """Generate a single self-play game with deep MCTS.
        
        Returns:
            examples: Training examples from the game
            result: '1-0', '0-1', or '1/2-1/2'
            moves: List of UCI move strings
        """
        board = Board()
        positions = []
        game_moves = []
        
        move_num = 0
        while not board.is_game_over() and move_num < 80:  # Fast games for training
            root_tensor = board.to_tensor()
            
            best_move, stats = self.search.search(
                board, time_limit_ms=200  # Fast data generation
            )
            
            if best_move is None:
                break
            
            # Extract visit distribution from MCTS
            # (In production, we'd extract the full distribution from the tree)
            policy = np.zeros(4096, dtype=np.float32)
            move_idx = self._move_to_index(best_move, board)
            policy[move_idx] = 1.0
            
            # Store position
            positions.append({
                'tensor': root_tensor,
                'policy': policy,
                'move': best_move,
            })
            
            board.make_move(best_move)
            game_moves.append(best_move.uci())
            move_num += 1
            
            # Reset search stats for next move
            self.search.stats = {k: 0 for k in self.search.stats}
        
        # Game result
        result = board.result() or '1/2-1/2'
        
        # Create training examples
        if result == '1-0':
            outcome = 1.0
        elif result == '0-1':
            outcome = -1.0
        else:
            outcome = 0.0
        
        # WDL targets
        if outcome > 0:
            wdl = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif outcome < 0:
            wdl = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            wdl = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        examples = []
        for pos in positions:
            # Value target: outcome from perspective of side that played the move
            # All positions stored from white's perspective
            examples.append(TrainingExample(
                board_tensor=pos['tensor'].astype(np.float32),
                value_target=outcome,
                policy_target=pos['policy'],
                wdl_target=wdl.copy(),
                outcome=outcome,
            ))
        
        return examples, result, game_moves
    
    def generate_dataset(self, num_games: int, save_path: str = None) -> List[TrainingExample]:
        """Generate multiple games of training data."""
        all_examples = []
        results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
        
        print(f"Generating {num_games} games ({self.sims_per_move} sims/move)...")
        t0 = time.time()
        
        for gi in range(num_games):
            examples, result, moves = self.generate_game()
            all_examples.extend(examples)
            results[result] = results.get(result, 0) + 1
            
            if (gi + 1) % 5 == 0:
                elapsed = time.time() - t0
                print(f"  Game {gi+1}/{num_games}: {result} in {len(moves)} moves "
                      f"({elapsed/(gi+1):.1f}s/game, {len(all_examples)} examples)")
        
        total_time = time.time() - t0
        print(f"\nDataset: {len(all_examples)} examples from {num_games} games")
        print(f"Results: {results}")
        print(f"Time: {total_time:.0f}s ({total_time/num_games:.1f}s/game)")
        
        if save_path:
            self._save_dataset(all_examples, save_path)
        
        return all_examples
    
    def _move_to_index(self, move, board):
        base = move.from_sq * 64 + move.to_sq
        if move.promotion: base += 4096 - 256
        return base % 4096
    
    def _save_dataset(self, examples, path):
        """Save dataset as compressed numpy arrays."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        tensors = np.stack([ex.board_tensor for ex in examples])
        values = np.array([ex.value_target for ex in examples], dtype=np.float32)
        policies = np.stack([ex.policy_target for ex in examples])
        wdls = np.stack([ex.wdl_target for ex in examples])
        
        np.savez_compressed(path / 'training_data.npz',
                           tensors=tensors, values=values,
                           policies=policies, wdls=wdls)
        print(f"Saved {len(examples)} examples to {path}/training_data.npz")


class DataAugmentor:
    """Augment chess positions through symmetries."""
    
    @staticmethod
    def flip_horizontal(tensor: np.ndarray) -> np.ndarray:
        """Mirror board left-right (swap kingside/queenside)."""
        return np.flip(tensor, axis=-1).copy()
    
    @staticmethod
    def flip_color(tensor: np.ndarray) -> np.ndarray:
        """Swap white/black pieces (planes 0-5 ↔ 6-11)."""
        result = tensor.copy()
        result[:6], result[6:12] = result[6:12].copy(), result[:6].copy()
        # Flip value
        return result
    
    @staticmethod
    def rotate_180(tensor: np.ndarray) -> np.ndarray:
        """Rotate board 180 degrees."""
        return np.rot90(tensor, 2, axes=(1, 2)).copy()
    
    @classmethod
    def augment(cls, example: TrainingExample) -> List[TrainingExample]:
        """Generate augmented examples."""
        augmented = [example]
        
        # Horizontal flip
        h_tensor = cls.flip_horizontal(example.board_tensor)
        augmented.append(TrainingExample(
            h_tensor, example.value_target, example.policy_target,
            example.wdl_target, example.outcome))
        
        # 180 rotation
        r_tensor = cls.rotate_180(example.board_tensor)
        augmented.append(TrainingExample(
            r_tensor, example.value_target, example.policy_target,
            example.wdl_target, example.outcome))
        
        return augmented


# ===========================================================================
# Stockfish data bridge (if python-chess available)
# ===========================================================================

def generate_stockfish_data(positions: List[str], stockfish_path: str = None,
                           depth: int = 12) -> List[dict]:
    """Generate Stockfish-evaluated training positions.
    
    Requires python-chess and a Stockfish binary.
    """
    if not _CHESS_AVAIL:
        print("[WARN] python-chess not available for Stockfish data generation")
        return []
    
    import subprocess
    
    sf = None
    try:
        sf = subprocess.Popen(
            [stockfish_path or 'stockfish'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True
        )
        sf.stdin.write('uci\n'); sf.stdin.flush()
        # Read until uciok
        for line in sf.stdout:
            if 'uciok' in line: break
        
        results = []
        for fen_or_start in positions[:100]:  # Limit
            if fen_or_start == 'startpos':
                fen = STARTING_FEN
            else:
                fen = fen_or_start
            
            sf.stdin.write(f'position fen {fen}\n')
            sf.stdin.write(f'go depth {depth}\n')
            sf.stdin.flush()
            
            score = 0
            best_move = None
            for line in sf.stdout:
                if line.startswith('bestmove'):
                    best_move = line.split()[1]
                    break
                if 'score cp' in line:
                    score = int(line.split('score cp ')[1].split()[0])
                elif 'score mate' in line:
                    mate_in = int(line.split('score mate ')[1].split()[0])
                    score = 20000 if mate_in > 0 else -20000
            
            board = Board(fen)
            results.append({
                'fen': fen,
                'tensor': board.to_tensor(),
                'score_cp': score,
                'value': np.tanh(score / 400.0).astype(np.float32),  # Normalize
                'best_move': best_move,
                'wdl': _score_to_wdl(score),
            })
        
        sf.stdin.write('quit\n'); sf.stdin.flush()
        sf.wait(timeout=5)
        return results
    
    except Exception as e:
        print(f"[WARN] Stockfish data generation failed: {e}")
        if sf: sf.kill()
        return []


def _score_to_wdl(score_cp: int) -> np.ndarray:
    """Convert centipawn score to WDL probabilities."""
    # Sigmoid-based model from Leela/Stockfish data
    w = 1.0 / (1.0 + np.exp(-score_cp / 200.0))
    d = np.exp(-abs(score_cp) / 400.0) * 0.5
    l = 1.0 - w - d
    # Ensure non-negative
    w, d, l = max(0, w), max(0, d), max(0, l)
    s = w + d + l
    return np.array([w/s, d/s, l/s], dtype=np.float32)
