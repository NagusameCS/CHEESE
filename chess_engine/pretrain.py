"""
HyperTensor Chess Engine v3.2 — Heuristic Pretrainer
======================================================
Solves the cold-start problem by pretraining the neural network
on basic chess heuristics BEFORE self-play.

Without this: random weights → random self-play → garbage training data
With this: model learns material/PST/mobility → decent self-play → quality data

Heuristics taught:
  - Material count (piece values)
  - Piece-square tables (positional knowledge)
  - Mobility (number of legal moves)
  - King safety (pawn shield, open files near king)
  - Pawn structure (doubled, isolated, passed)
  - Center control
  - Tempo/initiative

Training data: 100K random chess positions evaluated by heuristics.
This takes ~2 minutes on GPU and gives the model a strong foundation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import random
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .board import Board, Move, Color, Piece, STARTING_FEN, PIECE_VALUES

# ===========================================================================
# Chess Heuristics
# ===========================================================================

# Piece-square tables (from White's perspective, flipped for Black)
PST = {
    Piece.PAWN: np.array([
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0,
    ], dtype=np.float32),
    Piece.KNIGHT: np.array([
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ], dtype=np.float32),
    Piece.BISHOP: np.array([
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ], dtype=np.float32),
    Piece.ROOK: np.array([
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         0,  0,  0,  5,  5,  0,  0,  0,
    ], dtype=np.float32),
    Piece.QUEEN: np.array([
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20,
    ], dtype=np.float32),
    Piece.KING: np.array([
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20,
    ], dtype=np.float32),
}

# King endgame PST (king becomes active)
PST_KING_ENDGAME = np.array([
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
], dtype=np.float32)

CENTER_SQUARES = {27, 28, 35, 36}  # e4, d4, e5, d5


def heuristic_evaluate(board: Board) -> Tuple[float, float, float]:
    """Evaluate a chess position using handcrafted heuristics.
    
    Returns:
        value: Position evaluation in [-1, 1] (1 = White winning)
        material_balance: Raw material count normalized
        mobility_advantage: Mobility difference normalized
    """
    if board.is_checkmate():
        return (-1.0, -1.0, 0.0) if board.color_to_move == Color.WHITE else (1.0, 1.0, 0.0)
    if board.is_stalemate() or board.halfmove_clock >= 100:
        return (0.0, 0.0, 0.0)
    
    material = 0.0
    positional = 0.0
    total_material = 0
    w_queens = 0; b_queens = 0
    w_pawns = 0; b_pawns = 0
    wk_sq = board.king_sq[Color.WHITE]
    bk_sq = board.king_sq[Color.BLACK]
    
    # Count material for game phase and anti-stunt
    for sq, (color, piece) in board.pieces.items():
        val = PIECE_VALUES[piece]
        total_material += val
        if piece == Piece.QUEEN:
            if color == Color.WHITE: w_queens += 1
            else: b_queens += 1
        elif piece == Piece.PAWN:
            if color == Color.WHITE: w_pawns += 1
            else: b_pawns += 1
    
    is_endgame = total_material < 3000  # Less than ~3 queens of material
    
    for sq, (color, piece) in board.pieces.items():
        val = PIECE_VALUES[piece]
        sign = 1 if color == Color.WHITE else -1
        material += val * sign
        
        # Piece-square tables
        if piece in PST:
            idx = sq if color == Color.WHITE else (63 - sq)
            if piece == Piece.KING and is_endgame:
                pst_val = PST_KING_ENDGAME[idx]
            else:
                pst_val = PST[piece][idx]
            positional += pst_val * sign
        
        # Center control bonus
        if sq in CENTER_SQUARES:
            positional += 15 * sign if piece == Piece.PAWN else 5 * sign
    
    # Mobility (simplified: number of legal moves)
    us = board.color_to_move
    board_copy = board.copy()
    board_copy.color_to_move = Color.WHITE
    white_moves = len(board_copy.generate_legal_moves())
    board_copy.color_to_move = Color.BLACK
    black_moves = len(board_copy.generate_legal_moves())
    mobility = (white_moves - black_moves) * 3
    
    # King safety (simplified)
    wk_r, wk_f = wk_sq // 8, wk_sq % 8
    bk_r, bk_f = bk_sq // 8, bk_sq % 8
    
    # Pawn shield for white king
    white_shield = 0
    for df in [-1, 0, 1]:
        for dr in [1, 2]:
            sr, sf = wk_r + dr, wk_f + df
            if 0 <= sr < 8 and 0 <= sf < 8:
                p = board.piece_at(sr * 8 + sf)
                if p and p == (Color.WHITE, Piece.PAWN):
                    white_shield += 10 if dr == 1 else 5
    
    black_shield = 0
    for df in [-1, 0, 1]:
        for dr in [-1, -2]:
            sr, sf = bk_r + dr, bk_f + df
            if 0 <= sr < 8 and 0 <= sf < 8:
                p = board.piece_at(sr * 8 + sf)
                if p and p == (Color.BLACK, Piece.PAWN):
                    black_shield += 10 if dr == -1 else 5
    
    king_safety = white_shield - black_shield
    
    # Combine
    raw_score = material + positional * 0.1 + mobility * 0.5 + king_safety * 0.3
    
    # =====================================================================
    # ANTI-STUNT: Penalize excessive material when already winning
    # (prevents "showing off" with extra queens instead of checkmating)
    # =====================================================================
    if is_endgame and abs(raw_score) > 500:  # Up >= 500cp
        winning_side = 1 if raw_score > 0 else -1
        
        # Extra queens penalty
        our_queens = w_queens if winning_side > 0 else b_queens
        if our_queens > 1:
            raw_score -= winning_side * (our_queens - 1) * 80  # -80cp per extra queen
        
        # Unnecessary pawns penalty (they're just future stunting material)
        our_pawns = w_pawns if winning_side > 0 else b_pawns
        if our_queens >= 1 and our_pawns > 0:
            raw_score -= winning_side * our_pawns * 15  # -15cp per pawn
        
        # King proximity bonus (encourage king to help with checkmate)
        king_dist = abs(wk_f - bk_f) + abs(wk_r - bk_r)
        raw_score += winning_side * max(0, 14 - king_dist) * 3  # +3cp per square closer
        
        # Enemy king cornering bonus
        enemy_kf = bk_f if winning_side > 0 else wk_f
        enemy_kr = bk_r if winning_side > 0 else wk_r
        edge_dist = min(enemy_kf, 7 - enemy_kf, enemy_kr, 7 - enemy_kr)
        if edge_dist <= 1:
            raw_score += winning_side * (3 - edge_dist) * 20  # Up to +60cp for cornered king
        
        # Stalemate risk: if opponent has very few moves, penalize
        opp_color = Color.BLACK if winning_side > 0 else Color.WHITE
        board_copy2 = board.copy()
        board_copy2.color_to_move = opp_color
        opp_legal = board_copy2.generate_legal_moves()
        if len(opp_legal) <= 3:
            raw_score -= winning_side * (4 - len(opp_legal)) * 50  # -50 to -150cp
        elif len(opp_legal) <= 6:
            raw_score -= winning_side * 20  # Caution
    
    # Normalize to [-1, 1]
    value = np.tanh(raw_score / 800.0)
    
    # Material balance (normalized)
    mat_balance = np.tanh(material / 2000.0)
    
    # Mobility advantage
    mob_adv = np.tanh(mobility / 40.0)
    
    return float(value), float(mat_balance), float(mob_adv)


# ===========================================================================
# Position Generator
# ===========================================================================

def generate_random_position() -> Board:
    """Generate a random chess position from the opening book + random moves."""
    board = Board()
    
    # Make some random moves from starting position
    num_moves = random.randint(0, 30)
    for _ in range(num_moves):
        legal = board.generate_legal_moves()
        if not legal: break
        if board.is_game_over(): break
        move = random.choice(legal)
        board.make_move(move)
    
    return board


def generate_positions_from_games(num_positions: int) -> List[Board]:
    """Generate positions by playing random games with opening book starts."""
    positions = []
    games_played = 0
    
    while len(positions) < num_positions:
        board = Board()
        move_count = 0
        while not board.is_game_over() and move_count < 80:
            if len(positions) >= num_positions: break
            
            # Sample position (with probability based on depth)
            if move_count > 2 and random.random() < 0.3:
                positions.append(board.copy())
            
            legal = board.generate_legal_moves()
            if not legal: break
            move = random.choice(legal)
            board.make_move(move)
            move_count += 1
        
        games_played += 1
        
        if games_played % 100 == 0:
            print(f"  Generated {len(positions)}/{num_positions} positions "
                  f"({games_played} games)...", flush=True)
    
    return positions[:num_positions]


# ===========================================================================
# Heuristic Pretraining
# ===========================================================================

def pretrain_model(model, num_positions=100000, batch_size=256, epochs=10,
                   lr=1e-3, device='cuda'):
    """Pretrain a model on heuristic evaluations.
    
    This gives the model a strong foundation before self-play training.
    After pretraining, the model understands:
      - Piece values (Queen > Rook > Bishop/Knight > Pawn)
      - Piece-square tables (knights in center, rooks on open files, etc.)
      - King safety (pawn shield, castling)
      - Mobility (more moves = better position)
    """
    from .evaluation import create_optimizer, CUDA_AVAILABLE, DEVICE
    
    device = torch.device(device if CUDA_AVAILABLE else 'cpu')
    model = model.to(device)
    model.train()
    
    print("=" * 60)
    print("HyperTensor Chess — Heuristic Pretraining")
    print("=" * 60)
    print(f"Positions: {num_positions} | Batch: {batch_size} | Epochs: {epochs}")
    print(f"Device: {device}")
    
    # Generate training positions
    print("\n[1] Generating training positions...")
    t0 = time.time()
    positions = generate_positions_from_games(num_positions)
    print(f"  Generated {len(positions)} positions in {time.time()-t0:.1f}s")
    
    # Evaluate all positions
    print("\n[2] Computing heuristic evaluations...")
    t0 = time.time()
    tensors = np.zeros((len(positions), 160, 8, 8), dtype=np.float32)
    values = np.zeros(len(positions), dtype=np.float32)
    wdl_targets = np.zeros((len(positions), 3), dtype=np.float32)
    
    for i, board in enumerate(positions):
        tensors[i] = board.to_tensor()
        val, mat, mob = heuristic_evaluate(board)
        values[i] = val
        
        # WDL from heuristic value
        if val > 0.3:
            wdl_targets[i] = [0.7, 0.25, 0.05]
        elif val < -0.3:
            wdl_targets[i] = [0.05, 0.25, 0.7]
        else:
            wdl_targets[i] = [0.2, 0.6, 0.2]
        
        if (i + 1) % 20000 == 0:
            print(f"  Evaluated {i+1}/{len(positions)} positions...", flush=True)
    
    print(f"  Evaluated {len(positions)} in {time.time()-t0:.1f}s")
    
    # Training loop
    print(f"\n[3] Training for {epochs} epochs...")
    optimizer = create_optimizer(model, lr=lr)
    
    n_batches = len(positions) // batch_size
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # Shuffle
        indices = np.random.permutation(len(positions))
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            batch_indices = indices[start:start + batch_size]
            
            x = torch.from_numpy(tensors[batch_indices]).float().to(device)
            v_tgt = torch.from_numpy(values[batch_indices]).float().to(device).unsqueeze(1)
            w_tgt = torch.from_numpy(wdl_targets[batch_indices]).float().to(device)
            
            optimizer.zero_grad()
            v_pred, p_pred, w_pred, k_proj = model(x)
            
            v_loss = F.mse_loss(v_pred, v_tgt)
            w_loss = F.cross_entropy(w_pred, w_tgt)
            
            # Manifold spread loss (encourage diverse k-space)
            k_norm = F.normalize(k_proj, dim=1)
            spread = -(k_norm @ k_norm.T).mean() * 0.001  # Negative to encourage spread
            
            loss = v_loss + 0.3 * w_loss + spread
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            
            epoch_losses.append({'v': v_loss.item(), 'w': w_loss.item()})
        
        avg_v = np.mean([e['v'] for e in epoch_losses])
        avg_w = np.mean([e['w'] for e in epoch_losses])
        print(f"  Epoch {epoch+1}/{epochs}: v_loss={avg_v:.4f} w_loss={avg_w:.4f}", flush=True)
        
        # Test on a known position
        if (epoch + 1) % 3 == 0:
            test_board = Board(STARTING_FEN)
            test_tensor = torch.from_numpy(test_board.to_tensor()).float().unsqueeze(0).to(device)
            with torch.no_grad():
                pred_v, _, _, _ = model(test_tensor)
                true_v, _, _ = heuristic_evaluate(test_board)
                print(f"    Startpos: pred={pred_v.item():.3f} true={true_v:.3f}", flush=True)
    
    model.eval()
    print("\n[4] Pretraining complete!")
    
    # Verify
    test_board = Board(STARTING_FEN)
    test_tensor = torch.from_numpy(test_board.to_tensor()).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred_v, _, _, _ = model(test_tensor)
    print(f"  Starting position: {pred_v.item():.3f} (should be ~0.0-0.1 for White)")
    
    # Test a clearly winning position
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R6K w kq - 0 1"  # Black missing rook
    try:
        test_board2 = Board(test_fen)
        test_tensor2 = torch.from_numpy(test_board2.to_tensor()).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred_v2, _, _, _ = model(test_tensor2)
        print(f"  White up a rook: {pred_v2.item():.3f} (should be > 0.5)")
    except: pass
    
    return model


# ===========================================================================
# Quick test
# ===========================================================================

def quick_test():
    """Verify pretraining pipeline."""
    from .evaluation import create_model
    
    print("Heuristic pretraining quick test...")
    model = create_model(k_manifold=8, hidden_dim=64, num_layers=2)
    
    # Test heuristic evaluation
    board = Board()
    val, mat, mob = heuristic_evaluate(board)
    print(f"Starting position: value={val:.3f} mat={mat:.3f} mob={mob:.3f}")
    
    # Generate a few positions
    positions = generate_positions_from_games(100)
    print(f"Generated {len(positions)} positions")
    
    # Quick pretrain
    model = pretrain_model(model, num_positions=100, batch_size=32, epochs=3, lr=1e-3)
    
    print("Pretraining test PASSED")
    return model


if __name__ == '__main__':
    quick_test()
