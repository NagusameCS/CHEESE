"""
HyperTensor Chess — NNUE-Style Feature Transformer
====================================================
Implements HalfKA (king-relative) feature set, the architecture behind
Stockfish NNUE's 3600+ Elo evaluation.

Key concepts:
  - HalfKA: For each (our_king_sq, enemy_king_sq, piece_sq), encode piece type
  - King bucketing: 64 squares → 16 buckets via mirroring
  - Feature transformer: Linear(40960, hidden) → ClippedReLU → Linear(hidden, 1)
  - Incrementally updatable (for CPU search speed)
  - INT8 quantizable (2-4× throughput)

Architecture Overview:
  Input: Board position (8×8 with pieces)
    ↓
  Feature Extractor: HalfKA feature indices (sparse binary features)
    ↓
  Feature Transformer: WideLinear(40960 → 256) + ClippedReLU
    ↓                    ↓
  Bucket Accumulator    Hidden layers (× N)
    ↓                    ↓
  Output: Centipawn eval + WDL

For GPU training, we use dense feature lookup with EmbeddingBag
(same math as sparse incremental update, but batched and parallelized).

Reference: https://www.chessprogramming.org/NNUE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, List

# Handle both direct execution and package import
try:
    from .board import Board, Color, Piece, PIECE_VALUES
    from .evaluation import CUDA_AVAILABLE, DEVICE
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from chess_engine.board import Board, Color, Piece, PIECE_VALUES
    from chess_engine.evaluation import CUDA_AVAILABLE, DEVICE


# ===========================================================================
# Piece Encoding for NNUE
# ===========================================================================

# NNUE piece encoding: 0=none, 1=wP, 2=wN, 3=wB, 4=wR, 5=wQ, 6=wK,
#                       7=bP, 8=bN, 9=bB, 10=bR, 11=bQ, 12=bK
# Total: 13 types (including empty)
NNUE_PIECE_NONE = 0
NNUE_PIECE_TYPES = 13

# Map our Piece enum to NNUE piece type
_PIECE_TO_NNUE = {
    None: 0,
    (Color.WHITE, Piece.PAWN): 1,
    (Color.WHITE, Piece.KNIGHT): 2,
    (Color.WHITE, Piece.BISHOP): 3,
    (Color.WHITE, Piece.ROOK): 4,
    (Color.WHITE, Piece.QUEEN): 5,
    (Color.WHITE, Piece.KING): 6,
    (Color.BLACK, Piece.PAWN): 7,
    (Color.BLACK, Piece.KNIGHT): 8,
    (Color.BLACK, Piece.BISHOP): 9,
    (Color.BLACK, Piece.ROOK): 10,
    (Color.BLACK, Piece.QUEEN): 11,
    (Color.BLACK, Piece.KING): 12,
}

# King bucket mapping: 64 squares → 16 buckets
# Mirror and rotate to reduce 64 king positions to 16 canonical buckets
def king_bucket(sq: int) -> int:
    """Map king square (0-63) to bucket (0-15) using symmetry."""
    rank, file = sq >> 3, sq & 7
    # Mirror to a1-d1-d4-a4 quadrant
    if file > 3: file = 7 - file
    if rank > 3: rank = 7 - rank
    # If file > rank, swap (triangular reduction)
    if file > rank:
        file, rank = rank, file
    # Now we have 10 buckets (a1-a4, b2-b4, c3-c4, d4)
    # Simple formula: bucket = rank * 4 + file (after mirroring)
    return rank * 4 + file


# ===========================================================================
# HalfKA Feature Extractor
# ===========================================================================

class HalfKAFeatureExtractor:
    """
    Extracts HalfKA feature indices from a board position.
    
    HalfKA: For each (king_sq_us, king_sq_them, piece_sq, piece_type),
    create a feature. Features are indexed by:
      idx = king_bucket * 64 * NNUE_PIECE_TYPES + sq * NNUE_PIECE_TYPES + piece_type
    
    This gives 16 × 64 × 13 = 13,312 features per perspective.
    Stockfish uses 2 perspectives (us + them) = 26,624 features.
    """
    
    NUM_FEATURES = 16 * 64 * NNUE_PIECE_TYPES  # 13,312 per perspective
    
    @staticmethod
    def extract_indices(board: Board) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract feature indices for NNUE.
        
        Returns:
            indices_us: array of feature indices for our perspective
            indices_them: array of feature indices for their perspective
        """
        our_color = board.color_to_move
        their_color = Color.BLACK if our_color == Color.WHITE else Color.WHITE
        
        # Find kings
        our_king_sq = HalfKAFeatureExtractor._find_king(board, our_color)
        their_king_sq = HalfKAFeatureExtractor._find_king(board, their_color)
        
        if our_king_sq is None or their_king_sq is None:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        
        # Bucket kings
        our_king_bucket = king_bucket(our_king_sq)
        their_king_bucket = king_bucket(their_king_sq)
        
        indices_us = []
        indices_them = []
        
        # For each square
        for sq in range(64):
            piece = board.piece_at(sq)
            nnue_type = _PIECE_TO_NNUE.get(piece, 0)
            
            # Feature: (our_king_bucket, sq, piece_type)
            feat_us = (our_king_bucket * 64 * NNUE_PIECE_TYPES + 
                       sq * NNUE_PIECE_TYPES + nnue_type)
            indices_us.append(feat_us)
            
            # Feature: (their_king_bucket, sq, piece_type) - for opponent perspective
            # Mirror the square for their perspective
            sq_mirrored = sq ^ 0o70  # Flip rank (like rank 1↔8)
            nnue_type_mirrored = HalfKAFeatureExtractor._mirror_piece(nnue_type)
            feat_them = (their_king_bucket * 64 * NNUE_PIECE_TYPES + 
                         sq_mirrored * NNUE_PIECE_TYPES + nnue_type_mirrored)
            indices_them.append(feat_them)
        
        return np.array(indices_us, dtype=np.int64), np.array(indices_them, dtype=np.int64)
    
    @staticmethod
    def extract_indices_combined(board: Board) -> np.ndarray:
        """
        Extract combined feature indices (us + them perspectives).
        Returns indices in range [0, 2 * NUM_FEATURES).
        """
        idx_us, idx_them = HalfKAFeatureExtractor.extract_indices(board)
        combined = np.concatenate([idx_us, idx_them + HalfKAFeatureExtractor.NUM_FEATURES])
        return combined
    
    @staticmethod
    def _find_king(board: Board, color: Color) -> Optional[int]:
        """Find king square for given color."""
        for sq in range(64):
            piece = board.piece_at(sq)
            if piece is not None and piece[0] == color and piece[1] == Piece.KING:
                return sq
        return None
    
    @staticmethod
    def _mirror_piece(nnue_type: int) -> int:
        """Mirror piece color (white <-> black)."""
        if nnue_type == 0:
            return 0
        if 1 <= nnue_type <= 6:  # White piece
            return nnue_type + 6  # → Black
        elif 7 <= nnue_type <= 12:  # Black piece
            return nnue_type - 6  # → White
        return nnue_type


# ===========================================================================
# NNUE Model (PyTorch)
# ===========================================================================

class NNUEFeatureTransformer(nn.Module):
    """
    Feature transformer: WideLinear(40960 → hidden) + ClippedReLU.
    
    For GPU training, we use EmbeddingBag which efficiently computes
    sum over feature embeddings. This is mathematically equivalent to
    the incremental accumulator update used in CPU NNUE inference.
    
    Args:
        num_features: Total number of HalfKA features (26624 for full set)
        hidden_dim: Output dimension (256 for Stockfish, 512 for us)
        clip_value: ReLU clamp value (127 for INT8 compatibility)
    """
    
    def __init__(self, num_features: int = 26624, hidden_dim: int = 512, 
                 clip_value: float = 127.0):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.clip_value = clip_value
        
        # Feature embedding weights: each feature contributes to hidden_dim
        # Shape: (num_features, hidden_dim)
        # Using EmbeddingBag for efficient sparse lookup on GPU
        self.embedding = nn.EmbeddingBag(
            num_features, hidden_dim, mode='sum', sparse=False
        )
        
        # Initialize with small weights (important for NNUE convergence)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(hidden_dim))
    
    def forward(self, feature_indices: torch.Tensor, 
                offsets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            feature_indices: (total_features,) - concatenated feature indices
                             for all positions in batch
            offsets: (batch_size,) - offsets into feature_indices for each position
        
        Returns:
            (batch_size, hidden_dim) - feature transformer output
        """
        # EmbeddingBag: sum embeddings for features in each position
        x = self.embedding(feature_indices, offsets)  # (batch, hidden_dim)
        
        # ClippedReLU: min(clip_value, max(0, x))
        x = torch.clamp(x, 0.0, self.clip_value)
        
        return x


class NNUEEvaluationHead(nn.Module):
    """NNUE evaluation head with hidden layers."""
    
    def __init__(self, input_dim: int = 512, hidden_layers: int = 3, 
                 hidden_dim: int = 32, output_dim: int = 1):
        super().__init__()
        layers = []
        
        # First layer: feature transformer output → hidden
        layers.append(nn.Linear(input_dim * 2, hidden_dim * 2))  # ×2 for us+them
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_dim * 2, hidden_dim * 2))
            layers.append(nn.ReLU())
        
        # Output heads
        self.shared = nn.Sequential(*layers)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Tanh()
        )
        self.wdl_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32), nn.ReLU(),
            nn.Linear(32, 3)  # Win/Draw/Loss logits
        )
    
    def forward(self, ft_us: torch.Tensor, ft_them: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ft_us: (batch, hidden_dim) - feature transformer output (our perspective)
            ft_them: (batch, hidden_dim) - feature transformer output (their perspective)
        
        Returns:
            value: (batch, 1) - centipawn evaluation in [-1, 1]
            wdl: (batch, 3) - WDL logits
        """
        # Concatenate both perspectives (Stockfish NNUE architecture)
        x = torch.cat([ft_us, ft_them], dim=1)  # (batch, hidden_dim * 2)
        x = self.shared(x)
        
        value = self.value_head(x)
        wdl = self.wdl_head(x)
        
        return value, wdl


class NNUEChessNet(nn.Module):
    """
    Complete NNUE-style chess evaluation network.
    
    Architecture (simplified NNUE for GPU training):
      Board → HalfKA Feature Indices (us + them combined, 128 features)
        → Feature Transformer (EmbeddingBag + ClippedReLU) → hidden_dim
        → Hidden Layers (hidden_dim → 32 → ...)
        → Value Head (centipawn) + WDL Head
    
    For GPU, we use a single combined feature space (no separate us/them
    accumulators). This is mathematically equivalent to the incremental
    NNUE approach but fully batched and GPU-friendly.
    """
    
    def __init__(self, 
                 hidden_dim: int = 512,
                 hidden_layers: int = 3,
                 head_hidden: int = 64,
                 clip_value: float = 127.0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Feature transformer: HalfKA features → hidden_dim
        # NUM_FEATURES * 2 because we encode both us+them perspectives
        # 128 features per position (64 squares × 2 perspectives)
        self.ft = NNUEFeatureTransformer(
            num_features=HalfKAFeatureExtractor.NUM_FEATURES * 2,
            hidden_dim=hidden_dim,
            clip_value=clip_value,
        )
        
        # Hidden layers
        layers = []
        in_dim = hidden_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, head_hidden))
            layers.append(nn.ReLU())
            in_dim = head_hidden
        
        self.body = nn.Sequential(*layers)
        
        # Output heads
        self.value_head = nn.Sequential(
            nn.Linear(in_dim, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Tanh()
        )
        self.wdl_head = nn.Sequential(
            nn.Linear(in_dim, 16), nn.ReLU(),
            nn.Linear(16, 3)  # Win/Draw/Loss logits
        )
        
        # Dummy policy head for interface compatibility
        self._dummy_policy = nn.Linear(in_dim, 4096)
    
    def extract_features(self, boards: List[Board]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract HalfKA feature indices for a list of boards.
        
        Returns:
            indices: (total_features,) - concatenated feature indices
            offsets: (batch_size,) - offsets for EmbeddingBag
        """
        all_indices = []
        offsets = []
        offset = 0
        
        for board in boards:
            idx = HalfKAFeatureExtractor.extract_indices_combined(board)
            offsets.append(offset)
            all_indices.extend(idx.tolist())
            offset += len(idx)
        
        indices_t = torch.tensor(all_indices, dtype=torch.long)
        offsets_t = torch.tensor(offsets, dtype=torch.long)
        
        return indices_t, offsets_t
    
    def forward(self, x: torch.Tensor = None, 
                feature_indices: torch.Tensor = None,
                offsets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Legacy board tensors (ignored if feature_indices provided)
            feature_indices: (total_features,) - pre-extracted indices
            offsets: (batch_size,) - offsets into feature_indices
        
        Returns:
            value: (batch, 1)
            policy_logits: (batch, 4096) - dummy
            wdl_logits: (batch, 3)
            kp: (batch, hidden_dim) - dummy
        """
        if feature_indices is not None and offsets is not None:
            device = feature_indices.device
            batch_size = len(offsets)
            
            # Feature transformer
            ft_output = self.ft(feature_indices, offsets)  # (batch, hidden_dim)
            
            # Body
            h = self.body(ft_output)  # (batch, head_hidden)
            
            # Heads
            value = self.value_head(h)
            wdl = self.wdl_head(h)
            policy = self._dummy_policy(h)
            kp = torch.zeros(batch_size, self.hidden_dim, device=device)
        else:
            # Legacy compat: no features provided
            batch_size = len(x) if x is not None else 1
            device = x.device if x is not None else torch.device('cpu')
            value = torch.zeros(batch_size, 1, device=device)
            wdl = torch.zeros(batch_size, 3, device=device)
            policy = torch.zeros(batch_size, 4096, device=device)
            kp = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        return value, policy, wdl, kp


# ===========================================================================
# Utility: Convert Board list to feature tensors
# ===========================================================================

def boards_to_features(boards: List[Board], device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a list of Board objects to NNUE feature tensors.
    
    Returns:
        board_tensors: (batch, 160, 8, 8) - for legacy compat
        feature_indices: (total_features,) - for NNUE
        offsets: (batch_size,) - for NNUE EmbeddingBag
    """
    # Legacy tensor (for compat)
    tensors = np.stack([b.to_tensor() for b in boards]).astype(np.float32)
    board_tensors = torch.from_numpy(tensors)
    
    # NNUE features
    all_indices = []
    offsets = []
    for board in boards:
        idx = HalfKAFeatureExtractor.extract_indices_combined(board)
        offsets.append(len(all_indices))
        all_indices.extend(idx.tolist())
    
    feature_indices = torch.tensor(all_indices, dtype=torch.long)
    offsets_t = torch.tensor(offsets, dtype=torch.long)
    
    if device is not None:
        board_tensors = board_tensors.to(device)
        feature_indices = feature_indices.to(device)
        offsets_t = offsets_t.to(device)
    
    return board_tensors, feature_indices, offsets_t


# ===========================================================================
# NNUE Model Factory
# ===========================================================================

def create_nnue_model(hidden_dim: int = 512, hidden_layers: int = 3) -> NNUEChessNet:
    """Create an NNUE-style model."""
    return NNUEChessNet(
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
    )


def count_nnue_parameters(model: NNUEChessNet) -> Tuple[int, str]:
    """Count parameters in NNUE model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Breakdown
    ft_params = sum(p.numel() for p in model.ft.parameters())
    body_params = sum(p.numel() for p in model.body.parameters())
    head_params = (sum(p.numel() for p in model.value_head.parameters()) +
                   sum(p.numel() for p in model.wdl_head.parameters()))
    
    if total > 1_000_000:
        size_str = f'{total/1_000_000:.1f}M'
    else:
        size_str = f'{total/1_000:.0f}K'
    
    print(f'NNUE Model: {size_str} params '
          f'(FT: {ft_params:,}, Head: {head_params:,})')
    
    return total, size_str


# ===========================================================================
# Test
# ===========================================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    print('Testing NNUE Feature Extractor...')
    
    from chess_engine.board import Board
    
    board = Board()  # Startpos
    
    # Extract features
    idx_us, idx_them = HalfKAFeatureExtractor.extract_indices(board)
    idx_combined = HalfKAFeatureExtractor.extract_indices_combined(board)
    
    print(f'  Features (us):   {len(idx_us)}')
    print(f'  Features (them): {len(idx_them)}')
    print(f'  Combined:        {len(idx_combined)}')
    print(f'  Expected:        128 (64 squares × 2 colors)')
    
    # Create model
    model = create_nnue_model(hidden_dim=256, hidden_layers=2)
    count_nnue_parameters(model)
    
    # Test forward pass
    boards = [Board(), Board('rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6')]
    _, fi, off = boards_to_features(boards)
    
    with torch.no_grad():
        val, pol, wdl, kp = model(
            torch.randn(2, 160, 8, 8),  # dummy legacy tensor
            feature_indices=fi,
            offsets=off,
        )
    
    print(f'  Value: {val.squeeze().tolist()}')
    print(f'  WDL:   {wdl.tolist()}')
    print(f'  Policy: {pol.shape}')
    print('\nNNUE Feature Extractor working!')
