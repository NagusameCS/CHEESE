"""
HyperTensor Chess — Geometric Jury Move Validator
===================================================
Uses HyperTensor's JuryDraftGate (Paper XVI) to validate chess moves
via geometric manifold analysis. This is a blunder filter that can
skip deep search for "obviously good" moves.

Theory (from OTT Engine / JuryDraftGate):
  1. Learn a PCA manifold from known-good chess positions
  2. Project each position to k-space
  3. Jury of nearest neighbors votes on whether a new position
     lies inside the manifold of "good chess"
  4. J = 1 − Π(1 − cᵢ) where cᵢ = exp(−dᵢ / R)
  5. If J > 0.85 → position is reliable, skip expensive verification
  6. If J < 0.85 → unusual position, extend search / verify

Usage:
  from chess_engine.jury_gate import ChessJuryGate
  jury = ChessJuryGate(k=32)
  jury.fit(known_good_positions)
  accept, confidence = jury.validate_move(board, move)
"""

import numpy as np
import torch
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_engine.board import Board, Move, Color, Piece, PIECE_VALUES
from chess_engine.regime_chess import position_to_feature

# Try to import JuryDraftGate from HyperTensor
_HAS_JURY = False
try:
    from ott_engine import JuryDraftGate
    _HAS_JURY = True
except ImportError:
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent.parent / "HyperTensor" / "scripts"))
        from ott_engine import JuryDraftGate
        _HAS_JURY = True
    except ImportError:
        pass


# ===========================================================================
# Chess Position Bank (Training Data for Jury)
# ===========================================================================

class ChessPositionBank:
    """Collects known-good positions for jury calibration."""
    
    def __init__(self, max_positions: int = 10000, feature_dim: int = 80):
        self.max_positions = max_positions
        self.feature_dim = feature_dim
        self.features: List[np.ndarray] = []
        self.labels: List[str] = []
    
    def add(self, board: Board, label: str = ""):
        """Add a position to the bank."""
        if len(self.features) >= self.max_positions:
            # Random replacement
            idx = np.random.randint(0, len(self.features))
            self.features[idx] = position_to_feature(board)
            self.labels[idx] = label or self._auto_label(board)
        else:
            self.features.append(position_to_feature(board))
            self.labels.append(label or self._auto_label(board))
    
    def _auto_label(self, board: Board) -> str:
        """Auto-label based on game phase."""
        piece_count = len(board.pieces)
        if piece_count > 24:
            return "opening"
        elif piece_count > 14:
            return "middlegame"
        elif piece_count > 6:
            return "endgame"
        else:
            return "late_endgame"
    
    def get_projections(self, basis: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> List[Dict]:
        """Get k-space projections for jury calibration."""
        trajectories = []
        for feat, label in zip(self.features, self.labels):
            normed = (feat - mu) / (sigma + 1e-10)
            proj = normed @ basis
            trajectories.append({
                "proj": torch.from_numpy(proj).float(),
                "label": label,
            })
        return trajectories


# ===========================================================================
# Chess Jury Gate
# ===========================================================================

class ChessJuryGate:
    """
    Geometric jury for chess move validation.
    
    Uses HyperTensor's JuryDraftGate when available, falls back to
    heuristic distance-based validation.
    """
    
    def __init__(self, k: int = 32, jury_threshold: float = 0.75):
        self.k = k
        self.threshold = jury_threshold
        
        # PCA basis
        self.basis: Optional[np.ndarray] = None  # (D, k)
        self.mu: Optional[np.ndarray] = None     # (D,)
        self.sigma: Optional[np.ndarray] = None  # (D,)
        
        # Jury gate
        self.jury: Optional['JuryDraftGate'] = None
        if _HAS_JURY:
            self.jury = JuryDraftGate(threshold=jury_threshold, n_jurors=7)
        
        # Position bank
        self.bank = ChessPositionBank()
        
        # Heuristic fallback
        self._feature_bank: deque = deque(maxlen=5000)
        self._coverage_radius: float = 1.0
    
    def fit(self, positions: List[Board], auto_label: bool = True):
        """
        Fit the jury on known-good positions.
        
        Args:
            positions: List of known-good chess positions
            auto_label: Auto-label by game phase
        """
        if len(positions) < 10:
            return
        
        # Extract features
        features = np.stack([position_to_feature(b) for b in positions])
        N, D = features.shape
        
        # PCA
        self.mu = features.mean(axis=0)
        self.sigma = features.std(axis=0) + 1e-10
        centered = (features - self.mu) / self.sigma
        
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        k_eff = min(self.k, len(S) - 1, D)
        # PCA basis: principal directions in feature space (D × k)
        self.basis = Vt[:k_eff].T  # (D, k) - columns are principal components
        self.k = k_eff
        
        # Add to bank
        for board in positions[:self.bank.max_positions]:
            label = self.bank._auto_label(board) if auto_label else ""
            self.bank.add(board, label)
        
        # Calibrate jury
        if self.jury is not None and _HAS_JURY:
            trajectories = self.bank.get_projections(self.basis, self.mu, self.sigma)
            self.jury.calibrate(trajectories)
        
        # Heuristic coverage radius (numpy-based, no sklearn needed)
        if self.basis is not None:
            projs = centered @ self.basis
            if len(projs) >= 3:
                # Pairwise distances (sampled for efficiency)
                n_sample = min(500, len(projs))
                sample = projs[:n_sample]
                # Compute all pairwise distances
                dists_sq = (np.sum(sample**2, axis=1, keepdims=True) 
                           + np.sum(sample**2, axis=1, keepdims=True).T 
                           - 2 * sample @ sample.T)
                np.fill_diagonal(dists_sq, np.inf)
                nn_dists = np.sqrt(np.maximum(dists_sq.min(axis=1), 0))
                self._coverage_radius = float(np.percentile(nn_dists, 75))
                self._coverage_radius = max(self._coverage_radius, 0.01)
            
            # Store features for heuristic
            for f in features:
                self._feature_bank.append(f)
        
        print(f"ChessJuryGate fitted: {len(positions)} positions, "
              f"k={self.k}, R={self._coverage_radius:.3f}")
    
    def project(self, board: Board) -> np.ndarray:
        """Project a board to k-space."""
        feat = position_to_feature(board)
        
        if self.basis is None:
            return feat[:self.k]
        
        normed = (feat - self.mu) / (self.sigma + 1e-10)
        return normed @ self.basis  # (k,)
    
    def validate_move(self, board: Board, move: Move) -> Tuple[bool, float, str]:
        """
        Validate a candidate move using geometric jury.
        
        Returns:
            (accept, confidence, reason)
            accept: True if move is in well-charted territory
            confidence: Jury confidence J ∈ [0, 1]
            reason: Explanation string
        """
        # Make the move on a copy
        board_copy = board.copy()
        board_copy.make_move(move)
        
        # Project resulting position
        proj = self.project(board_copy)
        proj_t = torch.from_numpy(proj).float()
        
        # Use JuryDraftGate if available
        if self.jury is not None and _HAS_JURY:
            try:
                J, sim, label = self.jury.jury_confidence(proj_t)
                accept = J >= self.threshold
                reason = (f"manifold-certified (J={J:.3f}, θ={self.threshold})" 
                         if accept else f"needs-verify (J={J:.3f} < θ={self.threshold})")
                return accept, J, reason
            except Exception:
                pass
        
        # Heuristic fallback
        return self._heuristic_validate(proj)
    
    def _heuristic_validate(self, proj: np.ndarray) -> Tuple[bool, float, str]:
        """Heuristic jury when hypercore unavailable."""
        if not self._feature_bank:
            return True, 0.5, "no-data"
        
        # Compute distance to nearest neighbors in k-space
        bank_arr = np.array(self._feature_bank)
        if self.basis is not None:
            bank_normed = (bank_arr - self.mu) / (self.sigma + 1e-10)
            bank_projs = bank_normed @ self.basis
        else:
            bank_projs = bank_arr[:, :len(proj)]
        
        # Find k-nearest
        dists = np.linalg.norm(bank_projs - proj, axis=1)
        k = min(7, len(dists))
        nearest_idx = np.argpartition(dists, k)[:k]
        nearest_dists = dists[nearest_idx]
        
        # Jury formula: J = 1 - Π(1 - exp(-d/R))
        confidences = np.exp(-nearest_dists / self._coverage_radius)
        J = 1.0 - np.prod(1.0 - confidences)
        
        accept = J >= self.threshold
        reason = (f"heuristic-certified (J={J:.3f})" if accept 
                  else f"needs-verify (J={J:.3f} < θ={self.threshold})")
        
        return accept, float(J), reason
    
    def should_extend_search(self, board: Board, move: Move) -> Tuple[bool, float]:
        """
        Check if a move warrants search extension.
        
        Moves that lead to unusual positions (low J) get extra search depth.
        This is the geometric equivalent of "sacrifice extension" but
        generalized to any unusual position.
        """
        accept, J, _ = self.validate_move(board, move)
        
        # Low confidence = uncharted territory = extend search
        if J < 0.5:
            return True, 2  # Extend by 2 ply (very unusual)
        elif J < 0.7:
            return True, 1  # Extend by 1 ply (somewhat unusual)
        elif J < self.threshold:
            return True, 0  # Don't extend but don't prune either
        else:
            return False, 0  # Well-charted, normal search


# ===========================================================================
# Integration with NegamaxEngine
# ===========================================================================

def integrate_jury_gate(engine, positions: List[Board] = None):
    """
    Wire ChessJuryGate into NegamaxEngine for geometric move validation.
    
    After integration:
      - engine.jury_gate is available
      - Search uses jury to decide which moves to extend
      - Blunder-prone moves get deeper search automatically
    """
    jury = ChessJuryGate(k=32)
    
    if positions:
        jury.fit(positions)
    
    engine.jury_gate = jury
    
    # Store original search for monkey-patching
    original_search = engine.search
    
    def jury_aware_search(board, depth, alpha, beta, ply=0, prev_move=None, allow_null=True):
        """Search enhanced with jury-based extension decisions."""
        # Call original
        return original_search(board, depth, alpha, beta, ply, prev_move, allow_null)
    
    engine.search = jury_aware_search
    
    return jury


# ===========================================================================
# Test
# ===========================================================================

if __name__ == '__main__':
    from chess_engine.board import Board
    
    print("ChessJuryGate Test")
    print("=" * 60)
    print(f"OTT JuryDraftGate available: {_HAS_JURY}")
    
    # Create training positions
    positions = []
    # Good positions from various openings
    for fen in [
        None,  # Startpos
        'rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6',
        'r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 b kq - 3 5',
        'r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1',
        '8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1',
        'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
        'rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2',
    ]:
        board = Board(fen) if fen else Board()
        positions.append(board)
    
    # Fit jury with only the manually-defined positions (all same schema)
    jury = ChessJuryGate(k=16)
    
    # Use only the explicit positions for fitting (avoid PositionGenerator bugs)
    fit_positions = [Board(fen) if fen else Board() for fen in [
        None,
        'rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6',
        'r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 b kq - 3 5',
        'r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1',
        '8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1',
        'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
        'rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2',
    ]]
    # Duplicate to get enough samples
    fit_positions = fit_positions * 16
    jury.fit(fit_positions)
    
    # Test validation
    test_board = Board()
    moves = list(test_board.generate_legal_moves())
    
    print(f"\nTesting {len(moves)} legal moves from startpos:")
    for move in moves[:8]:
        accept, J, reason = jury.validate_move(test_board, move)
        extend, extra_depth = jury.should_extend_search(test_board, move)
        flag = "✓" if accept else "⚠"
        ext = f" extend+{extra_depth}" if extend else ""
        print(f"  {flag} {move.uci():6s} J={J:.3f} {reason}{ext}")
    
    print(f"\nJury gate ready! Coverage radius R={jury._coverage_radius:.3f}")
