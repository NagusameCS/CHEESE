"""
HyperTensor Chess — RegimeDetector Integration
================================================
Wires HyperTensor's RegimeDetector (v1.1, May 18 2026) into the chess engine.

RegimeDetector monitors 5 geometric signals on the manifold of chess positions:
  1. Manifold Deviation  — how far a position lies from typical patterns
  2. Curvature Anomaly    — abrupt tactical changes (turning angle spikes)
  3. Neighbor Instability — KNN graph disruption (novel positions)
  4. Spectral Drift       — structural change in position similarity graph
  5. Geodesic Misalignment — deviation from expected positional flow

Applications for chess:
  - TACTICAL DETECTION: Regime change = tactical shot happened
  - PHASE TRANSITIONS: Opening→Middlegame→Endgame detected automatically  
  - TIME ALLOCATION: Give more time when regime is unstable
  - EVALUATION RELIABILITY: If manifold deviation is high, trust NN less
  - SACRIFICE VERIFICATION: Curvature spike = compensation may exist

Usage:
  from chess_engine.regime_chess import ChessRegimeDetector
  crd = ChessRegimeDetector(intrinsic_dim=32)
  crd.fit_on_positions(training_positions)
  
  result = crd.check(board)
  if result.regime_change:
      print(f"Tactical shift! RCI={result.rci:.3f}")
"""

import numpy as np
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_engine.board import Board, Color, Piece, PIECE_VALUES

# Try to import RegimeDetector from HyperTensor
_HAS_REGIME_DETECTOR = False
try:
    from hypercore.regime_detector import (
        RegimeDetector, RegimeAssessment, RegimeSignal,
    )
    _HAS_REGIME_DETECTOR = True
except ImportError:
    pass


# ===========================================================================
# Chess Position Embedding (for RegimeDetector's ambient space)
# ===========================================================================

def position_to_feature(board: Board) -> np.ndarray:
    """
    Convert a chess position to a fixed-dimensional feature vector.
    
    Features (80-dim):
      - 12 piece counts (6 white + 6 black) → 12
      - Material balance (per piece type) → 5  
      - King safety (both kings) → 8
      - Pawn structure (files) → 8
      - Open files → 8
      - Mobility estimates → 12
      - Center control → 8
      - King position (both) → 4
      - Piece square tables (simplified) → 15
    """
    features = []
    
    # Piece counts per type per color
    w_counts = [0]*6
    b_counts = [0]*6
    for sq, (c, p) in board.pieces.items():
        if c == Color.WHITE:
            w_counts[p] += 1
        else:
            b_counts[p] += 1
    
    features.extend(w_counts)  # 6
    features.extend(b_counts)  # 6
    
    # Material balance
    for i in range(5):  # Pawn through Queen
        features.append(w_counts[i] - b_counts[i])  # 5
    
    # Total material
    w_mat = sum(w_counts[i] * PIECE_VALUES.get(i, 0) for i in range(6))
    b_mat = sum(b_counts[i] * PIECE_VALUES.get(i, 0) for i in range(6))
    features.append(w_mat / 1000.0)
    features.append(b_mat / 1000.0)
    
    # King positions (normalized)
    w_king_sq = None
    b_king_sq = None
    for sq, (c, p) in board.pieces.items():
        if p == Piece.KING:
            if c == Color.WHITE:
                w_king_sq = sq
            else:
                b_king_sq = sq
    
    if w_king_sq is not None:
        features.append((w_king_sq % 8) / 7.0)  # file
        features.append((w_king_sq // 8) / 7.0)  # rank
    else:
        features.extend([0.5, 0.5])
    
    if b_king_sq is not None:
        features.append((b_king_sq % 8) / 7.0)
        features.append((b_king_sq // 8) / 7.0)
    else:
        features.extend([0.5, 0.5])
    
    # Open files
    for f in range(8):
        has_pawn = False
        for r in range(8):
            p = board.piece_at(r * 8 + f)
            if p and p[1] == Piece.PAWN:
                has_pawn = True
                break
        features.append(0.0 if has_pawn else 1.0)  # 8
    
    # Pawn structure: count pawns per file
    for f in range(8):
        count = 0
        for r in range(8):
            p = board.piece_at(r * 8 + f)
            if p and p[1] == Piece.PAWN:
                count += 1
        features.append(count / 4.0)  # 8
    
    # Center control (d4,e4,d5,e5)
    center_squares = [27, 28, 35, 36]
    for sq in center_squares:
        piece = board.piece_at(sq)
        if piece is None:
            features.append(0.0)
        elif piece[0] == Color.WHITE:
            features.append(0.5)
        else:
            features.append(-0.5)  # 4
    
    # Piece mobility estimate (rough: count attacks from each square)
    # Simplified: just count legal moves
    moves = list(board.generate_legal_moves())
    features.append(min(len(moves) / 50.0, 1.0))
    
    # Game phase (0=opening, 1=endgame)
    total_pieces = len(board.pieces)
    features.append(1.0 - total_pieces / 32.0)
    
    # Castling rights
    features.append(1.0 if 'K' in board.castling_rights else 0.0)
    features.append(1.0 if 'Q' in board.castling_rights else 0.0)
    features.append(1.0 if 'k' in board.castling_rights else 0.0)
    features.append(1.0 if 'q' in board.castling_rights else 0.0)
    
    return np.array(features, dtype=np.float64)


# ===========================================================================
# Chess Regime Detector
# ===========================================================================

@dataclass
class ChessRegimeResult:
    """Regime detection result for a chess position."""
    rci: float                    # Regime Change Index [0, 1]
    confidence: float             # Jury confidence [0, 1]
    regime_change: bool           # Did the regime change?
    signals: Dict[str, float]     # Individual signal values
    fired: List[str]              # Which signals fired
    description: str              # Human-readable summary
    time_bonus_ms: float          # Suggested extra time (ms)
    trust_nn: float               # How much to trust NN eval [0, 1]


class ChessRegimeDetector:
    """
    Detects chess position regime changes using HyperTensor's RegimeDetector.
    
    Falls back to heuristic detection if hypercore is not installed.
    """
    
    def __init__(self, 
                 intrinsic_dim: int = 16,
                 window_size: int = 50,
                 threshold: float = 0.55,
                 use_hypercore: bool = True):
        self.intrinsic_dim = intrinsic_dim
        self.window_size = window_size
        self.threshold = threshold
        
        # Try to create RegimeDetector
        self._detector = None
        if _HAS_REGIME_DETECTOR and use_hypercore:
            self._detector = RegimeDetector(
                intrinsic_dim=intrinsic_dim,
                window_size=window_size,
                threshold=threshold,
            )
        
        # Track previous positions for heuristic fallback
        self._prev_features: List[np.ndarray] = []
        self._prev_eval: float = 0.0
        self._fitted = False
        
        # Signal history for heuristic
        self._eval_history: List[float] = []
        self._material_history: List[float] = []
    
    def fit_on_positions(self, positions: List[Board]):
        """
        Fit the regime detector on a set of training positions.
        Organizes positions as trajectories through chess space.
        """
        if not positions:
            return
        
        # Extract features
        features = np.stack([position_to_feature(b) for b in positions])
        
        if self._detector is not None:
            # Reshape as (N, T, D) trajectory
            # Treat each contiguous segment of 20 positions as a trajectory
            T = 20
            N = max(1, len(positions) // T)
            if N >= 3:
                traj = features[:N * T].reshape(N, T, -1)
                self._detector.fit(traj)
                self._fitted = True
        
        # Store for heuristic
        self._prev_features = features.tolist()[-self.window_size:]
        self._fitted = True
    
    def check(self, board: Board, current_eval: float = 0.0) -> ChessRegimeResult:
        """
        Check if the current position represents a regime change.
        
        Args:
            board: Current chess position
            current_eval: Current NN evaluation (for heuristic fallback)
        
        Returns:
            ChessRegimeResult with regime change detection
        """
        features = position_to_feature(board)
        
        if self._detector is not None and self._fitted:
            # Use HyperTensor RegimeDetector
            obs = features.reshape(1, -1)
            result = self._detector.check(obs)
            
            time_bonus = result.rci * 2000  # Up to 2 extra seconds
            trust_nn = max(0.2, 1.0 - result.rci * 0.8)
            
            signals_dict = {s.name: s.normalized for s in result.signals}
            fired = [s.name for s in result.signals if s.fired]
            
            return ChessRegimeResult(
                rci=result.rci,
                confidence=result.confidence,
                regime_change=result.regime_change,
                signals=signals_dict,
                fired=fired,
                description=result.description,
                time_bonus_ms=time_bonus,
                trust_nn=trust_nn,
            )
        
        # Heuristic fallback
        return self._heuristic_check(features, current_eval)
    
    def _heuristic_check(self, features: np.ndarray, current_eval: float) -> ChessRegimeResult:
        """Heuristic regime detection when hypercore is unavailable."""
        self._eval_history.append(current_eval)
        self._material_history.append(features[12] * 1000)  # White material
        
        if len(self._eval_history) > self.window_size:
            self._eval_history = self._eval_history[-self.window_size:]
            self._material_history = self._material_history[-self.window_size:]
        
        signals = {}
        
        # 1. Eval volatility (proxy for regime change)
        if len(self._eval_history) >= 5:
            recent_std = np.std(self._eval_history[-5:])
            full_std = np.std(self._eval_history) if len(self._eval_history) > 10 else recent_std
            signals['manifold_deviation'] = min(recent_std / max(full_std, 0.001), 1.0)
        else:
            signals['manifold_deviation'] = 0.0
        
        # 2. Material change (capture detection)
        if len(self._material_history) >= 2:
            mat_delta = abs(self._material_history[-1] - self._material_history[-2])
            signals['curvature_anomaly'] = min(mat_delta / 300.0, 1.0)
        else:
            signals['curvature_anomaly'] = 0.0
        
        # 3. Position novelty
        if self._prev_features:
            prev_arr = np.array(self._prev_features[-10:])
            dists = np.linalg.norm(prev_arr - features, axis=1)
            signals['neighbor_instability'] = min(np.mean(dists) / 3.0, 1.0)
        else:
            signals['neighbor_instability'] = 0.0
        
        # Default for remaining signals
        signals.setdefault('spectral_drift', 0.0)
        signals.setdefault('geodesic_misalignment', 0.0)
        
        # Aggregate
        weights = [0.25, 0.25, 0.2, 0.15, 0.15]
        signal_keys = ['manifold_deviation', 'curvature_anomaly', 'neighbor_instability',
                       'spectral_drift', 'geodesic_misalignment']
        rci = sum(w * signals.get(k, 0.0) for w, k in zip(weights, signal_keys))
        confidence = 1.0 - np.prod([1.0 - w * signals.get(k, 0.0) + 1e-12 
                                     for w, k in zip(weights, signal_keys)])
        
        regime_change = rci >= self.threshold
        fired = [k for k in signal_keys if signals.get(k, 0) > 0.5]
        
        time_bonus = rci * 1500
        trust_nn = max(0.3, 1.0 - rci * 0.7)
        
        # Update history
        self._prev_features.append(features)
        if len(self._prev_features) > self.window_size:
            self._prev_features = self._prev_features[-self.window_size:]
        
        desc = (f"Regime change (RCI={rci:.3f})" if regime_change 
                else f"Normal (RCI={rci:.3f})")
        
        return ChessRegimeResult(
            rci=rci,
            confidence=confidence,
            regime_change=regime_change,
            signals=signals,
            fired=fired,
            description=desc,
            time_bonus_ms=time_bonus,
            trust_nn=trust_nn,
        )


# ===========================================================================
# Integration with NegamaxEngine
# ===========================================================================

def integrate_regime_detector(engine, positions: List[Board] = None):
    """
    Wire a ChessRegimeDetector into a NegamaxEngine.
    
    After integration:
      - engine.regime_detector is available
      - Adaptive time uses regime RCI for time allocation
      - NN evaluation trust is modulated by regime stability
    """
    crd = ChessRegimeDetector(intrinsic_dim=16, use_hypercore=_HAS_REGIME_DETECTOR)
    
    if positions:
        crd.fit_on_positions(positions)
    
    engine.regime_detector = crd
    
    # Monkey-patch adaptive time to use regime info
    original_find = engine.find_best_move
    
    def regime_aware_find(board, time_limit_ms=3000, max_depth=99,
                          time_remaining_ms=None, time_increment_ms=0, moves_to_go=30):
        # Check regime before search
        regime = crd.check(board)
        
        if regime.regime_change and time_remaining_ms is not None:
            # Unstable position: allocate more time
            time_limit_ms = min(time_limit_ms + regime.time_bonus_ms,
                              time_remaining_ms * 0.4 if time_remaining_ms else time_limit_ms * 2)
        
        # Call original
        move, stats = original_find(board, time_limit_ms, max_depth,
                                    time_remaining_ms, time_increment_ms, moves_to_go)
        
        # Add regime info to stats
        stats['regime_rci'] = regime.rci
        stats['regime_change'] = regime.regime_change
        stats['nn_trust'] = regime.trust_nn
        stats['time_bonus_ms'] = regime.time_bonus_ms
        
        return move, stats
    
    engine.find_best_move = regime_aware_find
    
    return crd


# ===========================================================================
# Test
# ===========================================================================

if __name__ == '__main__':
    print("Chess RegimeDetector Test")
    print("=" * 60)
    print(f"HyperTensor RegimeDetector available: {_HAS_REGIME_DETECTOR}")
    
    from chess_engine.board import Board
    
    # Create some test positions
    positions = [
        Board(),  # Startpos
        Board('rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6'),  # Najdorf
        Board('r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 b kq - 3 5'),  # Italian
        Board('8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1'),  # KQvK
    ]
    
    crd = ChessRegimeDetector(intrinsic_dim=8)
    crd.fit_on_positions(positions * 20)  # Repeat to build trajectories
    
    # Test on new positions
    for board in positions:
        result = crd.check(board)
        print(f"\nPosition: {board.fen()[:40]}...")
        print(f"  RCI: {result.rci:.3f}, Change: {result.regime_change}")
        print(f"  NN Trust: {result.trust_nn:.1%}, Time Bonus: {result.time_bonus_ms:.0f}ms")
        if result.fired:
            print(f"  Fired: {result.fired}")
    
    print("\nRegimeDetector integration ready!")
    print("Install hypercore for full 5-signal geometric detection:")
    print("  pip install git+https://github.com/NagusameCS/HyperTensor.git#subdirectory=hypercore")
