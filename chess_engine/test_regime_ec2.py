import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from regime_chess import ChessRegimeDetector
from chess_engine.board import Board

print("Testing RegimeDetector with hypercore...")
crd = ChessRegimeDetector(intrinsic_dim=8, use_hypercore=True)
b = Board()
crd.fit_on_positions([b]*60)
r = crd.check(b)
print(f"RCI={r.rci:.3f} hypercore_available={crd._detector is not None}")
print(f"Signals: {r.signals}")
print("OK")
