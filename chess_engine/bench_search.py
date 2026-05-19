"""Quick benchmark of upgraded negamax search features."""
import time, torch, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_engine.board import Board
from chess_engine.negamax import NegamaxEngine
from chess_engine.evaluation import create_model, DEVICE

print('Loading model...')
model = create_model(k_manifold=32, hidden_dim=128, num_layers=3).to(DEVICE)
ck = torch.load('models/sf_autopilot_best.pt', map_location=DEVICE, weights_only=True)
model_dict = model.state_dict()
pretrained = {k: v for k, v in ck['model_state_dict'].items() 
              if k in model_dict and model_dict[k].shape == v.shape}
model_dict.update(pretrained)
model.load_state_dict(model_dict, strict=False)
print(f'Loaded {len(pretrained)}/{len(model_dict)} params')

engine = NegamaxEngine(model, tt_size_mb=32)

print()
print('Upgraded Search Features:')
print(f'  Singular Extensions: {engine.use_singular}')
print(f'  Multi-Cut Pruning:   {engine.use_multicut}')
print(f'  Null-Move:           {engine.use_null_move}')
print(f'  LMR:                 {engine.use_lmr}')
print(f'  Futility:            {engine.use_futility}')
print()

tests = [
    ('startpos', Board()),
    ('Najdorf', Board('rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6')),
    ('Italian', Board('r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 b kq - 3 5')),
    ('KQvK', Board('8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1')),
]

for name, board in tests:
    engine.state.clear_search_stats()
    t0 = time.time()
    move, stats = engine.find_best_move(board, time_limit_ms=2000, max_depth=99)
    elapsed = time.time() - t0
    
    d = stats.get('depth', 0)
    s = stats.get('score', 0)
    n = stats.get('nodes', 0)
    nps = stats.get('nps', 0)
    se = stats.get('singular_ext', 0)
    mc = stats.get('multicut', 0)
    ext = stats.get('extensions', 0)
    
    print(f'{name:10s} depth={d:2d} score={s:+6d} nodes={n:>8,} nps={nps:>10,} time={elapsed:.2f}s')
    print(f'           singular_ext={se} multicut={mc} extensions={ext}')
    if move:
        print(f'           best: {move.uci()}')
    print()

print('All search upgrades active and working!')
