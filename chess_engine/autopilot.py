"""
Autopilot: Continuous Stockfish Distillation
==============================================
Generates Stockfish-evaluated data, trains NN, validates against SF ground truth.
Runs until NN evaluation matches Stockfish accuracy.
"""
import torch, torch.nn.functional as F
import numpy as np, time, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from chess_engine.stockfish_train import StockfishEvaluator, generate_positions
from chess_engine.evaluation import create_model, create_optimizer, count_parameters, CUDA_AVAILABLE, DEVICE
from chess_engine.board import Board

print('='*60)
print('AUTOPILOT: Stockfish Distillation Loop')
print('Target: NN evaluation error < 50 cp vs Stockfish')
print('='*60)

model = create_model(k_manifold=32, hidden_dim=128, num_layers=3).to(DEVICE)
best_path = Path('models/sf_autopilot_best.pt')
if best_path.exists():
    ck = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ck['model_state_dict'])
    print(f'Loaded previous best model (val_loss={ck.get("val_loss", "?")})')
print(f'Model: {count_parameters(model)[1]:,} params')

# Fixed validation set with Stockfish ground truth (depth 12)
val_boards = [
    Board(),
    Board('r6k/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1'),
    Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R6K w kq - 0 1'),
    Board('rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 1'),
]
val_names = ['startpos', 'White up Q', 'Black up Q', 'Najdorf']

print('Computing Stockfish ground truth (depth 12)...')
sf_val = StockfishEvaluator(depth=12)
val_targets = []
for board in val_boards:
    r = sf_val.evaluate(board)
    val_targets.append(r['value'])
    print(f'  SF ground truth: {r["score_cp"]:+d} cp')
sf_val.close()

val_tensors = torch.stack([
    torch.from_numpy(b.to_tensor()).float() for b in val_boards
]).to(DEVICE)
val_targets_t = torch.tensor(val_targets, dtype=torch.float32, device=DEVICE).unsqueeze(1)

# Loop
iteration = 0
best_val_loss = float('inf')
target_cp_error = 50  # Stop when average error < 50 centipawns

print(f'\nStarting autopilot...')
print(f'Target: avg error < {target_cp_error} cp')
print(f'Generating 10K positions per iteration\n')

while True:
    iteration += 1
    print(f'\n=== Iteration {iteration} ===', flush=True)
    
    # Generate data
    print(f'Generating 5K positions with Stockfish depth 10...', flush=True)
    positions = generate_positions(5000)
    
    sf2 = StockfishEvaluator(depth=10)
    tensors_list, values_list, wdls_list = [], [], []
    t0 = time.time()
    
    for i, board in enumerate(positions):
        tensors_list.append(board.to_tensor().astype(np.float32))
        r = sf2.evaluate(board)
        values_list.append(r['value'])
        wdls_list.append(r['wdl'])
        if (i+1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f'  {i+1}/{len(positions)} ({(i+1)/elapsed:.0f} pos/s)', flush=True)
    sf2.close()
    
    batch_path = f'models/sf_batch_{iteration}.npz'
    np.savez_compressed(batch_path,
        tensors=np.stack(tensors_list),
        values=np.array(values_list, dtype=np.float32),
        wdls=np.array(wdls_list, dtype=np.float32))
    print(f'  Saved {batch_path}', flush=True)
    
    # Train on all data
    all_files = list(Path('models').glob('sf_batch_*.npz'))
    if Path('models/sf_data.npz').exists():
        all_files.append(Path('models/sf_data.npz'))
    
    big_tensors, big_values = [], []
    for f in all_files:
        d = np.load(f)
        big_tensors.append(d['tensors'])
        big_values.append(d['values'])
    
    X = np.concatenate(big_tensors) if len(big_tensors) > 1 else big_tensors[0]
    Y = np.concatenate(big_values) if len(big_values) > 1 else big_values[0]
    print(f'  Training on {len(X)} positions...', flush=True)
    
    model.train()
    optimizer = create_optimizer(model, lr=5e-4)
    n = len(X); bs = 256
    
    for epoch in range(10):
        idx = np.random.permutation(n)
        losses = []
        for s in range(0, n, bs):
            i = idx[s:s+bs]
            x = torch.from_numpy(X[i]).float().to(DEVICE)
            y = torch.from_numpy(Y[i]).float().to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            loss = F.mse_loss(model(x)[0], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(loss.item())
    
    # Validate
    model.eval()
    with torch.no_grad():
        vp = model(val_tensors)[0]
        val_loss = F.mse_loss(vp, val_targets_t).item()
    
    # Compute centipawn errors
    errors = []
    for i, name in enumerate(val_names):
        pc = torch.tanh(vp[i]*3).item()*1000
        tc = val_targets[i]*1000
        err = abs(pc - tc)
        errors.append(err)
        print(f'    {name:15s}: pred={pc:+6.0f} true={tc:+6.0f} err={err:5.0f} cp', flush=True)
    
    avg_err = np.mean(errors)
    print(f'  Val loss: {val_loss:.4f} | Avg cp error: {avg_err:.0f} (target <{target_cp_error})', flush=True)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({'model_state_dict': model.state_dict(),
                    'iteration': iteration, 'val_loss': val_loss}, best_path)
        print(f'  [NEW BEST] Saved to {best_path}', flush=True)
    
    if avg_err < target_cp_error:
        print(f'\n*** CONVERGED! Avg error {avg_err:.0f} cp < {target_cp_error} cp ***', flush=True)
        print(f'Model saved to {best_path}', flush=True)
        break
    
    print(f'  Best so far: val_loss={best_val_loss:.4f}', flush=True)
