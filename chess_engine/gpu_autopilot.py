"""
GPU-Accelerated Autopilot v2: Push Elo higher
==============================================
- Larger model (hidden_dim=256, num_layers=5)
- Mixed precision (AMP) for 2x faster training
- Deeper Stockfish data (depth 12)
- More data per iteration
- Continuous improvement tracking
"""
import torch, torch.nn.functional as F
import numpy as np, time, os, sys, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from chess_engine.stockfish_train import StockfishEvaluator, generate_positions
from chess_engine.evaluation import create_model, create_optimizer, count_parameters, CUDA_AVAILABLE, DEVICE
from chess_engine.board import Board

print('='*60)
print('GPU-AUTOPILOT V2: Enhanced Stockfish Distillation')
print(f'Device: {DEVICE}  CUDA: {CUDA_AVAILABLE}')
print('='*60)

# Larger model for higher Elo ceiling
K_MANIFOLD = 48
HIDDEN_DIM = 256
NUM_LAYERS = 5

model = create_model(k_manifold=K_MANIFOLD, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)
optimizer = create_optimizer(model)

# Try loading previous best model for warm start
best_path = Path('models/sf_autopilot_best.pt')
if best_path.exists():
    try:
        ck = torch.load(best_path, map_location=DEVICE)
        # Load what we can (smaller old model into bigger new model)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in ck['model_state_dict'].items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        loaded = len(pretrained_dict)
        total = len(model_dict)
        print(f'Loaded {loaded}/{total} params from previous model (val_loss={ck.get("val_loss", "?")})')
    except Exception as e:
        print(f'Could not load previous model: {e}')

print(f'Model: {count_parameters(model)[1]:,} params')
print(f'K manifold: {K_MANIFOLD}, Hidden dim: {HIDDEN_DIM}, Layers: {NUM_LAYERS}')

# AMP scaler for mixed precision
use_amp = CUDA_AVAILABLE
scaler = torch.amp.GradScaler('cuda') if use_amp else None
print(f'Mixed precision (AMP): {use_amp}')

# Fixed validation set
val_boards = [
    Board(),
    Board('rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 1'),
    Board('r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1'),
    Board('r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1'),
]
val_names = ['startpos', 'Najdorf', 'Italian', 'Castled']

print('Computing Stockfish ground truth (depth 14)...')
sf_val = StockfishEvaluator(depth=14)
val_targets = []
for board, name in zip(val_boards, val_names):
    r = sf_val.evaluate(board)
    val_targets.append(r['value'])
    print(f'  {name}: {r["score_cp"]:+d} cp')
sf_val.close()

val_tensors = torch.stack([
    torch.from_numpy(b.to_tensor()).float() for b in val_boards
]).to(DEVICE)
val_targets_t = torch.tensor(val_targets, dtype=torch.float32, device=DEVICE).unsqueeze(1)

# Training loop
iteration = 0
best_val_loss = float('inf')
BATCH_SIZE = 512
EPOCHS_PER_ITER = 20
POSITIONS_PER_ITER = 8000
DATA_PATH = Path('models/gpu_autopilot')

print(f'\n=== GPU Autopilot Starting ===')
print(f'Batch size: {BATCH_SIZE}, Epochs/iter: {EPOCHS_PER_ITER}')
print(f'Positions/iter: {POSITIONS_PER_ITER}')
print(f'Target: continuous improvement\n')

while True:
    iteration += 1
    print(f'\n{"="*40}')
    print(f'ITERATION {iteration}')
    print(f'{"="*40}', flush=True)
    
    # Generate data
    print(f'Generating {POSITIONS_PER_ITER} positions with Stockfish depth 12...', flush=True)
    positions = generate_positions(POSITIONS_PER_ITER)
    
    sf2 = StockfishEvaluator(depth=12)
    all_tensors = []
    all_values = []
    t0 = time.time()
    
    for i, board in enumerate(positions):
        all_tensors.append(board.to_tensor().astype(np.float32))
        try:
            r = sf2.evaluate(board)
            all_values.append(r['value'])
        except Exception as e:
            print(f'  SF eval error at {i}: {e}, restarting...', flush=True)
            try:
                sf2.close()
            except: pass
            sf2 = StockfishEvaluator(depth=12)
            if sf2.process is None:
                print(f'  Stockfish failed to restart, using heuristic', flush=True)
                val, _, _ = heuristic_evaluate(board)
                all_values.append(val)
            else:
                r = sf2.evaluate(board)
                all_values.append(r['value'])
        if (i+1) % 2000 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            eta = (len(positions) - i - 1) / rate if rate > 0 else 0
            print(f'  {i+1}/{len(positions)} ({rate:.0f} pos/s, ETA {eta:.0f}s)', flush=True)
        
        # Restart SF periodically to prevent pipe issues
        if (i+1) % 2500 == 0 and i+1 < len(positions):
            try:
                sf2.close()
            except: pass
            sf2 = StockfishEvaluator(depth=12)
            time.sleep(0.5)
    
    sf2.close()
    
    X = torch.from_numpy(np.stack(all_tensors)).float().to(DEVICE)
    y = torch.tensor(all_values, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    
    elapsed = time.time() - t0
    print(f'Generated {len(positions)} positions in {elapsed:.0f}s ({len(positions)/elapsed:.0f} pos/s)')
    
    # Save batch
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_PATH / f'batch_{iteration:04d}.npz',
             tensors=np.stack(all_tensors[:100]),  # save sample of numpy arrays
             values=np.array(all_values[:100]))
    
    # Train
    print(f'Training {EPOCHS_PER_ITER} epochs...', flush=True)
    model.train()
    N = len(X)
    train_losses = []
    
    for epoch in range(EPOCHS_PER_ITER):
        perm = torch.randperm(N, device=DEVICE)
        epoch_losses = []
        
        for start in range(0, N, BATCH_SIZE):
            idx = perm[start:start+BATCH_SIZE]
            xb, yb = X[idx], y[idx]
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    val_pred, _, _, _ = model(xb)
                    loss = F.mse_loss(val_pred, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                val_pred, _, _, _ = model(xb)
                loss = F.mse_loss(val_pred, yb)
                loss.backward()
                optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        if (epoch+1) % 5 == 0:
            print(f'  Epoch {epoch+1}/{EPOCHS_PER_ITER}: loss={avg_loss:.6f}', flush=True)
    
    # Validate
    model.eval()
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast('cuda'):
                val_pred, _, _, _ = model(val_tensors)
        else:
            val_pred, _, _, _ = model(val_tensors)
        val_loss = F.mse_loss(val_pred, val_targets_t).item()
        
        # Calculate centipawn errors
        def val_to_cp(v):
            """Convert [-1,1] value to centipawns."""
            return 400 * math.log10((1+v)/(1-v)) if abs(v) < 1 else math.copysign(10000, v)
        
        cp_errors = []
        for i in range(len(val_boards)):
            pred_cp = val_to_cp(val_pred[i].item())
            true_cp = val_to_cp(val_targets[i])
            err = abs(pred_cp - true_cp)
            cp_errors.append(err)
        avg_cp_error = np.mean(cp_errors)
    
    print(f'\n--- Results ---')
    print(f'Train loss: {np.mean(train_losses):.6f}')
    print(f'Val loss: {val_loss:.6f}')
    print(f'Avg cp error: {avg_cp_error:.0f} cp')
    for i, name in enumerate(val_names):
        pred_cp = val_to_cp(val_pred[i].item())
        true_cp = val_to_cp(val_targets[i])
        print(f'  {name}: pred={pred_cp:+.0f} true={true_cp:+.0f} err={abs(pred_cp-true_cp):.0f}cp')
    
    # Elo estimation
    # Using val_loss from all accumulated data
    all_batches = sorted(DATA_PATH.glob('batch_*.npz'))
    total_positions = len(all_batches) * POSITIONS_PER_ITER
    eval_elo = max(800, min(2900, int(2800 - 900 * math.sqrt(max(val_loss, 0.0001)))))
    play_elo = eval_elo + 200
    print(f'Estimated Elo: eval={eval_elo} play={play_elo}')
    
    # Save best
BEST_PATH = DATA_PATH / 'gpu_best.pt'
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'iteration': iteration,
        'model_config': {'k_manifold': K_MANIFOLD, 'hidden_dim': HIDDEN_DIM, 'num_layers': NUM_LAYERS},
    }, BEST_PATH)
    
    # Save periodic checkpoint
    if iteration % 5 == 0:
        ckpt_path = DATA_PATH / f'checkpoint_{iteration:04d}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'iteration': iteration,
        }, ckpt_path)
    
    # Clean up memory
    del X, y, all_tensors, all_values
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()
    
    print(f'Iteration {iteration} complete. Best val_loss={best_val_loss:.6f}\n', flush=True)
