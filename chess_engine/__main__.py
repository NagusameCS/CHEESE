"""
HyperTensor Chess Engine v3.0 — Main Entry
============================================
Usage:
  python -m chess_engine              # UCI mode
  python -m chess_engine demo         # Quick demo
  python -m chess_engine train        # Training pipeline
  python -m chess_engine play         # Interactive play
  python -m chess_engine selfplay     # Engine vs itself
  python -m chess_engine benchmark    # GPU benchmarks
"""
import sys, argparse, time, numpy as np
from pathlib import Path

HT = Path(__file__).parent.parent / "HyperTensor"
if HT.exists():
    sys.path.insert(0, str(HT)); sys.path.insert(0, str(HT/"scripts"))

def main():
    p = argparse.ArgumentParser(description="HyperTensor Chess Engine v3.0")
    p.add_argument('mode', nargs='?', default='uci',
                   choices=['uci','demo','train','trainloop','pretrain','expert','tui','web','play','selfplay','benchmark'])
    p.add_argument('--model', type=str, default=None)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--k-start', type=int, default=4)
    p.add_argument('--k-target', type=int, default=64)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--games-per-iter', type=int, default=20)
    p.add_argument('--time', type=int, default=3000)
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--no-opening', action='store_true')
    args = p.parse_args()
    {'uci':uci,'demo':demo,'train':train,'trainloop':trainloop,'pretrain':pretrain_cmd,'expert':expert_cmd,'tui':tui_cmd,'web':web_cmd,'play':play,
     'selfplay':selfplay,'benchmark':benchmark}[args.mode](args)

def uci(args):
    from .uci import UCIEngine; from .evaluation import create_model
    m=create_model(use_jit=True,use_fp16=args.fp16)
    if args.model:
        import torch; ck=torch.load(args.model,map_location='cpu')
        m.load_state_dict(ck['model_state_dict'])
    e=UCIEngine(model=m); e.uci_loop()

def demo(args):
    from .train import quick_demo; quick_demo()

def train(args):
    from .train import HyperTensorTrainer
    t=HyperTensorTrainer(k_start=args.k_start,k_target=args.k_target,
        total_iterations=args.iterations,games_per_iter=args.games_per_iter,
        device=args.device,use_amp=args.fp16)
    if args.model: t.load_checkpoint(args.model)
    t.train()

def play(args):
    from .board import Board, Color, Move
    from .evaluation import create_model
    from .search import HyperTensorSearch
    m=create_model(use_jit=True,use_fp16=args.fp16)
    if args.model:
        import torch; ck=torch.load(args.model,map_location='cpu')
        m.load_state_dict(ck['model_state_dict'])
    s=HyperTensorSearch(m,use_opening_book=not args.no_opening)
    m.eval(); board=Board(); player=Color.WHITE
    print("\n"+"="*50)
    print("HyperTensor Chess v3.0 — Interactive Play")
    print("="*50)
    print("You: White | UCI format | 'quit'/'moves'/'board'/'fen'")
    while not board.is_game_over():
        print(f"\n{board}")
        if board.color_to_move==player:
            leg=board.generate_legal_moves(); lu={m.uci():m for m in leg}
            while True:
                cmd=input("Move: ").strip()
                if cmd=='quit': return
                elif cmd=='moves': print(f"Legal: {', '.join(sorted(lu.keys()))}"); continue
                elif cmd=='board': print(board); continue
                elif cmd=='fen': print(board.fen()); continue
                elif cmd in lu: board.make_move(lu[cmd]); break
                else: print(f"Invalid. Try: {', '.join(sorted(lu.keys())[:8])}...")
        else:
            print("Thinking...")
            mv,st=s.search(board,time_limit_ms=args.time)
            if mv:
                info=f"Engine: {mv.uci()}"
                if st.get('book'): info+=" [book]"
                elif st.get('best_value'): info+=f" [v={st['best_value']:.2f}]"
                if st.get('tt_hits'): info+=f" tt:{st['tt_hits']}"
                print(info); board.make_move(mv)
    r=board.result()
    print(f"\n{'1-0 White wins!' if r=='1-0' else '0-1 Black wins!' if r=='0-1' else 'Draw!'}")

def selfplay(args):
    from .evaluation import create_model
    from .search import play_game
    m=create_model(use_jit=True,use_fp16=args.fp16)
    if args.model:
        import torch; ck=torch.load(args.model,map_location='cpu')
        m.load_state_dict(ck['model_state_dict'])
    m.eval()
    print("\nHyperTensor Chess v3.0 — Self-Play\n"+"="*50)
    r,moves=play_game(m,m,time_per_move_ms=args.time,max_moves=300,
                      use_opening=not args.no_opening)
    print(f"\nResult: {r} | Moves: {len(moves)}")
    for i in range(0,len(moves),8): print(' '.join(moves[i:i+8]))

def benchmark(args):
    import torch
    from .evaluation import create_model, CUDA_AVAILABLE, DEVICE, count_parameters
    print("="*60); print("HyperTensor Chess v3.0 — GPU Benchmark"); print("="*60)
    print(f"Device: {DEVICE} | CUDA: {CUDA_AVAILABLE}")
    if CUDA_AVAILABLE:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    for name,k,h,L in [("Small",8,64,2),("Medium",32,128,3),("Large",64,256,4)]:
        print(f"\n--- {name} (k={k},h={h},L={L}) ---")
        m=create_model(k_manifold=k,hidden_dim=h,num_layers=L,use_jit=False,use_fp16=args.fp16)
        t,tr=count_parameters(m); print(f"  Params: {tr:,}")
        for bs in [1,8,32,64,128,256]:
            batch=np.zeros((bs,160,8,8),dtype=np.float32)
            # warmup
            for _ in range(10): m.evaluate_batch(batch)
            times=[]
            for _ in range(30):
                r=m.evaluate_batch(batch); times.append(r['time_ms'])
            avg=np.mean(times); pps=bs/(avg/1000)
            print(f"  Batch {bs:3d}: {avg:7.2f}ms | {pps:10.0f} pos/s")
    if CUDA_AVAILABLE:
        print("\n--- CUDA Graph ---")
        m=create_model(k_manifold=64,hidden_dim=256,num_layers=4,use_jit=False)
        ok=m.capture_cuda_graph()
        if ok:
            b=np.zeros((160,8,8),dtype=np.float32)
            for _ in range(100): m.evaluate_graph(b)
            torch.cuda.synchronize(); t0=time.time()
            for _ in range(2000): m.evaluate_graph(b)
            torch.cuda.synchronize()
            print(f"  CUDA graph: {(time.time()-t0)*1000/2000*1000:.1f} µs/pos")
    print("\nBenchmark complete!")

def trainloop(args):
    """Continuous training loop — runs until interrupted."""
    import torch
    from .train_loop import train_loop
    from .evaluation import create_model, CUDA_AVAILABLE
    
    model = None
    if args.model:
        model = create_model(use_jit=False).to('cuda' if CUDA_AVAILABLE else 'cpu')
        ck = torch.load(args.model, map_location='cpu')
        model.load_state_dict(ck['model_state_dict'])
    
    train_loop(
        model=model,
        k_start=args.k_start,
        k_target=args.k_target,
        games_per_batch=args.games_per_iter,
        sims_per_move=50,
        checkpoint_interval=25,
        resume_path=args.model,
    )

def pretrain_cmd(args):
    """Heuristic pretraining — teaches basic chess before self-play."""
    import torch
    from .pretrain import pretrain_model
    from .evaluation import create_model, CUDA_AVAILABLE
    
    print("Creating model for pretraining...", flush=True)
    model = create_model(k_manifold=8, hidden_dim=128, num_layers=3, use_jit=False)
    
    print("Pretraining on 100K positions (~5 minutes)...", flush=True)
    model = pretrain_model(
        model, num_positions=100000, batch_size=512,
        epochs=10, lr=1e-3,
        device='cuda' if CUDA_AVAILABLE else 'cpu'
    )
    
    path = 'models/pretrained_base.pt'
    torch.save({'model_state_dict': model.state_dict()}, path)
    print(f"Saved to {path}", flush=True)
    print("Now run: python -m chess_engine trainloop --model models/pretrained_base.pt")

def expert_cmd(args):
    """Expert iteration — heuristic MCTS → NN training loop."""
    import torch
    from .expert_iteration import expert_iteration
    from .evaluation import create_model, CUDA_AVAILABLE
    
    model = None
    if args.model:
        model = create_model(use_jit=False).to('cuda' if CUDA_AVAILABLE else 'cpu')
        ck = torch.load(args.model, map_location='cpu')
        model.load_state_dict(ck['model_state_dict'])
        print(f"Loaded model from {args.model}")
    
    expert_iteration(
        model=model,
        num_iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        sims_per_move=400,
        epochs_per_iter=5,
        batch_size=256,
        k_start=args.k_start,
        k_target=args.k_target,
    )

def tui_cmd(args):
    """Fancy terminal UI for interactive play."""
    from .tui import ChessTUI
    from .evaluation import create_model, CUDA_AVAILABLE
    import torch
    
    model = None
    if args.model:
        model = create_model(use_jit=False).to('cuda' if CUDA_AVAILABLE else 'cpu')
        ck = torch.load(args.model, map_location='cpu')
        model.load_state_dict(ck['model_state_dict'])
    
    tui = ChessTUI(
        model=model,
        time_limit_ms=args.time,
    )
    tui.run()

def web_cmd(args):
    """Launch web UI server."""
    from .web_ui import app
    print('\n  HyperTensor Chess Web UI')
    print('  ' + '-'*40)
    print('  Open: http://127.0.0.1:8080')
    print('  ' + '-'*40 + '\n')
    app.run(host='127.0.0.1', port=8080, debug=False, threaded=True)

if __name__=='__main__': main()
