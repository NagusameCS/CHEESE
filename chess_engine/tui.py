"""
HyperTensor Chess Engine — Terminal UI
========================================
Beautiful terminal chess interface with:
  - Unicode chess pieces + ANSI 24-bit colors
  - Live evaluation bar
  - Search statistics
  - Move history
  - Works on Windows/Linux/Mac

Usage:
  python -m chess_engine tui           # Play vs engine
  python -m chess_engine tui --watch   # Watch self-play
"""
import sys, os, time, threading, queue
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Enable Windows VT100/ANSI support
if sys.platform == 'win32':
    import ctypes
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except: pass

HT = Path(__file__).parent.parent / "HyperTensor"
if HT.exists(): sys.path.insert(0, str(HT))

from .board import Board, Move, Color, Piece, STARTING_FEN
from .evaluation import create_model, CUDA_AVAILABLE, DEVICE
from .search import HyperTensorSearch
from .opening_book import get_opening_move
from .pretrain import heuristic_evaluate

# ============================================================================
# ANSI helpers
# ============================================================================
def ansi(code): return f'\033[{code}m'
RST = ansi(0); BLD = ansi(1); DIM = ansi(2)
def rgb(r,g,b): return f'\033[38;2;{r};{g};{b}m'
def bgrgb(r,g,b): return f'\033[48;2;{r};{g};{b}m'

# Square colors
LIGHT = bgrgb(240,217,181); DARK = bgrgb(181,136,99)
LAST  = bgrgb(205,210,130);  SEL = bgrgb(200,200,100)
WHITE_PC = rgb(255,255,255) + BLD; BLACK_PC = rgb(30,30,30) + BLD
BORDER = rgb(120,100,70)

# Unicode pieces — CORRECT: white=white symbols, black=black symbols
PIECE_GLYPH = {
    (Color.WHITE, Piece.KING):   '\u2654',  # ♔
    (Color.WHITE, Piece.QUEEN):  '\u2655',  # ♕
    (Color.WHITE, Piece.ROOK):   '\u2656',  # ♖
    (Color.WHITE, Piece.BISHOP): '\u2657',  # ♗
    (Color.WHITE, Piece.KNIGHT): '\u2658',  # ♘
    (Color.WHITE, Piece.PAWN):   '\u2659',  # ♙
    (Color.BLACK, Piece.KING):   '\u265A',  # ♚
    (Color.BLACK, Piece.QUEEN):  '\u265B',  # ♛
    (Color.BLACK, Piece.ROOK):   '\u265C',  # ♜
    (Color.BLACK, Piece.BISHOP): '\u265D',  # ♝
    (Color.BLACK, Piece.KNIGHT): '\u265E',  # ♞
    (Color.BLACK, Piece.PAWN):   '\u265F',  # ♟
}

# ============================================================================
# Rendering functions
# ============================================================================
def render_board(board: Board, last_move=None, selected=None) -> str:
    lines = []
    lines.append(f"{BORDER}  +---+---+---+---+---+---+---+---+{RST}")
    for rank in range(7, -1, -1):
        row = f"{BORDER}{rank+1} |{RST}"
        for file in range(8):
            sq = rank * 8 + file
            is_light = (rank + file) % 2 == 0
            if sq == selected:   bg = SEL
            elif last_move and (sq == last_move.from_sq or sq == last_move.to_sq): bg = LAST
            elif is_light: bg = LIGHT
            else: bg = DARK
            piece = board.piece_at(sq)
            if piece:
                color = WHITE_PC if piece[0] == Color.WHITE else BLACK_PC
                glyph = PIECE_GLYPH.get(piece, '?')
                row += f"{bg}{color} {glyph} {RST}"
            else:
                row += f"{bg}   {RST}"
            row += f"{BORDER}|{RST}"
        lines.append(row)
        if rank > 0:
            lines.append(f"{BORDER}  +---+---+---+---+---+---+---+---+{RST}")
    lines.append(f"{BORDER}  +---+---+---+---+---+---+---+---+{RST}")
    lines.append(f"{BORDER}    a   b   c   d   e   f   g   h  {RST}")
    return '\n'.join(lines)

def render_eval_bar(score_cp, width=20):
    score = max(-1000, min(1000, score_cp))
    frac = (score + 1000) / 2000
    w = int(frac * width); b = width - w
    bar = f"{bgrgb(240,240,240)}{rgb(0,0,0)}{' ' * w}{bgrgb(40,40,40)}{rgb(255,255,255)}{' ' * b}{RST}"
    return f"  {bar} {score:+.0f} cp"

def render_search_info(stats: Dict) -> str:
    items = [
        ('Simulations', stats.get('simulations', 0)),
        ('Nodes/sec', stats.get('nps', 0)),
        ('Time', f"{stats.get('time_ms', 0):.0f} ms"),
        ('Best value', f"{stats.get('best_value', 0):+.3f}"),
        ('Jury accepts', stats.get('jury_accepts', 0)),
        ('Cache hits', stats.get('cache_hits', 0)),
        ('TT hits', stats.get('tt_hits', 0)),
        ('Unsafe', stats.get('unsafe_count', 0)),
    ]
    w = max(len(l) for l,_ in items)
    lines = [f'{DIM}  {"─"*40}{RST}']
    for label, val in items:
        lines.append(f'  {DIM}{label:<{w}}{RST} {BLD}{str(val):>10}{RST}')
    lines.append(f'{DIM}  {"─"*40}{RST}')
    return '\n'.join(lines)

def render_move_list(moves: List[str]) -> str:
    if not moves: return f'{DIM}  No moves yet{RST}'
    lines = [f'{DIM}  {"─"*40}{RST}']
    for i in range(0, min(len(moves), 40), 4):
        parts = []
        for j in range(4):
            idx = i + j
            if idx < len(moves):
                num = idx // 2 + 1
                dot = '.' if idx % 2 == 0 else ' '
                parts.append(f'{num:3d}{dot} {moves[idx]:5s}')
        lines.append(f'  {DIM}{" ".join(parts)}{RST}')
    if len(moves) > 40:
        lines.append(f'  {DIM}... ({len(moves)} total moves){RST}')
    lines.append(f'{DIM}  {"─"*40}{RST}')
    return '\n'.join(lines)

# ============================================================================
# Main TUI
# ============================================================================
class ChessTUI:
    def __init__(self, model=None, player_color=Color.WHITE, time_ms=3000, watch=False):
        self.board = Board()
        self.player = player_color
        self.time_ms = time_ms
        self.watch = watch
        
        # Load model
        if model is None:
            model = create_model(k_manifold=32, hidden_dim=128, num_layers=3)
            best = Path('models/sf_autopilot_best.pt')
            if best.exists():
                import torch
                ck = torch.load(best, map_location='cpu')
                model.load_state_dict(ck['model_state_dict'])
        self.model = model.eval()
        
        self.search = HyperTensorSearch(model, num_simulations=400,
            use_opening_book=True, tt_size_mb=32)
        self.moves: List[str] = []
        self.last_move: Optional[Move] = None
        self.stats: Dict = {}
        self.msg = ''
        self.running = True
        self.thinking = False
        self._queue = queue.Queue()
        self._thread = None
    
    def _think(self):
        try:
            move, stats = self.search.search(self.board, time_limit_ms=self.time_ms)
            self._queue.put(('done', move, stats))
        except Exception as e:
            self._queue.put(('error', None, str(e)))
    
    def engine_move(self):
        if self.thinking: return
        self.thinking = True
        self._thread = threading.Thread(target=self._think, daemon=True)
        self._thread.start()
    
    def check_done(self):
        try:
            msg = self._queue.get_nowait()
            if msg[0] == 'done':
                self.thinking = False
                return msg[1], msg[2]
            elif msg[0] == 'error':
                self.thinking = False
                self.msg = f'Engine error: {msg[2]}'
        except queue.Empty: pass
        return None
    
    def make_move(self, m: Move):
        self.last_move = m; self.moves.append(m.uci()); self.board.make_move(m)
    
    def render(self) -> str:
        out = []
        out.append(f'{BLD}{rgb(100,180,255)}  HyperTensor Chess v3.3{RST}')
        out.append(f'{DIM}  {"="*50}{RST}')
        out.append('')
        out.append(render_board(self.board, self.last_move))
        out.append('')
        
        # Eval
        if self.stats:
            val = self.stats.get('best_value', 0) * 1000
        else:
            hv, _, _ = heuristic_evaluate(self.board)
            val = hv * 1000
        out.append(render_eval_bar(val))
        out.append('')
        
        # Turn indicator
        turn = 'White' if self.board.color_to_move == Color.WHITE else 'Black'
        if self.thinking:
            dots = '.' * (int(time.time() * 3) % 4 + 1)
            out.append(f'  {rgb(255,200,50)}Engine thinking{dots}{RST}')
        elif self.board.is_game_over():
            r = self.board.result()
            if r == '1-0': out.append(f'  {rgb(100,255,100)}{BLD}WHITE WINS!{RST}')
            elif r == '0-1': out.append(f'  {rgb(100,255,100)}{BLD}BLACK WINS!{RST}')
            else: out.append(f'  {rgb(255,255,100)}{BLD}DRAW!{RST}')
        else:
            out.append(f'  {BLD}{turn} to move{RST}')
        out.append('')
        
        # Search info
        if self.stats:
            out.append(render_search_info(self.stats))
            out.append('')
        
        # Move history
        out.append(render_move_list(self.moves))
        out.append('')
        
        # Message
        if self.msg:
            out.append(f'  {rgb(255,255,100)}{self.msg}{RST}')
            out.append('')
        
        # Help
        if not self.board.is_game_over():
            if self.thinking:
                out.append(f'{DIM}  (waiting for engine...){RST}')
            elif self.board.color_to_move == self.player or self.watch:
                out.append(f'{DIM}  [move UCI] [quit] [undo] [new] [board] [moves] [eval] [book] [help]{RST}')
            else:
                out.append(f'{DIM}  Press ENTER for engine move{RST}')
        else:
            out.append(f'{DIM}  Game over. [new] for new game, [quit] to exit{RST}')
        
        return '\n'.join(out)
    
    def command(self, cmd: str) -> bool:
        c = cmd.strip().lower()
        if c in ('q', 'quit', 'exit'): self.running = False; return False
        if c in ('u', 'undo') and len(self.moves) >= 2:
            try: self.board.unmake_move(); self.moves.pop(); self.msg='Undid move'; return True
            except: pass
        if c in ('n', 'new', 'reset'):
            self.board=Board(); self.moves=[]; self.stats={}; self.last_move=None; self.msg='New game'; return True
        if c in ('b', 'board'): self.msg=self.board.fen(); return True
        if c in ('f', 'fen'): self.msg=f'FEN: {self.board.fen()}'; return True
        if c in ('m', 'moves', 'legal'):
            lm=self.board.generate_legal_moves(); self.msg=f'Legal: {",".join(m.uci() for m in lm[:8])}'; return True
        if c in ('e', 'eval'):
            hv,hmat,hmob=heuristic_evaluate(self.board); self.msg=f'Heuristic: {hv*1000:+.0f} cp'; return True
        if c in ('book',):
            bm=get_opening_move(self.board); self.msg=f'Book: {bm.uci()}' if bm else 'No book move'; return True
        if c in ('h', 'help'):
            self.msg='Commands: uci-move quit undo new board moves eval book'; return True
        if not c or c == '':
            if self.board.color_to_move != self.player and not self.thinking:
                self.engine_move()
            return True
        # Try as UCI move
        try:
            m = Move.from_uci(c)
            if m in self.board.generate_legal_moves():
                self.make_move(m); self.msg = ''
                return True
            self.msg = f'Illegal: {c}'
        except: self.msg = f'Unknown: {c}'
        return True
    
    def run(self):
        print('\033[2J\033[H', end='')  # Clear screen
        
        while self.running:
            # Render
            print('\033[H', end='')  # Home cursor
            print(self.render())
            sys.stdout.flush()
            
            # Auto engine moves
            if self.watch and not self.thinking and not self.board.is_game_over():
                self.engine_move()
                time.sleep(0.3)
                continue
            
            # Check engine done
            if self.thinking:
                result = self.check_done()
                if result:
                    move, stats = result
                    self.stats = stats or {}
                    if move:
                        self.make_move(move)
                if self.thinking:
                    time.sleep(0.15)
                    continue
            
            # Auto-trigger engine if its turn
            if self.board.color_to_move != self.player and not self.thinking and not self.board.is_game_over() and not self.watch:
                self.engine_move()
                continue
            
            # Read input
            try:
                cmd = input().strip()
                if not self.command(cmd): break
            except (EOFError, KeyboardInterrupt): break
            except Exception as e:
                self.msg = str(e)
        
        print(f'\n{RST}Goodbye!\n')


def main():
    import argparse
    p = argparse.ArgumentParser(description='HyperTensor Chess TUI')
    p.add_argument('--watch', action='store_true')
    p.add_argument('--black', action='store_true')
    p.add_argument('--time', type=int, default=3000)
    p.add_argument('--model', type=str, default=None)
    args = p.parse_args()
    
    model = None
    if args.model:
        import torch
        model = create_model(k_manifold=32, hidden_dim=128, num_layers=3)
        ck = torch.load(args.model, map_location='cpu')
        model.load_state_dict(ck['model_state_dict'])
    
    tui = ChessTUI(
        model=model,
        player_color=Color.BLACK if args.black else Color.WHITE,
        time_ms=args.time,
        watch=args.watch,
    )
    tui.run()

if __name__ == '__main__':
    main()
