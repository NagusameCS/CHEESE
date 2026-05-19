"""
HyperTensor Chess Engine — UCI Protocol Interface
==================================================
Universal Chess Interface implementation for the HyperTensor chess engine.
Compatible with Arena, CuteChess, and other UCI-compatible GUIs.
"""

import sys
import threading
import time
from typing import Optional

from .board import Board, Move, STARTING_FEN
from .evaluation import HyperTensorChessNet, create_model
from .search import HyperTensorSearch, IterativeDeepeningSearch


class UCIEngine:
    """UCI protocol engine using HyperTensor evaluation."""
    
    def __init__(self, model: HyperTensorChessNet = None):
        self.board = Board()
        self.model = model
        self.search = None
        
        # Engine state
        self.debug_mode = False
        self.position_fen = STARTING_FEN
        self.position_moves = []
        
        # Search parameters
        self.time_limit_ms = 1000
        self.max_depth = 99
        self.use_jury = True
        self.use_gtc = True
        
        # Pondering
        self.ponder = False
        self.ponder_thread = None
        
        # Move overhead
        self.move_overhead_ms = 50
    
    def _ensure_model(self):
        """Lazy-load model if not provided."""
        if self.model is None:
            print("info string Loading HyperTensor chess model...")
            self.model = create_model(k_manifold=64, hidden_dim=256)
            self.model.eval()
            print("info string Model loaded (random weights — needs training)")
        
        if self.search is None:
            self.search = HyperTensorSearch(
                self.model,
                use_jury_gate=self.use_jury,
                use_gtc_cache=self.use_gtc,
            )
    
    def uci_loop(self):
        """Main UCI command loop."""
        print("HyperTensor Chess Engine v1.0")
        print("Powered by HyperTensor Geometric Framework (NagusameCS)")
        
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            if self.debug_mode:
                print(f"info string Debug: received '{line}'")
            
            try:
                self._handle_command(line)
            except Exception as e:
                print(f"info string Error: {e}")
                if self.debug_mode:
                    import traceback
                    traceback.print_exc()
    
    def _handle_command(self, line: str):
        """Parse and execute a UCI command."""
        parts = line.split()
        if not parts:
            return
        
        cmd = parts[0]
        
        if cmd == 'uci':
            self._cmd_uci()
        elif cmd == 'isready':
            self._cmd_isready()
        elif cmd == 'setoption':
            self._cmd_setoption(parts[1:])
        elif cmd == 'ucinewgame':
            self._cmd_ucinewgame()
        elif cmd == 'position':
            self._cmd_position(parts[1:])
        elif cmd == 'go':
            self._cmd_go(parts[1:])
        elif cmd == 'stop':
            self._cmd_stop()
        elif cmd == 'ponderhit':
            self._cmd_ponderhit()
        elif cmd == 'quit':
            self._cmd_quit()
        elif cmd == 'debug':
            self._cmd_debug(parts[1:])
        elif cmd == 'd':
            self._cmd_display()
        elif cmd == 'eval':
            self._cmd_eval()
        else:
            print(f"info string Unknown command: {cmd}")
    
    def _cmd_uci(self):
        """Identify engine and available options."""
        self._ensure_model()
        
        print("id name HyperTensor Chess 1.0")
        print("id author HyperTensor (NagusameCS) + Copilot")
        print()
        print("option name Debug Log type check default false")
        print("option name Use Jury Gate type check default true")
        print("option name Use GTC Cache type check default true")
        print("option name Jury Threshold type spin default 85 min 50 max 99")
        print("option name N Jurors type spin default 7 min 3 max 15")
        print("option name MCTS Simulations type spin default 800 min 100 max 5000")
        print("option name CPuct type string default 1.5")
        print("option name Move Overhead type spin default 50 min 0 max 500")
        print("uciok")
    
    def _cmd_isready(self):
        """Check if engine is ready."""
        self._ensure_model()
        print("readyok")
    
    def _cmd_setoption(self, args):
        """Set engine option."""
        if len(args) < 4:
            return
        
        # Parse "name X value Y" format
        name_parts = []
        val_parts = []
        current = name_parts
        for part in args:
            if part == 'value':
                current = val_parts
            else:
                current.append(part)
        
        name = ' '.join(name_parts).lower().replace('name ', '')
        value = ' '.join(val_parts)
        
        if name == 'debug log':
            self.debug_mode = value.lower() == 'true'
        elif name == 'use jury gate':
            self.use_jury = value.lower() == 'true'
            if self.search:
                self.search.use_jury_gate = self.use_jury
        elif name == 'use gtc cache':
            self.use_gtc = value.lower() == 'true'
            if self.search:
                self.search.use_gtc_cache = self.use_gtc
        elif name == 'jury threshold':
            threshold = int(value) / 100.0
            if self.search:
                self.search.jury_threshold = threshold
        elif name == 'n jurors':
            if self.search:
                self.search.n_jurors = int(value)
        elif name == 'mcts simulations':
            if self.search:
                self.search.num_simulations = int(value)
        elif name == 'cpuct':
            if self.search:
                self.search.c_puct = float(value)
        elif name == 'move overhead':
            self.move_overhead_ms = int(value)
    
    def _cmd_ucinewgame(self):
        """Start a new game."""
        self.board = Board()
        self.position_moves = []
        if self.search:
            self.search.stats = {
                'jury_accepts': 0, 'jury_rejects': 0,
                'cache_hits': 0, 'cache_misses': 0, 'total_evals': 0,
            }
    
    def _cmd_position(self, args):
        """Set up position. Format: [fen <fen> | startpos] [moves <m1> ... <mn>]"""
        self.board = Board()
        self.position_moves = []
        
        idx = 0
        if args[idx] == 'startpos':
            self.board = Board(STARTING_FEN)
            idx += 1
        elif args[idx] == 'fen':
            idx += 1
            fen_parts = []
            while idx < len(args) and args[idx] != 'moves':
                fen_parts.append(args[idx])
                idx += 1
            fen = ' '.join(fen_parts)
            self.board = Board(fen)
        
        # Parse moves
        if idx < len(args) and args[idx] == 'moves':
            idx += 1
            while idx < len(args):
                move = Move.from_uci(args[idx])
                self.board.make_move(move)
                self.position_moves.append(move)
                idx += 1
        
        self.position_fen = self.board.fen()
    
    def _cmd_go(self, args):
        """Start calculating. Parse time controls."""
        self._ensure_model()
        
        # Parse time control
        wtime = None
        btime = None
        winc = 0
        binc = 0
        movestogo = None
        depth = None
        movetime = None
        
        i = 0
        while i < len(args):
            if args[i] == 'wtime' and i + 1 < len(args):
                wtime = int(args[i + 1]); i += 2
            elif args[i] == 'btime' and i + 1 < len(args):
                btime = int(args[i + 1]); i += 2
            elif args[i] == 'winc' and i + 1 < len(args):
                winc = int(args[i + 1]); i += 2
            elif args[i] == 'binc' and i + 1 < len(args):
                binc = int(args[i + 1]); i += 2
            elif args[i] == 'movestogo' and i + 1 < len(args):
                movestogo = int(args[i + 1]); i += 2
            elif args[i] == 'depth' and i + 1 < len(args):
                depth = int(args[i + 1]); i += 2
            elif args[i] == 'movetime' and i + 1 < len(args):
                movetime = int(args[i + 1]); i += 2
            elif args[i] == 'infinite':
                movetime = 999999999
                i += 1
            elif args[i] == 'ponder':
                self.ponder = True
                i += 1
            else:
                i += 1
        
        # Calculate time limit
        if movetime is not None:
            self.time_limit_ms = movetime - self.move_overhead_ms
        elif wtime is not None and btime is not None:
            if self.board.color_to_move == 0:  # White
                time_left = wtime
                inc = winc
            else:
                time_left = btime
                inc = binc
            
            if movestogo is None:
                movestogo = 30  # Assume ~30 moves left
            
            # Allocate time: use ~5% of remaining time + increment
            self.time_limit_ms = max(100, 
                int(time_left / movestogo * 0.8 + inc * 0.5) - self.move_overhead_ms
            )
        elif depth is not None:
            self.max_depth = depth
            self.time_limit_ms = 60000  # 60s if only depth specified
        else:
            self.time_limit_ms = 3000  # Default 3 seconds
        
        # Cap at reasonable limits
        self.time_limit_ms = min(self.time_limit_ms, 300000)  # Max 5 minutes
        
        print(f"info string Searching for {self.time_limit_ms}ms...")
        
        # Run search
        if self.search:
            if self.ponder:
                # Run in ponder mode (background thread)
                def ponder_search():
                    self.search.search(self.board, time_limit_ms=self.time_limit_ms * 10)
                self.ponder_thread = threading.Thread(target=ponder_search, daemon=True)
                self.ponder_thread.start()
                return
            
            best_move, stats = self.search.search(
                self.board, time_limit_ms=self.time_limit_ms
            )
            
            if best_move:
                print(f"bestmove {best_move.uci()}")
                
                # Print search info
                if stats:
                    sims = stats.get('simulations', 0)
                    depth = stats.get('depth', 0)
                    value = stats.get('best_value', 0)
                    jury_acc = stats.get('jury_accepts', 0)
                    cache_hits = stats.get('cache_hits', 0)
                    time_ms = stats.get('time_ms', 0)
                    nps = int(sims / (time_ms / 1000)) if time_ms > 0 else 0
                    
                    print(f"info depth {depth} score cp {int(value * 100)} "
                          f"nodes {sims} nps {nps} time {int(time_ms)} "
                          f"jury_accepts {jury_acc} cache_hits {cache_hits}")
            else:
                # No move found (shouldn't happen)
                legal = self.board.generate_legal_moves()
                if legal:
                    print(f"bestmove {legal[0].uci()}")
    
    def _cmd_stop(self):
        """Stop current search."""
        # In a real implementation, we'd set a stop flag
        # For now, searches time out on their own
        pass
    
    def _cmd_ponderhit(self):
        """Ponder move was played — switch from ponder to normal search."""
        self.ponder = False
        if self.ponder_thread and self.ponder_thread.is_alive():
            # Cancel pondering and start normal search with reduced time
            pass
    
    def _cmd_quit(self):
        """Exit the engine."""
        sys.exit(0)
    
    def _cmd_debug(self, args):
        """Toggle debug mode."""
        if args and args[0] in ('on', 'true', '1'):
            self.debug_mode = True
        else:
            self.debug_mode = False
    
    def _cmd_display(self):
        """Display current board (non-UCI, for debugging)."""
        print(self.board)
    
    def _cmd_eval(self):
        """Show static evaluation of current position."""
        self._ensure_model()
        tensor = self.board.to_tensor()
        score, policy = self.model.evaluate_position(tensor)
        print(f"Evaluation: {score:.1f} cp")


# ===========================================================================
# Main entry point
# ===========================================================================

def main():
    """Run the UCI engine."""
    engine = UCIEngine()
    engine.uci_loop()


if __name__ == '__main__':
    main()
