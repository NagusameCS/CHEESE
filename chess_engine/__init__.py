"""
HyperTensor Chess Engine v3.1
=============================
Elite chess engine with Stockfish-level search techniques,
CNN evaluation, Syzygy tablebases, and full CUDA.

v3.1 Additions:
  - SEE, null-move, LMR, futility, countermove heuristics
  - Syzygy endgame tablebase probing
  - Elite move ordering with all standard heuristics
  - LazySMP multi-threaded search
  - Intelligent time management
"""

from .board import (Board, Move, Color, Piece, STARTING_FEN,
                     SQUARE_NAMES, NAME_TO_SQUARE, to_pychess, from_pychess)
from .evaluation import (HyperTensorChessNet, ChessNativeLinear,
    KExpansionScheduler, RiemannianAdamW, create_model, create_optimizer,
    count_parameters, CUDA_AVAILABLE, DEVICE)
from .search import (HyperTensorSearch, IterativeDeepeningSearch,
    TranspositionTable, play_game)
from .strong_search import (EliteMoveOrdering, SyzygyProbe, TimeManager,
    CountermoveTable, LazySMP, see, see_capture)
from .uci import UCIEngine, main as uci_main
from .train import HyperTensorTrainer, quick_demo
from .data_pipeline import DataGenerator, DataAugmentor
from .opening_book import OpeningBook, get_opening_move

from .train_loop import train_loop, FastDataGenerator, PrioritizedBuffer
from .pretrain import pretrain_model, heuristic_evaluate
from .expert_iteration import expert_iteration, HeuristicMCTS, ExpertDataGenerator
# Lazy imports for heavy modules (avoid loading models at import time)
def _get_web_main():
    from .web_ui import main as web_main
    return web_main

def _get_tui_main():
    from .tui import main as tui_main
    return tui_main

__version__ = "3.3.0"
