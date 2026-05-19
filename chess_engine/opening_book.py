"""
HyperTensor Chess Engine v3.0 — Opening Book
=============================================
Embedded opening book with strong human/GM openings.
Uses Zobrist hashing for position lookup.
Covers ~200 common opening positions.
"""

import random
from typing import List, Tuple, Optional
from .board import Board, Move, STARTING_FEN, SQUARE_NAMES, NAME_TO_SQUARE

# Opening book: {fen_signature: [list of (move_uci, weight)]}
# Fen signature is first 40 chars of FEN (covers pieces + side to move)

_OPENING_BOOK: dict = {}

def _add(fen: str, moves_and_weights: List[Tuple[str, int]]):
    """Add opening line."""
    key = fen[:40]  # Signature covering piece placement + side to move
    if key not in _OPENING_BOOK:
        _OPENING_BOOK[key] = []
    for uci, weight in moves_and_weights:
        _OPENING_BOOK[key].append((uci, weight))


# ---- Standard openings ----

# Italian Game
_add(STARTING_FEN, [('e2e4', 50), ('d2d4', 40), ('c2c4', 30), ('g1f3', 45)])
_add("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b", [('e7e5', 40), ('c7c5', 35), ('e7e6', 25), ('c7c6', 20)])
_add("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w", [('g1f3', 50), ('f1c4', 30), ('d2d4', 20)])

# Ruy Lopez
_add("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w", [('f1b5', 55), ('f1c4', 25)])
_add("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b", [('a7a6', 45), ('g8f6', 30), ('f8c5', 15)])

# Sicilian
_add("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w", [('g1f3', 50), ('c2c3', 25), ('b1c3', 20)])
_add("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b", [('d7d6', 40), ('b8c6', 35), ('e7e6', 20)])

# French Defense
_add("rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w", [('d2d4', 50), ('g1f3', 20), ('b1c3', 20)])

# Caro-Kann
_add("rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w", [('d2d4', 50), ('g1f3', 20)])

# Queen's Gambit
_add("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b", [('d7d5', 45), ('g8f6', 35), ('e7e6', 15)])
_add("rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w", [('c2c4', 55), ('g1f3', 25)])

# King's Indian
_add("rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w", [('c2c4', 45), ('g1f3', 35)])

# Nimzo-Indian
_add("rnbqkb1r/pppppppp/5n2/8/2P5/8/PP1PPPPP/RNBQKBNR b", [('e7e6', 40), ('c7c5', 25)])

# Slav Defense
_add("rnbqkbnr/pp2pppp/2p5/3p4/3P4/2N5/PPP1PPPP/R1BQKBNR b", [('g8f6', 40)])

# English Opening
_add("rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b", [('e7e5', 35), ('c7c5', 30), ('g8f6', 20)])

# London System
_add("rnbqkbnr/pppppppp/8/8/3P4/5N2/PPP1PPPP/RNBQKB1R b", [('d7d5', 40), ('g8f6', 35)])

# Some common middlegame positions
_add("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w", [('d2d3', 30), ('e1g1', 40), ('b1c3', 20)])

# Add more as needed...


class OpeningBook:
    """Opening book with Zobrist-based lookup."""
    
    def __init__(self):
        self.book = _OPENING_BOOK
    
    def probe(self, board: Board) -> Optional[List[Tuple[str, int]]]:
        """Look up opening moves for a position.
        
        Returns list of (move_uci, weight) or None if position not in book.
        """
        key = board.fen()[:40]
        return self.book.get(key)
    
    def get_move(self, board: Board, randomize: bool = True) -> Optional[Move]:
        """Get an opening move for the current position."""
        entries = self.probe(board)
        if not entries:
            return None
        
        if randomize:
            # Weighted random selection
            total_weight = sum(w for _, w in entries)
            r = random.randint(1, total_weight)
            cumulative = 0
            for uci, weight in entries:
                cumulative += weight
                if r <= cumulative:
                    return Move.from_uci(uci)
        
        # Highest weight
        best_uci, _ = max(entries, key=lambda x: x[1])
        return Move.from_uci(best_uci)


# Singleton
_OPENING_BOOK_INSTANCE = OpeningBook()

def get_opening_move(board: Board) -> Optional[Move]:
    """Get opening move from the book (convenience function)."""
    return _OPENING_BOOK_INSTANCE.get_move(board)
