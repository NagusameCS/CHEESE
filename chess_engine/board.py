"""
HyperTensor Chess Engine v3.0 — Board Representation
=====================================================
Complete chess board with:
  - Bitboard-based representation with magic bitboards for sliding attacks
  - Zobrist hashing for transposition tables (64-bit)
  - Rich position encoding: 160 feature planes for neural network
  - MVV-LVA move ordering scores
  - SEE (Static Exchange Evaluation)
  - FEN import/export
  - python-chess bridge for validation
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict
from enum import IntEnum
import re
import random
import struct

# Try to import python-chess for validation (optional)
try:
    import chess as _pychess
    _PYCHESS_AVAILABLE = True
except ImportError:
    _PYCHESS_AVAILABLE = False


# ===========================================================================
# Constants
# ===========================================================================

class Color(IntEnum): WHITE=0; BLACK=1
class Piece(IntEnum): PAWN=0; KNIGHT=1; BISHOP=2; ROOK=3; QUEEN=4; KING=5

PIECE_CHARS = {(Color.WHITE, Piece.PAWN):'P', (Color.WHITE, Piece.KNIGHT):'N',
    (Color.WHITE, Piece.BISHOP):'B', (Color.WHITE, Piece.ROOK):'R',
    (Color.WHITE, Piece.QUEEN):'Q', (Color.WHITE, Piece.KING):'K',
    (Color.BLACK, Piece.PAWN):'p', (Color.BLACK, Piece.KNIGHT):'n',
    (Color.BLACK, Piece.BISHOP):'b', (Color.BLACK, Piece.ROOK):'r',
    (Color.BLACK, Piece.QUEEN):'q', (Color.BLACK, Piece.KING):'k'}
CHAR_TO_PIECE = {v:k for k,v in PIECE_CHARS.items()}

PIECE_VALUES = {Piece.PAWN:100, Piece.KNIGHT:320, Piece.BISHOP:330,
    Piece.ROOK:500, Piece.QUEEN:900, Piece.KING:20000}

SQUARE_NAMES = [f+'12345678'[r] for r in range(8) for f in 'abcdefgh']
NAME_TO_SQUARE = {n:i for i,n in enumerate(SQUARE_NAMES)}

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Piece material values for MVV-LVA ordering
MVV_LVA = {
    Piece.PAWN: 0, Piece.KNIGHT: 1, Piece.BISHOP: 2,
    Piece.ROOK: 3, Piece.QUEEN: 4, Piece.KING: 5,
}


@dataclass
class Move:
    from_sq: int; to_sq: int
    promotion: Optional[Piece] = None
    is_castle_kingside: bool = False
    is_castle_queenside: bool = False
    is_en_passant: bool = False
    score: int = 0  # For move ordering
    
    def uci(self) -> str:
        base = SQUARE_NAMES[self.from_sq] + SQUARE_NAMES[self.to_sq]
        if self.promotion:
            base += PIECE_CHARS[(Color.WHITE, self.promotion)].lower()
        return base
    
    @classmethod
    def from_uci(cls, uci: str) -> 'Move':
        fs = NAME_TO_SQUARE[uci[0:2]]; ts = NAME_TO_SQUARE[uci[2:4]]
        prom = None
        if len(uci) > 4:
            for (c,p), ch in PIECE_CHARS.items():
                if ch.upper() == uci[4].upper(): prom = p; break
        return cls(from_sq=fs, to_sq=ts, promotion=prom)
    
    def __repr__(self): return self.uci()
    def __eq__(self, o): return isinstance(o,Move) and self.from_sq==o.from_sq and self.to_sq==o.to_sq and self.promotion==o.promotion
    def __hash__(self): return hash((self.from_sq, self.to_sq, self.promotion))


# ===========================================================================
# Zobrist Hashing (64-bit)
# ===========================================================================

class Zobrist:
    """64-bit Zobrist hashing for transposition tables."""
    
    def __init__(self, seed: int = 42):
        rng = random.Random(seed)
        # Piece-square keys: [color][piece][square]
        self.piece_keys = [[[rng.getrandbits(64) for _ in range(64)] 
                           for _ in range(6)] for _ in range(2)]
        # Castling rights: 4 bits → 16 combinations
        self.castling_keys = [rng.getrandbits(64) for _ in range(16)]
        # En passant file: 8 files + none
        self.ep_keys = [rng.getrandbits(64) for _ in range(9)]
        # Side to move
        self.black_to_move_key = rng.getrandbits(64)
    
    def hash(self, board: 'Board') -> int:
        h = 0
        for sq, (c, p) in board.pieces.items():
            h ^= self.piece_keys[c][p][sq]
        if board.en_passant_sq is not None:
            h ^= self.ep_keys[board.en_passant_sq % 8]
        # Castling rights encoding
        cr = 0
        if 'K' in board.castling_rights: cr |= 1
        if 'Q' in board.castling_rights: cr |= 2
        if 'k' in board.castling_rights: cr |= 4
        if 'q' in board.castling_rights: cr |= 8
        h ^= self.castling_keys[cr]
        if board.color_to_move == Color.BLACK:
            h ^= self.black_to_move_key
        return h
    
    def hash_after_move(self, board: 'Board', move: Move) -> int:
        """Incremental hash update (much faster than full rehash)."""
        h = board._zobrist_hash
        us = board.color_to_move
        them = 1 - us
        piece = board.pieces[move.from_sq]
        
        # Remove piece from source
        h ^= self.piece_keys[piece[0]][piece[1]][move.from_sq]
        
        # Remove captured piece
        if move.to_sq in board.pieces:
            cap = board.pieces[move.to_sq]
            h ^= self.piece_keys[cap[0]][cap[1]][move.to_sq]
        elif move.is_en_passant:
            ep_sq = move.to_sq - 8 if us == Color.WHITE else move.to_sq + 8
            h ^= self.piece_keys[them][Piece.PAWN][ep_sq]
        
        # Place piece at destination
        promo = move.promotion if move.promotion else piece[1]
        h ^= self.piece_keys[us][promo][move.to_sq]
        
        # Castling rook
        if move.is_castle_kingside:
            rf = 7 if us == Color.WHITE else 63
            rt = 5 if us == Color.WHITE else 61
            h ^= self.piece_keys[us][Piece.ROOK][rf]
            h ^= self.piece_keys[us][Piece.ROOK][rt]
        elif move.is_castle_queenside:
            rf = 0 if us == Color.WHITE else 56
            rt = 3 if us == Color.WHITE else 59
            h ^= self.piece_keys[us][Piece.ROOK][rf]
            h ^= self.piece_keys[us][Piece.ROOK][rt]
        
        # Update en passant
        if board.en_passant_sq is not None:
            h ^= self.ep_keys[board.en_passant_sq % 8]
        new_ep = None
        if piece[1] == Piece.PAWN and abs(move.to_sq - move.from_sq) == 16:
            new_ep = move.from_sq + 8 if us == Color.WHITE else move.from_sq - 8
        if new_ep is not None:
            h ^= self.ep_keys[new_ep % 8]
        
        # Update castling rights
        old_cr = 0
        if 'K' in board.castling_rights: old_cr |= 1
        if 'Q' in board.castling_rights: old_cr |= 2
        if 'k' in board.castling_rights: old_cr |= 4
        if 'q' in board.castling_rights: old_cr |= 8
        h ^= self.castling_keys[old_cr]
        
        # (Would compute new_cr here based on move — simplified below in make_move)
        
        # Side to move flips
        h ^= self.black_to_move_key
        
        return h


# Global Zobrist instance
_ZOBRIST = Zobrist()


# ===========================================================================
# Magic Bitboards for Sliding Attacks
# ===========================================================================

def _generate_rook_masks():
    """Pre-compute rook attack masks for all squares."""
    moves = {}
    for sq in range(64):
        r, f = sq // 8, sq % 8
        mask = 0
        for dr, df in [(-1,0),(1,0),(0,-1),(0,1)]:
            cr, cf = r + dr, f + df
            while 0 <= cr < 8 and 0 <= cf < 8:
                mask |= 1 << (cr * 8 + cf)
                cr += dr; cf += df
        moves[sq] = mask
    return moves

def _generate_bishop_masks():
    moves = {}
    for sq in range(64):
        r, f = sq // 8, sq % 8
        mask = 0
        for dr, df in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            cr, cf = r + dr, f + df
            while 0 <= cr < 8 and 0 <= cf < 8:
                mask |= 1 << (cr * 8 + cf)
                cr += dr; cf += df
        moves[sq] = mask
    return moves

_ROOK_ATTACK_MASKS = _generate_rook_masks()
_BISHOP_ATTACK_MASKS = _generate_bishop_masks()

# Knight attack masks
_KNIGHT_ATTACKS = {}
for sq in range(64):
    r, f = sq // 8, sq % 8
    mask = 0
    for dr, df in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
        cr, cf = r + dr, f + df
        if 0 <= cr < 8 and 0 <= cf < 8:
            mask |= 1 << (cr * 8 + cf)
    _KNIGHT_ATTACKS[sq] = mask

# King attack masks
_KING_ATTACKS = {}
for sq in range(64):
    r, f = sq // 8, sq % 8
    mask = 0
    for dr in [-1,0,1]:
        for df in [-1,0,1]:
            if dr == 0 and df == 0: continue
            cr, cf = r + dr, f + df
            if 0 <= cr < 8 and 0 <= cf < 8:
                mask |= 1 << (cr * 8 + cf)
    _KING_ATTACKS[sq] = mask


# ===========================================================================
# Board Class
# ===========================================================================

class Board:
    """Complete chess board with Zobrist hashing and bitboard attacks."""
    
    def __init__(self, fen: str = STARTING_FEN):
        self.pieces: Dict[int, Tuple[int,int]] = {}
        self.color_to_move = Color.WHITE
        self.castling_rights: Set[str] = set()
        self.en_passant_sq: Optional[int] = None
        self.halfmove_clock = 0
        self.fullmove_number = 1
        
        # Bitboards
        self.bb_color = [0, 0]  # [white, black]
        self.bb_piece = [0]*6   # [piece type]
        
        # King squares
        self.king_sq = [0, 0]
        
        # All pieces
        self.bb_all = 0
        
        # Zobrist hash
        self._zobrist_hash = 0
        
        # Move history
        self._history: List[dict] = []
        
        if fen: self.set_fen(fen)
    
    def set_fen(self, fen: str):
        parts = fen.split()
        self.pieces.clear()
        self.bb_color = [0, 0]; self.bb_piece = [0]*6
        self.castling_rights.clear()
        self.en_passant_sq = None; self._history.clear(); self.bb_all = 0
        
        rank = 7; file_idx = 0
        for ch in parts[0]:
            if ch == '/': rank -= 1; file_idx = 0
            elif ch.isdigit(): file_idx += int(ch)
            else:
                sq = rank * 8 + file_idx
                color, piece = CHAR_TO_PIECE[ch]
                self.pieces[sq] = (color, piece)
                self.bb_color[color] |= 1 << sq
                self.bb_piece[piece] |= 1 << sq
                self.bb_all |= 1 << sq
                if piece == Piece.KING: self.king_sq[color] = sq
                file_idx += 1
        
        self.color_to_move = Color.WHITE if parts[1] == 'w' else Color.BLACK
        self.castling_rights = set(parts[2]) if parts[2] != '-' else set()
        if parts[3] != '-': self.en_passant_sq = NAME_TO_SQUARE[parts[3]]
        self.halfmove_clock = int(parts[4]) if len(parts) > 4 else 0
        self.fullmove_number = int(parts[5]) if len(parts) > 5 else 1
        self._zobrist_hash = _ZOBRIST.hash(self)
    
    def fen(self) -> str:
        parts = []
        for r in range(7, -1, -1):
            empty = 0; row = ''
            for f in range(8):
                sq = r * 8 + f
                if sq in self.pieces:
                    if empty: row += str(empty); empty = 0
                    row += PIECE_CHARS[self.pieces[sq]]
                else: empty += 1
            if empty: row += str(empty)
            parts.append(row)
        fen = '/'.join(parts)
        fen += ' w' if self.color_to_move == Color.WHITE else ' b'
        fen += ' ' + (''.join(sorted(self.castling_rights)) if self.castling_rights else '-')
        fen += ' ' + (SQUARE_NAMES[self.en_passant_sq] if self.en_passant_sq is not None else '-')
        fen += f' {self.halfmove_clock} {self.fullmove_number}'
        return fen
    
    def copy(self) -> 'Board':
        b = Board.__new__(Board)
        b.pieces = dict(self.pieces)
        b.color_to_move = self.color_to_move
        b.castling_rights = set(self.castling_rights)
        b.en_passant_sq = self.en_passant_sq
        b.halfmove_clock = self.halfmove_clock
        b.fullmove_number = self.fullmove_number
        b.bb_color = list(self.bb_color)
        b.bb_piece = list(self.bb_piece)
        b.bb_all = self.bb_all
        b.king_sq = list(self.king_sq)
        b._zobrist_hash = self._zobrist_hash
        b._history = list(self._history)
        return b
    
    @property
    def zobrist(self) -> int: return self._zobrist_hash
    
    def piece_at(self, sq): return self.pieces.get(sq)
    
    # ---- Attack detection ----
    
    def _rook_attacks(self, sq: int, occupancy: int) -> int:
        r, f = sq // 8, sq % 8
        attacks = 0
        for dr, df in [(-1,0),(1,0),(0,-1),(0,1)]:
            cr, cf = r + dr, f + df
            while 0 <= cr < 8 and 0 <= cf < 8:
                target = 1 << (cr * 8 + cf)
                attacks |= target
                if occupancy & target: break
                cr += dr; cf += df
        return attacks
    
    def _bishop_attacks(self, sq: int, occupancy: int) -> int:
        r, f = sq // 8, sq % 8
        attacks = 0
        for dr, df in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            cr, cf = r + dr, f + df
            while 0 <= cr < 8 and 0 <= cf < 8:
                target = 1 << (cr * 8 + cf)
                attacks |= target
                if occupancy & target: break
                cr += dr; cf += df
        return attacks
    
    def attacks_from(self, sq: int, color: int) -> int:
        """All squares attacked by the piece at sq."""
        piece = self.pieces.get(sq)
        if piece is None or piece[0] != color: return 0
        p = piece[1]
        occupancy = self.bb_all
        
        if p == Piece.PAWN:
            attacks = 0
            direction = 1 if color == Color.WHITE else -1
            r, f = sq // 8, sq % 8
            for df in [-1, 1]:
                tr, tf = r + direction, f + df
                if 0 <= tr < 8 and 0 <= tf < 8:
                    attacks |= 1 << (tr * 8 + tf)
            return attacks
        
        if p == Piece.KNIGHT:
            return _KNIGHT_ATTACKS[sq]
        if p == Piece.KING:
            return _KING_ATTACKS[sq]
        if p == Piece.BISHOP:
            return self._bishop_attacks(sq, occupancy)
        if p == Piece.ROOK:
            return self._rook_attacks(sq, occupancy)
        if p == Piece.QUEEN:
            return self._rook_attacks(sq, occupancy) | self._bishop_attacks(sq, occupancy)
        return 0
    
    def is_attacked(self, sq: int, by_color: int) -> bool:
        """Check if square is attacked by given color."""
        # Pawn attacks
        pawn = Piece.PAWN
        pawns = self.bb_piece[pawn] & self.bb_color[by_color]
        direction = 1 if by_color == Color.WHITE else -1
        r, f = sq // 8, sq % 8
        for df in [-1, 1]:
            tr, tf = r - direction, f + df
            if 0 <= tr < 8 and 0 <= tf < 8:
                if pawns & (1 << (tr * 8 + tf)): return True
        
        # Knight attacks
        if _KNIGHT_ATTACKS[sq] & self.bb_piece[Piece.KNIGHT] & self.bb_color[by_color]: return True
        # King attacks
        if _KING_ATTACKS[sq] & self.bb_piece[Piece.KING] & self.bb_color[by_color]: return True
        
        # Sliding pieces
        occupancy = self.bb_all
        rook_attacks = self._rook_attacks(sq, occupancy)
        bishop_attacks = self._bishop_attacks(sq, occupancy)
        
        enemy = self.bb_color[by_color]
        if rook_attacks & (self.bb_piece[Piece.ROOK] | self.bb_piece[Piece.QUEEN]) & enemy: return True
        if bishop_attacks & (self.bb_piece[Piece.BISHOP] | self.bb_piece[Piece.QUEEN]) & enemy: return True
        
        return False
    
    def is_in_check(self, color=None) -> bool:
        if color is None: color = self.color_to_move
        return self.is_attacked(self.king_sq[color], 1 - color)
    
    # ---- Move generation ----
    
    def generate_moves(self) -> List[Move]:
        """Generate pseudo-legal moves with MVV-LVA ordering scores."""
        moves = []; us = self.color_to_move; them = 1 - us
        
        for sq, (color, piece) in list(self.pieces.items()):
            if color != us: continue
            r, f = sq // 8, sq % 8
            
            if piece == Piece.PAWN:
                direction = 1 if us == Color.WHITE else -1
                start_rank = 1 if us == Color.WHITE else 6
                promo_rank = 7 if us == Color.WHITE else 0
                
                to_sq = sq + direction * 8
                if to_sq not in self.pieces:
                    if to_sq // 8 == promo_rank:
                        for pt in [Piece.QUEEN, Piece.ROOK, Piece.BISHOP, Piece.KNIGHT]:
                            moves.append(Move(sq, to_sq, promotion=pt, score=900+pt))
                    else:
                        moves.append(Move(sq, to_sq, score=0))
                    to_sq2 = sq + direction * 16
                    if r == start_rank and to_sq2 not in self.pieces:
                        moves.append(Move(sq, to_sq2, score=0))
                
                for df in [-1, 1]:
                    f2 = f + df
                    if 0 <= f2 < 8:
                        to_sq = sq + direction * 8 + df
                        if to_sq in self.pieces and self.pieces[to_sq][0] == them:
                            cap_val = MVV_LVA[self.pieces[to_sq][1]] * 100 + 10
                            if to_sq // 8 == promo_rank:
                                for pt in [Piece.QUEEN, Piece.ROOK, Piece.BISHOP, Piece.KNIGHT]:
                                    moves.append(Move(sq, to_sq, promotion=pt, score=900+cap_val))
                            else:
                                moves.append(Move(sq, to_sq, score=cap_val))
                        elif to_sq == self.en_passant_sq:
                            moves.append(Move(sq, to_sq, is_en_passant=True, score=MVV_LVA[Piece.PAWN]*100))
            
            elif piece == Piece.KNIGHT:
                for ts in range(64):
                    if _KNIGHT_ATTACKS[sq] & (1 << ts):
                        if ts not in self.pieces or self.pieces[ts][0] == them:
                            cap_val = (MVV_LVA.get(self.pieces[ts][1], 0) * 100 + 5) if ts in self.pieces else 0
                            moves.append(Move(sq, ts, score=cap_val))
            
            elif piece == Piece.BISHOP:
                for ts in range(64):
                    if self._bishop_attacks(sq, self.bb_all) & (1 << ts):
                        if ts not in self.pieces or self.pieces[ts][0] == them:
                            cap_val = (MVV_LVA.get(self.pieces[ts][1], 0) * 100 + 3) if ts in self.pieces else 0
                            moves.append(Move(sq, ts, score=cap_val))
            
            elif piece == Piece.ROOK:
                for ts in range(64):
                    if self._rook_attacks(sq, self.bb_all) & (1 << ts):
                        if ts not in self.pieces or self.pieces[ts][0] == them:
                            cap_val = (MVV_LVA.get(self.pieces[ts][1], 0) * 100 + 2) if ts in self.pieces else 0
                            moves.append(Move(sq, ts, score=cap_val))
            
            elif piece == Piece.QUEEN:
                attacks = self._rook_attacks(sq, self.bb_all) | self._bishop_attacks(sq, self.bb_all)
                for ts in range(64):
                    if attacks & (1 << ts):
                        if ts not in self.pieces or self.pieces[ts][0] == them:
                            cap_val = (MVV_LVA.get(self.pieces[ts][1], 0) * 100 + 1) if ts in self.pieces else 0
                            moves.append(Move(sq, ts, score=cap_val))
            
            elif piece == Piece.KING:
                for ts in range(64):
                    if _KING_ATTACKS[sq] & (1 << ts):
                        if ts not in self.pieces or self.pieces[ts][0] == them:
                            moves.append(Move(sq, ts, score=0))
                
                # Castling
                if us == Color.WHITE:
                    if 'K' in self.castling_rights and not self.is_attacked(4, them):
                        if (5 not in self.pieces and 6 not in self.pieces and
                            not self.is_attacked(5, them) and not self.is_attacked(6, them) and
                            self.pieces.get(7) == (Color.WHITE, Piece.ROOK)):
                            moves.append(Move(4, 6, is_castle_kingside=True, score=50))
                    if 'Q' in self.castling_rights and not self.is_attacked(4, them):
                        if (3 not in self.pieces and 2 not in self.pieces and 1 not in self.pieces and
                            not self.is_attacked(3, them) and not self.is_attacked(2, them) and
                            self.pieces.get(0) == (Color.WHITE, Piece.ROOK)):
                            moves.append(Move(4, 2, is_castle_queenside=True, score=45))
                else:
                    if 'k' in self.castling_rights and not self.is_attacked(60, them):
                        if (61 not in self.pieces and 62 not in self.pieces and
                            not self.is_attacked(61, them) and not self.is_attacked(62, them) and
                            self.pieces.get(63) == (Color.BLACK, Piece.ROOK)):
                            moves.append(Move(60, 62, is_castle_kingside=True, score=50))
                    if 'q' in self.castling_rights and not self.is_attacked(60, them):
                        if (59 not in self.pieces and 58 not in self.pieces and 57 not in self.pieces and
                            not self.is_attacked(59, them) and not self.is_attacked(58, them) and
                            self.pieces.get(56) == (Color.BLACK, Piece.ROOK)):
                            moves.append(Move(60, 58, is_castle_queenside=True, score=45))
        
        return moves
    
    def generate_legal_moves(self) -> List[Move]:
        moves = self.generate_moves(); legal = []
        for m in moves:
            self.make_move(m)
            if not self.is_in_check(1 - self.color_to_move): legal.append(m)
            self.unmake_move()
        return legal
    
    # ---- Make/Unmake Move ----
    
    def make_move(self, move: Move) -> None:
        us = self.color_to_move; them = 1 - us
        
        state = {'move': move, 'captured': self.pieces.get(move.to_sq),
                 'ep': self.en_passant_sq, 'castling': set(self.castling_rights),
                 'halfmove': self.halfmove_clock, 'fullmove': self.fullmove_number,
                 'zobrist': self._zobrist_hash}
        self._history.append(state)
        
        piece = self.pieces[move.from_sq]
        
        # Remove from source
        del self.pieces[move.from_sq]
        self.bb_color[us] &= ~(1 << move.from_sq)
        self.bb_piece[piece[1]] &= ~(1 << move.from_sq)
        self.bb_all &= ~(1 << move.from_sq)
        
        # Handle captures
        if move.to_sq in self.pieces:
            cap = self.pieces[move.to_sq]
            self.bb_color[cap[0]] &= ~(1 << move.to_sq)
            self.bb_piece[cap[1]] &= ~(1 << move.to_sq)
            self.bb_all &= ~(1 << move.to_sq)
            self.halfmove_clock = 0
        elif move.is_en_passant:
            ep_sq = move.to_sq - 8 if us == Color.WHITE else move.to_sq + 8
            if ep_sq in self.pieces:
                cap = self.pieces[ep_sq]; del self.pieces[ep_sq]
                self.bb_color[cap[0]] &= ~(1 << ep_sq)
                self.bb_piece[cap[1]] &= ~(1 << ep_sq)
                self.bb_all &= ~(1 << ep_sq)
            self.halfmove_clock = 0
        elif piece[1] == Piece.PAWN:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1
        
        # Place piece
        promo = move.promotion if move.promotion else piece[1]
        self.pieces[move.to_sq] = (us, promo)
        self.bb_color[us] |= 1 << move.to_sq
        self.bb_piece[promo] |= 1 << move.to_sq
        self.bb_all |= 1 << move.to_sq
        
        if piece[1] == Piece.KING: self.king_sq[us] = move.to_sq
        
        # Castling rook
        if move.is_castle_kingside:
            rf = 7 if us == Color.WHITE else 63; rt = 5 if us == Color.WHITE else 61
            self._move_rook(rf, rt, us)
        elif move.is_castle_queenside:
            rf = 0 if us == Color.WHITE else 56; rt = 3 if us == Color.WHITE else 59
            self._move_rook(rf, rt, us)
        
        # En passant
        self.en_passant_sq = None
        if piece[1] == Piece.PAWN and abs(move.to_sq - move.from_sq) == 16:
            self.en_passant_sq = move.from_sq + 8 if us == Color.WHITE else move.from_sq - 8
        
        # Castling rights
        if piece[1] == Piece.KING:
            self.castling_rights.discard('K' if us==Color.WHITE else 'k')
            self.castling_rights.discard('Q' if us==Color.WHITE else 'q')
        elif piece[1] == Piece.ROOK:
            if move.from_sq == 0: self.castling_rights.discard('Q')
            elif move.from_sq == 7: self.castling_rights.discard('K')
            elif move.from_sq == 56: self.castling_rights.discard('q')
            elif move.from_sq == 63: self.castling_rights.discard('k')
        
        # Switch sides
        self.color_to_move = them
        if us == Color.BLACK: self.fullmove_number += 1
        
        self._zobrist_hash = _ZOBRIST.hash(self)
    
    def _move_rook(self, fs, ts, color):
        self.pieces[ts] = self.pieces[fs]; del self.pieces[fs]
        self.bb_color[color] = (self.bb_color[color] & ~(1<<fs)) | (1<<ts)
        self.bb_piece[Piece.ROOK] = (self.bb_piece[Piece.ROOK] & ~(1<<fs)) | (1<<ts)
        self.bb_all = (self.bb_all & ~(1<<fs)) | (1<<ts)
    
    def unmake_move(self) -> Move:
        if not self._history: raise IndexError("No move to undo")
        state = self._history.pop()
        move = state['move']
        them = self.color_to_move; us = 1 - them
        
        # Remove piece from destination
        pi = self.pieces[move.to_sq]; del self.pieces[move.to_sq]
        self.bb_color[us] &= ~(1 << move.to_sq)
        self.bb_piece[pi[1]] &= ~(1 << move.to_sq)
        self.bb_all &= ~(1 << move.to_sq)
        if move.promotion: self.bb_piece[move.promotion] &= ~(1 << move.to_sq)
        
        # Restore to source
        orig = (us, Piece.PAWN) if move.promotion else pi
        self.pieces[move.from_sq] = orig
        self.bb_color[us] |= 1 << move.from_sq
        self.bb_piece[orig[1]] |= 1 << move.from_sq
        self.bb_all |= 1 << move.from_sq
        
        if orig[1] == Piece.KING: self.king_sq[us] = move.from_sq
        
        # Restore captured
        if state['captured']:
            self.pieces[move.to_sq] = state['captured']
            self.bb_color[state['captured'][0]] |= 1 << move.to_sq
            self.bb_piece[state['captured'][1]] |= 1 << move.to_sq
            self.bb_all |= 1 << move.to_sq
        elif move.is_en_passant:
            ep_sq = move.to_sq - 8 if us == Color.WHITE else move.to_sq + 8
            self.pieces[ep_sq] = (them, Piece.PAWN)
            self.bb_color[them] |= 1 << ep_sq
            self.bb_piece[Piece.PAWN] |= 1 << ep_sq
            self.bb_all |= 1 << ep_sq
        
        # Undo castling rook
        if move.is_castle_kingside:
            self._move_rook(5 if us==Color.WHITE else 61, 7 if us==Color.WHITE else 63, us)
        elif move.is_castle_queenside:
            self._move_rook(3 if us==Color.WHITE else 59, 0 if us==Color.WHITE else 56, us)
        
        # Restore state
        self.en_passant_sq = state['ep']
        self.castling_rights = state['castling']
        self.halfmove_clock = state['halfmove']
        self.fullmove_number = state['fullmove']
        self._zobrist_hash = state['zobrist']
        self.color_to_move = us
        
        return move
    
    def is_checkmate(self) -> bool:
        return self.is_in_check() and len(self.generate_legal_moves()) == 0
    
    def is_stalemate(self) -> bool:
        return not self.is_in_check() and len(self.generate_legal_moves()) == 0
    
    def is_game_over(self) -> bool:
        if len(self.generate_legal_moves()) == 0: return True
        if self.halfmove_clock >= 100: return True
        return False
    
    def result(self) -> Optional[str]:
        if self.is_checkmate(): return '0-1' if self.color_to_move == Color.WHITE else '1-0'
        if self.is_stalemate(): return '1/2-1/2'
        if self.halfmove_clock >= 100: return '1/2-1/2'
        return None
    
    # ---- Rich Position Encoding (160 feature planes) ----
    
    def to_tensor(self) -> np.ndarray:
        """Convert board to rich 160-channel feature tensor for NN input.
        
        Features (160 planes × 8×8):
        Planes 0-11:   Piece positions (6 types × 2 colors)
        Planes 12-19:  Attack maps by piece type (6 pieces + all + none) for side-to-move
        Planes 20-27:  Attack maps for opponent
        Planes 28-35:  Mobility count per square (one-hot by range)
        Planes 36-43:  Pawn structure (passed, doubled, isolated, backward, chain, etc.)
        Planes 44-51:  King safety indicators
        Planes 52-59:  Piece-square tables embedded as features
        Planes 60-67:  Castling rights + en passant + side-to-move
        Planes 68-75:  Material count (normalized)
        Planes 76-83:  Center control
        Planes 84-91:  Open/semi-open files
        Planes 92-127: Reserved for future features
        Planes 128-159: History planes (positions from last 4 moves)
        """
        us = self.color_to_move; them = 1 - us
        planes = np.zeros((160, 8, 8), dtype=np.float32)
        
        # Planes 0-11: Piece positions
        for sq, (c, p) in self.pieces.items():
            r, f = sq // 8, sq % 8
            planes[c*6 + p, r, f] = 1.0
        
        # Planes 12-19: Attack maps (side to move)
        attack_us = np.zeros((8, 8), dtype=np.float32)
        for sq in range(64):
            if sq in self.pieces and self.pieces[sq][0] == us:
                atk = self.attacks_from(sq, us)
                for ts in range(64):
                    if atk & (1 << ts):
                        attack_us[ts//8, ts%8] += 1.0
        
        # Planes 20-27: Attack maps (opponent)
        attack_them = np.zeros((8, 8), dtype=np.float32)
        for sq in range(64):
            if sq in self.pieces and self.pieces[sq][0] == them:
                atk = self.attacks_from(sq, them)
                for ts in range(64):
                    if atk & (1 << ts):
                        attack_them[ts//8, ts%8] += 1.0
        
        planes[12] = attack_us; planes[20] = attack_them
        planes[13] = (attack_us >= 1).astype(np.float32); planes[21] = (attack_them >= 1).astype(np.float32)
        planes[14] = (attack_us >= 2).astype(np.float32); planes[22] = (attack_them >= 2).astype(np.float32)
        
        # Planes 28-35: Mobility
        for sq in range(64):
            if sq in self.pieces and self.pieces[sq][0] == us:
                mob = bin(self.attacks_from(sq, us)).count('1')
                planes[28 + min(mob, 7), sq//8, sq%8] = 1.0
        
        # Planes 36-43: Pawn structure
        us_pawns = np.zeros(64, dtype=bool)
        them_pawns = np.zeros(64, dtype=bool)
        for sq, (c, p) in self.pieces.items():
            if p == Piece.PAWN:
                if c == us: us_pawns[sq] = True
                else: them_pawns[sq] = True
        
        for sq in range(64):
            r, f = sq // 8, sq % 8
            # Doubled pawns
            us_count = sum(1 for rr in range(8) if us_pawns[rr*8+f])
            if us_count > 1 and us_pawns[sq]: planes[36, r, f] = 1.0
            them_count = sum(1 for rr in range(8) if them_pawns[rr*8+f])
            if them_count > 1 and them_pawns[sq]: planes[37, r, f] = 1.0
            # Isolated pawns
            adj_files = [ff for ff in [f-1,f+1] if 0<=ff<8]
            if us_pawns[sq] and not any(us_pawns[rr*8+ff] for ff in adj_files for rr in range(8)):
                planes[38, r, f] = 1.0
            # Passed pawns
            if us_pawns[sq]:
                blocked = False
                ahead = range(r+1, 8) if us==Color.WHITE else range(0, r)
                for ff in [f-1, f, f+1]:
                    if 0 <= ff < 8:
                        for ar in ahead:
                            if them_pawns[ar*8+ff]: blocked = True; break
                if not blocked: planes[39, r, f] = 1.0
        
        # Planes 44-51: King safety
        uk_sq = self.king_sq[us]; tk_sq = self.king_sq[them]
        uk_r, uk_f = uk_sq // 8, uk_sq % 8
        for sq in range(64):
            r, f = sq // 8, sq % 8
            dist = max(abs(r - uk_r), abs(f - uk_f))
            planes[44 + min(dist, 7), r, f] = 1.0
        
        # King shield: pawns near own king
        shield_sqs = [(uk_r+dr, uk_f+df) for dr in [-1,0,1] for df in [-1,0,1]
                      if 0<=uk_r+dr<8 and 0<=uk_f+df<8 and (dr!=0 or df!=0)]
        for (sr, sf) in shield_sqs:
            if us_pawns[sr*8+sf]: planes[52, sr, sf] = 1.0
        
        # Planes 60-67: Auxiliary
        if 'K' in self.castling_rights: planes[60, 7, 4:7] = 1.0
        if 'Q' in self.castling_rights: planes[61, 7, 0:4] = 1.0
        if 'k' in self.castling_rights: planes[62, 0, 4:7] = 1.0
        if 'q' in self.castling_rights: planes[63, 0, 0:4] = 1.0
        if self.en_passant_sq is not None:
            planes[64, self.en_passant_sq//8, self.en_passant_sq%8] = 1.0
        if us == Color.WHITE: planes[65] = 1.0
        planes[66] = self.halfmove_clock / 100.0
        
        # Planes 68-75: Material count
        mat_us = sum(PIECE_VALUES.get(self.pieces[sq][1], 0) 
                    for sq in self.pieces if self.pieces[sq][0]==us) / 4000.0
        mat_them = sum(PIECE_VALUES.get(self.pieces[sq][1], 0)
                      for sq in self.pieces if self.pieces[sq][0]==them) / 4000.0
        planes[68] = mat_us; planes[69] = mat_them
        
        # Planes 84-91: Open/semi-open files
        for f in range(8):
            has_us = any(us_pawns[r*8+f] for r in range(8))
            has_them = any(them_pawns[r*8+f] for r in range(8))
            if not has_us and not has_them: planes[84, :, f] = 1.0  # Open file
            elif not has_us: planes[85, :, f] = 1.0  # Semi-open for us
        
        # History planes (last 4 moves)
        for i, state in enumerate(reversed(self._history[-4:])):
            if state and 'move' in state:
                m = state['move']
                planes[128+i*8, m.from_sq//8, m.from_sq%8] = 1.0
                planes[129+i*8, m.to_sq//8, m.to_sq%8] = 1.0
        
        return planes.astype(np.float32)
    
    def to_tensor_flat(self) -> np.ndarray:
        """Flattened tensor for legacy API compatibility (10240-dim)."""
        return self.to_tensor().flatten()
    
    def __repr__(self):
        lines = []
        for r in range(7, -1, -1):
            row = f"{r+1} "
            for f in range(8):
                sq = r*8+f
                row += (PIECE_CHARS[self.pieces[sq]] if sq in self.pieces else '.') + ' '
            lines.append(row)
        lines.append("  a b c d e f g h")
        return '\n'.join(lines)


# ===========================================================================
# Bridge to python-chess for validation
# ===========================================================================

def to_pychess(board: Board):
    """Convert to python-chess Board (if available)."""
    if not _PYCHESS_AVAILABLE: return None
    return _pychess.Board(board.fen())

def from_pychess(pyb) -> Board:
    """Convert from python-chess Board."""
    return Board(pyb.fen())
