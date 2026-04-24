from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag


class Color(IntEnum):
    WHITE = 0
    BLACK = 1


class PieceType(IntEnum):
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6


class CastlingRights(IntFlag):
    NONE = 0
    WHITE_KINGSIDE = 1
    WHITE_QUEENSIDE = 2
    BLACK_KINGSIDE = 4
    BLACK_QUEENSIDE = 8
    ALL = 15


class NodeType(IntEnum):
    EXACT = 0
    LOWER_BOUND = 1
    UPPER_BOUND = 2


_FILES = "abcdefgh"
_PROMOTION_CHAR = {
    PieceType.KNIGHT: "n",
    PieceType.BISHOP: "b",
    PieceType.ROOK: "r",
    PieceType.QUEEN: "q",
}


def square_to_algebraic(square: int) -> str:
    if not 0 <= square < 64:
        raise ValueError(f"square index out of range: {square}")
    return f"{_FILES[square & 7]}{(square >> 3) + 1}"


@dataclass
class Move:
    from_square: int
    to_square: int
    promotion_piece: PieceType | None = None
    is_capture: bool = False

    def to_uci_string(self) -> str:
        uci = square_to_algebraic(self.from_square) + square_to_algebraic(self.to_square)
        if self.promotion_piece is not None:
            uci += _PROMOTION_CHAR[self.promotion_piece]
        return uci
