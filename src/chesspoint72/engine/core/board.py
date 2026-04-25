from __future__ import annotations

from abc import ABC, abstractmethod

from chesspoint72.engine.core.types import CastlingRights, Color, Move, PieceType


class Board(ABC):
    """Base class for board representations.

    Holds the game-state slots required by the search and evaluator.
    Subclasses own the move-generation, make/unmake, and hashing logic.
    """

    bitboards: list[int]
    side_to_move: Color
    castling_rights: CastlingRights
    en_passant_target: int | None
    halfmove_clock: int
    move_history: list

    def __init__(self) -> None:
        self.bitboards = [0] * 12
        self.side_to_move = Color.WHITE
        self.castling_rights = CastlingRights.ALL
        self.en_passant_target = None
        self.halfmove_clock = 0
        self.move_history = []

    @abstractmethod
    def set_position_from_fen(self, fen_string: str) -> None: ...

    @abstractmethod
    def get_current_fen(self) -> str: ...

    @abstractmethod
    def generate_legal_moves(self) -> list[Move]: ...

    @abstractmethod
    def make_move(self, move: Move) -> None: ...

    @abstractmethod
    def unmake_move(self) -> None: ...

    @abstractmethod
    def is_king_in_check(self) -> bool: ...

    @abstractmethod
    def calculate_zobrist_hash(self) -> int: ...

    def get_piece_at(self, square: int) -> tuple[PieceType, Color] | None:
        """Return (PieceType, Color) for the piece on *square*, or None if empty.

        Reads directly from the 12-entry bitboards list using the layout:
            index = color.value * 6 + (piece_type.value - 1)
        """
        bit = 1 << square
        for color in Color:
            for piece_type in PieceType:
                if self.bitboards[color.value * 6 + (piece_type.value - 1)] & bit:
                    return piece_type, color
        return None
