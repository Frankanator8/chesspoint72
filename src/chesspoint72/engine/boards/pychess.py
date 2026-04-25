"""PyChessBoard — a concrete Board ABC backed by python-chess.

Wraps a ``chess.Board`` and exposes the engine's ``Board`` interface, so any
``Search`` implementation that talks to the ABC (NegamaxSearch et al.) can
operate on a position with real legal-move generation, make/unmake, and
zobrist hashing — without us having to write a bitboard backend from scratch.

Move conversion is done at the boundary: python-chess uses ``chess.Move``
(from, to, promotion-int); we use the engine's ``Move`` dataclass (from, to,
promotion-PieceType, is_capture). PieceType values 1..6 happen to match
python-chess's ``chess.PAWN``..``chess.KING`` integers, so the cast is direct.
"""
# @capability: board
# @capability: move_generation
from __future__ import annotations

import chess
import chess.polyglot

from chesspoint72.engine.core.board import Board
from chesspoint72.engine.core.types import Color, Move, PieceType


def _engine_move(py_board: chess.Board, m: chess.Move) -> Move:
    promo = PieceType(m.promotion) if m.promotion is not None else None
    return Move(
        from_square=m.from_square,
        to_square=m.to_square,
        promotion_piece=promo,
        is_capture=py_board.is_capture(m),
    )


def _pychess_move(m: Move) -> chess.Move:
    return chess.Move(
        from_square=m.from_square,
        to_square=m.to_square,
        promotion=int(m.promotion_piece) if m.promotion_piece is not None else None,
    )


class PyChessBoard(Board):
    """Board ABC implementation using python-chess as the move-gen backend."""

    def __init__(self) -> None:
        super().__init__()
        self._py_board = chess.Board()
        self._null_stack: list[chess.Move] = []
        self._refresh_state()

    # ------------------------------------------------------------------ #
    # Board ABC
    # ------------------------------------------------------------------ #

    def set_position_from_fen(self, fen_string: str) -> None:
        self._py_board.set_fen(fen_string)
        self.move_history.clear()
        self._null_stack.clear()
        self._refresh_state()

    def get_current_fen(self) -> str:
        return self._py_board.fen()

    def generate_legal_moves(self) -> list[Move]:
        pyb = self._py_board
        is_capture = pyb.is_capture
        out: list[Move] = []
        append = out.append
        for m in pyb.legal_moves:
            promo = PieceType(m.promotion) if m.promotion is not None else None
            append(Move(m.from_square, m.to_square, promo, is_capture(m)))
        return out

    def make_move(self, move: Move) -> None:
        self._py_board.push(_pychess_move(move))
        self.move_history.append(move)
        self._refresh_state()

    def unmake_move(self) -> None:
        self._py_board.pop()
        if self.move_history:
            self.move_history.pop()
        self._refresh_state()

    def is_king_in_check(self) -> bool:
        return self._py_board.is_check()

    def calculate_zobrist_hash(self) -> int:
        return chess.polyglot.zobrist_hash(self._py_board)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @property
    def py_board(self) -> chess.Board:
        """Underlying python-chess board, exposed for adapters/tests."""
        return self._py_board

    def push_uci(self, uci: str) -> None:
        """Apply a UCI move string (e.g. 'e2e4', 'e7e8q'). Used by the controller
        when replaying ``position … moves …``."""
        self._py_board.push_uci(uci)
        # No engine Move object retained for UCI replays — search relies on
        # zobrist-hash equality, not the move history list.
        self._refresh_state()

    def is_game_over(self) -> bool:
        return self._py_board.is_game_over(claim_draw=True)

    def make_null_move(self) -> None:
        """Apply a null move (pass the turn). Only call when not in check."""
        self._py_board.push(chess.Move.null())
        self._null_stack.append(chess.Move.null())
        self._refresh_state()

    def unmake_null_move(self) -> None:
        """Undo the most recent null move."""
        self._py_board.pop()
        if self._null_stack:
            self._null_stack.pop()
        self._refresh_state()

    def has_only_king_and_pawns(self, side) -> bool:
        """True iff *side* holds only king and pawns (no minor/major pieces).

        Used by the zugzwang guard in null-move pruning.
        *side* is a Color enum value (0=WHITE, 1=BLACK).
        """
        chess_color = chess.WHITE if int(side) == int(Color.WHITE) else chess.BLACK
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            if self._py_board.pieces_mask(piece_type, chess_color):
                return False
        return True

    # ------------------------------------------------------------------ #
    # Internal: keep Board ABC slots in sync with the python-chess board
    # ------------------------------------------------------------------ #

    def _refresh_state(self) -> None:
        pyb = self._py_board
        self.side_to_move = Color.WHITE if pyb.turn else Color.BLACK
        self.halfmove_clock = pyb.halfmove_clock
        self.en_passant_target = pyb.ep_square
        w = pyb.occupied_co[chess.WHITE]
        b = pyb.occupied_co[chess.BLACK]
        self.bitboards = [
            pyb.pawns   & w, pyb.knights & w, pyb.bishops & w,
            pyb.rooks   & w, pyb.queens  & w, pyb.kings   & w,
            pyb.pawns   & b, pyb.knights & b, pyb.bishops & b,
            pyb.rooks   & b, pyb.queens  & b, pyb.kings   & b,
        ]
