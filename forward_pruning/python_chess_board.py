"""Concrete ``Board`` implementation backed by ``python-chess``.

Used by the Phase-5 correctness harness (perft, tactics, zugzwang test).
The upstream codebase only ships an empty ``_StubBoard``, so without this
adapter we couldn't exercise the new search end-to-end.

This adapter also implements the two extra hooks the pruning module needs:
  * ``has_only_king_and_pawns(side)`` — for ``is_zugzwang_risk``.
  * ``make_null_move()`` / ``unmake_null_move()`` — for NMP.
"""
from __future__ import annotations

import chess
import chess.polyglot  # explicit — `chess` does not auto-import submodules

from chesspoint72.engine.board import Board
from chesspoint72.engine.types import CastlingRights, Color, Move, PieceType


_PROMO_TO_PIECE_TYPE = {
    chess.KNIGHT: PieceType.KNIGHT,
    chess.BISHOP: PieceType.BISHOP,
    chess.ROOK: PieceType.ROOK,
    chess.QUEEN: PieceType.QUEEN,
}
_PIECE_TYPE_TO_PROMO = {v: k for k, v in _PROMO_TO_PIECE_TYPE.items()}


class PythonChessBoard(Board):
    """Adapts ``chess.Board`` to the project's ``Board`` ABC + pruning protocols."""

    def __init__(self, fen: str | None = None) -> None:
        super().__init__()
        self._board = chess.Board(fen) if fen is not None else chess.Board()
        # ``Board`` ABC declares ``move_history`` as a list — we shadow with our own
        # null-move stack as well so make/unmake_null_move can roundtrip cleanly.
        self.move_history = []
        self._null_stack: list[chess.Move] = []
        self._sync_state_fields()

    # ---------------------------------------------------------------------- #
    # Board ABC implementation
    # ---------------------------------------------------------------------- #

    def set_position_from_fen(self, fen_string: str) -> None:
        self._board = chess.Board(fen_string)
        self.move_history = []
        self._null_stack.clear()
        self._sync_state_fields()

    def get_current_fen(self) -> str:
        return self._board.fen()

    def generate_legal_moves(self) -> list[Move]:
        out: list[Move] = []
        for m in self._board.legal_moves:
            promo = _PROMO_TO_PIECE_TYPE.get(m.promotion) if m.promotion else None
            is_capture = self._board.is_capture(m)
            out.append(Move(
                from_square=m.from_square,
                to_square=m.to_square,
                promotion_piece=promo,
                is_capture=is_capture,
            ))
        return out

    def make_move(self, move: Move) -> None:
        promo = _PIECE_TYPE_TO_PROMO.get(move.promotion_piece) if move.promotion_piece else None
        cm = chess.Move(move.from_square, move.to_square, promotion=promo)
        self._board.push(cm)
        self.move_history.append(cm)
        self._sync_state_fields()

    def unmake_move(self) -> None:
        self._board.pop()
        if self.move_history:
            self.move_history.pop()
        self._sync_state_fields()

    def is_king_in_check(self) -> bool:
        return self._board.is_check()

    def calculate_zobrist_hash(self) -> int:
        # python-chess uses Polyglot Zobrist; good enough for TT keys.
        return chess.polyglot.zobrist_hash(self._board)

    # ---------------------------------------------------------------------- #
    # Hooks consumed by the pruning module
    # ---------------------------------------------------------------------- #

    def has_only_king_and_pawns(self, side) -> bool:
        """True iff *side* (Color or chess color int) holds only K and P."""
        chess_color = chess.WHITE if int(side) == int(Color.WHITE) else chess.BLACK
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            if self._board.pieces_mask(piece_type, chess_color):
                return False
        return True

    def make_null_move(self) -> None:
        # python-chess refuses null moves while in check, but NMP should never
        # be invoked in check — caller already gates on in_check.
        self._board.push(chess.Move.null())
        self._null_stack.append(chess.Move.null())
        self._sync_state_fields()

    def unmake_null_move(self) -> None:
        self._board.pop()
        if self._null_stack:
            self._null_stack.pop()
        self._sync_state_fields()

    # ---------------------------------------------------------------------- #
    # Internal
    # ---------------------------------------------------------------------- #

    def _sync_state_fields(self) -> None:
        """Mirror python-chess state into the ABC's declared slots.

        The ABC declares ``side_to_move``, ``castling_rights``, ``en_passant_target``,
        ``halfmove_clock`` — keep them coherent with the underlying chess.Board so
        anything else that reads these fields sees consistent values.
        """
        self.side_to_move = Color.WHITE if self._board.turn == chess.WHITE else Color.BLACK

        rights = CastlingRights.NONE
        if self._board.has_kingside_castling_rights(chess.WHITE):
            rights |= CastlingRights.WHITE_KINGSIDE
        if self._board.has_queenside_castling_rights(chess.WHITE):
            rights |= CastlingRights.WHITE_QUEENSIDE
        if self._board.has_kingside_castling_rights(chess.BLACK):
            rights |= CastlingRights.BLACK_KINGSIDE
        if self._board.has_queenside_castling_rights(chess.BLACK):
            rights |= CastlingRights.BLACK_QUEENSIDE
        self.castling_rights = rights

        self.en_passant_target = self._board.ep_square
        self.halfmove_clock = self._board.halfmove_clock
