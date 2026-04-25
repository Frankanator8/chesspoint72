"""Shims used ONLY by the Phase-5 correctness harness.

The real ``PruningConfig`` will be authored by a separate prompt — this file
defines the smallest stand-in that lets us run the search end-to-end so the
correctness tests have something to call. Do not promote any of these to
production.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from chesspoint72.engine.core.board import Board
from chesspoint72.engine.core.evaluator import Evaluator
from chesspoint72.engine.core.policies import MoveOrderingPolicy, PruningPolicy
from chesspoint72.engine.core.types import Move

import chess

from chesspoint72.forward_pruning.python_chess_board import PythonChessBoard


@dataclass
class TestPruningConfig:
    """Mirror of the real PruningConfig surface — see INTERFACE_CONTRACT.md."""

    nmp_enabled: bool = True
    futility_enabled: bool = True
    razoring_enabled: bool = True
    lmr_enabled: bool = True
    futility_margin: int = 200
    razoring_margins: tuple[int, int, int] = (300, 500, 900)
    lmr_min_depth: int = 3
    lmr_min_move_index: int = 3


# ---- Material-only evaluator (centipawns, side-to-move POV) ---- #

_PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


class MaterialEvaluator(Evaluator):
    def evaluate_position(self, board: Board) -> int:
        cb: chess.Board = board._board  # type: ignore[attr-defined]
        score = 0
        for piece_type, value in _PIECE_VALUE.items():
            score += value * (
                len(cb.pieces(piece_type, chess.WHITE))
                - len(cb.pieces(piece_type, chess.BLACK))
            )
        # Negamax POV: positive = good for side to move.
        return score if cb.turn == chess.WHITE else -score


class CapturesFirstOrdering(MoveOrderingPolicy):
    """Cheap ordering: TT move, then captures, then quiets."""

    def order_moves(
        self,
        moves: list[Move],
        board: Board,
        tt_best_move: Move | None = None,
    ) -> list[Move]:
        captures = [m for m in moves if m.is_capture]
        quiets = [m for m in moves if not m.is_capture]
        ordered = captures + quiets
        if tt_best_move is not None:
            for i, m in enumerate(ordered):
                if (
                    m.from_square == tt_best_move.from_square
                    and m.to_square == tt_best_move.to_square
                    and m.promotion_piece == tt_best_move.promotion_piece
                ):
                    ordered.pop(i)
                    ordered.insert(0, m)
                    break
        return ordered


class NullPruningPolicy(PruningPolicy):
    """The original extension hook is preserved in the modified search;
    we feed it a no-op so it never overlaps with the new module."""

    def try_prune(self, board, depth, alpha, beta, static_eval):
        return None


__all__ = [
    "TestPruningConfig",
    "MaterialEvaluator",
    "CapturesFirstOrdering",
    "NullPruningPolicy",
    "PythonChessBoard",
]
