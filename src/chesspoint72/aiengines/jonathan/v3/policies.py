"""Concrete MoveOrderingPolicy / PruningPolicy implementations used by Calix.

These are deliberately small. The Calix engine reuses the upstream
``ForwardPruningPolicy`` for actual pruning; this module only adds two
things the upstream factory's stubs cannot express:

* ``CapturesFirstOrderingPolicy`` — TT-first + captures-first ordering that
  needs no shared search state, so it is safe to inject into NegamaxSearch
  without coupling its internal killer/history bookkeeping.
* ``StubMoveOrderingPolicy`` / ``StubPruningPolicy`` — duplicated locally so
  the engine package is self-contained per the README in this folder.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from chesspoint72.engine.core.policies import MoveOrderingPolicy, PruningPolicy

if TYPE_CHECKING:
    from chesspoint72.engine.core.board import Board
    from chesspoint72.engine.core.types import Move


class StubMoveOrderingPolicy(MoveOrderingPolicy):
    """Identity ordering — used in minimal mode."""

    def order_moves(
        self,
        moves: list[Move],
        board: Board,
        tt_best_move: Move | None = None,
    ) -> list[Move]:
        return moves


class StubPruningPolicy(PruningPolicy):
    """No-op pruning — used in minimal mode."""

    def try_prune(
        self,
        board: Board,
        depth: int,
        alpha: int,
        beta: int,
        static_eval: int,
    ) -> int | None:
        return None


class CapturesFirstOrderingPolicy(MoveOrderingPolicy):
    """Cheap ordering: TT move first, then captures, then quiets.

    This avoids the shared-state problems of full MoveSorter integration
    (which expects access to NegamaxSearch's own killer/history tables)
    while still beating identity ordering by a wide margin.
    """

    def order_moves(
        self,
        moves: list[Move],
        board: Board,
        tt_best_move: Move | None = None,
    ) -> list[Move]:
        if not moves:
            return moves
        tt_first: list[Move] = []
        captures: list[Move] = []
        quiets: list[Move] = []
        for move in moves:
            if (
                tt_best_move is not None
                and move.from_square == tt_best_move.from_square
                and move.to_square == tt_best_move.to_square
                and move.promotion_piece == tt_best_move.promotion_piece
            ):
                tt_first.append(move)
            elif move.is_capture:
                captures.append(move)
            else:
                quiets.append(move)
        return tt_first + captures + quiets
