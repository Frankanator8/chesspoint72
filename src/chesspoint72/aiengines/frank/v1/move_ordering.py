from __future__ import annotations

from typing import TYPE_CHECKING

from chesspoint72.engine.core.policies import MoveOrderingPolicy
from chesspoint72.engine.ordering.heuristics import HistoryTable
from chesspoint72.engine.ordering.mvv_lva import score_capture

if TYPE_CHECKING:
    from chesspoint72.engine.core.board import Board
    from chesspoint72.engine.core.types import Move


class FrankMoveOrdering(MoveOrderingPolicy):
    """TT-first, MVV-LVA captures, history-sorted quiets.

    Shares its ``HistoryTable`` reference with ``NegamaxSearch`` so the search's
    own β-cutoff bookkeeping (``update_history``) feeds future ordering.
    """

    def __init__(self, history: HistoryTable) -> None:
        self._history = history

    def order_moves(
        self,
        moves: list[Move],
        board: Board,
        tt_best_move: Move | None = None,
    ) -> list[Move]:
        history = self._history.scores[board.side_to_move.value]

        tt_move: Move | None = None
        captures: list[tuple[int, Move]] = []
        quiets: list[tuple[int, Move]] = []

        tt_key = None
        if tt_best_move is not None:
            tt_key = (
                tt_best_move.from_square,
                tt_best_move.to_square,
                tt_best_move.promotion_piece,
            )

        for move in moves:
            if tt_key is not None and (
                move.from_square == tt_key[0]
                and move.to_square == tt_key[1]
                and move.promotion_piece == tt_key[2]
            ):
                tt_move = move
                continue
            if move.is_capture:
                captures.append((score_capture(move, board), move))
            else:
                quiets.append((history[move.from_square][move.to_square], move))

        captures.sort(key=lambda p: p[0], reverse=True)
        quiets.sort(key=lambda p: p[0], reverse=True)

        ordered: list[Move] = []
        if tt_move is not None:
            ordered.append(tt_move)
        ordered.extend(m for _, m in captures)
        ordered.extend(m for _, m in quiets)
        return ordered
