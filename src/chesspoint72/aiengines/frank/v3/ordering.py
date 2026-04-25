from __future__ import annotations

from chesspoint72.engine.core.policies import MoveOrderingPolicy
from chesspoint72.engine.core.types import Move
from chesspoint72.engine.ordering.mvv_lva import score_capture


def _move_key(move: Move) -> tuple[int, int, int]:
    return (
        move.from_square,
        move.to_square,
        0 if move.promotion_piece is None else int(move.promotion_piece),
    )


class FrankV3MoveOrderingPolicy(MoveOrderingPolicy):
    """Practical move ordering for the current policy interface.

    Priorities:
    1) TT move
    2) captures by MVV-LVA
    3) promotions
    4) quiet move centralization bias
    """

    _TT_SCORE = 10_000_000
    _CAPTURE_BASE = 1_000_000
    _PROMOTION_BASE = 100_000

    def order_moves(
        self,
        moves: list[Move],
        board,
        tt_best_move: Move | None = None,
    ) -> list[Move]:
        if len(moves) < 2:
            return moves

        tt_key = _move_key(tt_best_move) if tt_best_move is not None else None

        scored: list[tuple[int, int, Move]] = []
        for idx, move in enumerate(moves):
            key = _move_key(move)
            if tt_key is not None and key == tt_key:
                score = self._TT_SCORE
            elif move.is_capture:
                score = self._CAPTURE_BASE + score_capture(move, board)
            else:
                score = 0
                if move.promotion_piece is not None:
                    score += self._PROMOTION_BASE + int(move.promotion_piece)

                to_file = move.to_square & 7
                to_rank = move.to_square >> 3
                center_distance_x2 = abs(2 * to_file - 7) + abs(2 * to_rank - 7)
                score += 32 - center_distance_x2

            scored.append((score, -idx, move))

        scored.sort(reverse=True)
        return [entry[2] for entry in scored]

