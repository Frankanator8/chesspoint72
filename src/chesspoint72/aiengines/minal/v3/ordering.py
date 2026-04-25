"""Minal v3 move ordering policy.

Adds the countermove heuristic on top of v2's killer + history ordering.

Countermove heuristic
---------------------
Every time a quiet move causes a beta-cutoff we record it as the "counter"
to the opponent's last move (indexed by from_sq × to_sq of that opponent
move).  At future nodes, if the opponent plays the same move again, the
recorded reply is given a bonus score sitting between killer moves and plain
history-scored quiet moves.

This is orthogonal to killers (which index by current depth) and history
(which index by colour × from × to regardless of what the opponent played).

Score tiers (descending priority)
----------------------------------
1.  TT best move          10 000 000
2.  SEE >= 0 captures      2 000 000 + victim_see_value
3.  Queen promotion        1 500 000
4.  Minor promotion        1 400 000 + piece_ordinal
5.  Killer move 0            900 000
6.  Killer move 1            800 000
7.  Countermove              700 000
8.  History-scored quiet           0 .. ~120
9.  SEE < 0 captures          -1 000 + mvv_lva
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from chesspoint72.engine.core.policies import MoveOrderingPolicy
from chesspoint72.engine.core.types import Move, PieceType
from chesspoint72.engine.ordering.mvv_lva import score_capture
from chesspoint72.engine.ordering.see import SEE_VALUES, see_ge

if TYPE_CHECKING:
    from chesspoint72.engine.ordering.heuristics import HistoryTable, KillerMoveTable

_TT_SCORE         = 10_000_000
_WINNING_CAP_BASE =  2_000_000
_QUEEN_PROMO      =  1_500_000
_MINOR_PROMO_BASE =  1_400_000
_KILLER_0         =    900_000
_KILLER_1         =    800_000
_COUNTERMOVE      =    700_000
_LOSING_CAP_BASE  =     -1_000


def _moves_equal(a: Move | None, b: Move) -> bool:
    return (a is not None
            and a.from_square == b.from_square
            and a.to_square == b.to_square
            and a.promotion_piece == b.promotion_piece)


def _development_bonus(move: Move, stm: int) -> int:
    back_rank = 0 if stm == 0 else 7
    return 12 if (move.from_square >> 3) == back_rank else 0


def _pawn_advance_bonus(move: Move, stm: int) -> int:
    to_rank = move.to_square >> 3
    if stm == 0:
        return max(0, to_rank - 3) * 4
    return max(0, 4 - to_rank) * 4


def _victim_type(bbs: list[int], sq: int) -> int:
    bit = 1 << sq
    if bbs[0]  & bit or bbs[6]  & bit: return 1
    if bbs[1]  & bit or bbs[7]  & bit: return 2
    if bbs[2]  & bit or bbs[8]  & bit: return 3
    if bbs[3]  & bit or bbs[9]  & bit: return 4
    if bbs[4]  & bit or bbs[10] & bit: return 5
    if bbs[5]  & bit or bbs[11] & bit: return 6
    return 0


def _is_pawn(bbs: list[int], sq: int, stm: int) -> bool:
    return bool(bbs[stm * 6] & (1 << sq))


class MinalV3MoveOrderingPolicy(MoveOrderingPolicy):
    """Killer + history + countermove ordering."""

    def __init__(self) -> None:
        self._search: object | None = None

    def attach_search(self, search: object) -> None:
        self._search = search

    def _context(self):
        """Return (ply, killer0, killer1, history, countermove) from search."""
        s = self._search
        if s is None:
            return 0, None, None, None, None
        ply = getattr(s, "_ply", 0)
        killers = s.killer_table.get(ply)       # type: ignore[attr-defined]
        history = s.history_table               # type: ignore[attr-defined]
        prev = getattr(s, "_prev_move", None)
        cm_table = getattr(s, "countermove_table", None)
        countermove = None
        if prev is not None and cm_table is not None:
            countermove = cm_table[prev.from_square][prev.to_square]
        return ply, killers[0], killers[1], history, countermove

    def order_moves(
        self,
        moves: list[Move],
        board,
        tt_best_move: Move | None = None,
    ) -> list[Move]:
        if len(moves) < 2:
            return moves

        tt_key = (tt_best_move.from_square, tt_best_move.to_square,
                  0 if tt_best_move.promotion_piece is None else int(tt_best_move.promotion_piece)
                  ) if tt_best_move is not None else None
        stm = board.side_to_move.value
        bbs = board.bitboards
        _, killer0, killer1, history, countermove = self._context()

        scored: list[tuple[int, int, Move]] = []
        for idx, move in enumerate(moves):

            mkey = (move.from_square, move.to_square,
                    0 if move.promotion_piece is None else int(move.promotion_piece))

            if tt_key is not None and mkey == tt_key:
                score = _TT_SCORE

            elif move.is_capture:
                if see_ge(board, move, 0):
                    vtype = _victim_type(bbs, move.to_square)
                    score = _WINNING_CAP_BASE + (SEE_VALUES[vtype] if vtype else 0)
                else:
                    score = _LOSING_CAP_BASE + score_capture(move, board)

            elif move.promotion_piece is not None:
                score = _QUEEN_PROMO if move.promotion_piece == PieceType.QUEEN else (
                    _MINOR_PROMO_BASE + int(move.promotion_piece))

            elif _moves_equal(killer0, move):
                score = _KILLER_0

            elif _moves_equal(killer1, move):
                score = _KILLER_1

            elif _moves_equal(countermove, move):
                score = _COUNTERMOVE

            else:
                to_file = move.to_square & 7
                to_rank = move.to_square >> 3
                center_dist = abs(2 * to_file - 7) + abs(2 * to_rank - 7)
                score = (32 - center_dist) + _development_bonus(move, stm)
                if _is_pawn(bbs, move.from_square, stm):
                    score += _pawn_advance_bonus(move, stm)
                if history is not None:
                    raw = history.get(board.side_to_move, move.from_square, move.to_square)
                    score += min(raw, 6_000) // 100

            scored.append((score, -idx, move))

        scored.sort(reverse=True)
        return [e[2] for e in scored]
