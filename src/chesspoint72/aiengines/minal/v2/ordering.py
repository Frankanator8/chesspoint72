"""Minal v2 move ordering policy.

Improvements over v1:
- Reads the search's own killer table and history table directly, so moves
  that caused beta-cutoffs earlier in the search rank above generic quiet moves.
- Same SEE-gated capture splitting as v1.
- Same development + centralization + pawn-advance bonuses for quiet moves.

Usage: call attach_search(search) before the first search begins so the policy
has access to the live tables.  MinalV2Search does this automatically.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from chesspoint72.engine.core.policies import MoveOrderingPolicy
from chesspoint72.engine.core.types import Move, PieceType
from chesspoint72.engine.ordering.mvv_lva import score_capture
from chesspoint72.engine.ordering.see import SEE_VALUES, see_ge

if TYPE_CHECKING:
    from chesspoint72.engine.ordering.heuristics import HistoryTable, KillerMoveTable


# ---------------------------------------------------------------------------
# Score tiers (descending priority)
# ---------------------------------------------------------------------------
_TT_SCORE         = 10_000_000
_WINNING_CAP_BASE =  2_000_000   # SEE >= 0; +victim_see_value inside tier
_QUEEN_PROMO      =  1_500_000
_MINOR_PROMO_BASE =  1_400_000
_KILLER_0_SCORE   =    900_000   # most-recent killer at this ply
_KILLER_1_SCORE   =    800_000   # second-most-recent killer at this ply
_LOSING_CAP_BASE  =     -1_000   # SEE < 0; least-losing first


def _move_key(move: Move) -> tuple[int, int, int]:
    return (
        move.from_square,
        move.to_square,
        0 if move.promotion_piece is None else int(move.promotion_piece),
    )


def _moves_equal(a: Move | None, b: Move) -> bool:
    if a is None:
        return False
    return (a.from_square == b.from_square
            and a.to_square == b.to_square
            and a.promotion_piece == b.promotion_piece)


def _development_bonus(move: Move, stm: int) -> int:
    """Reward lifting a piece off the back rank (+12)."""
    back_rank = 0 if stm == 0 else 7
    return 12 if (move.from_square >> 3) == back_rank else 0


def _pawn_advance_bonus(move: Move, stm: int) -> int:
    """Reward pawn pushes into the opponent's half (up to +16)."""
    to_rank = move.to_square >> 3
    if stm == 0:
        return max(0, to_rank - 3) * 4
    return max(0, 4 - to_rank) * 4


def _victim_type(bbs: list[int], sq: int) -> int:
    bit = 1 << sq
    if bbs[0]  & bit or bbs[6]  & bit: return 1  # PAWN
    if bbs[1]  & bit or bbs[7]  & bit: return 2  # KNIGHT
    if bbs[2]  & bit or bbs[8]  & bit: return 3  # BISHOP
    if bbs[3]  & bit or bbs[9]  & bit: return 4  # ROOK
    if bbs[4]  & bit or bbs[10] & bit: return 5  # QUEEN
    if bbs[5]  & bit or bbs[11] & bit: return 6  # KING
    return 0


def _is_pawn(bbs: list[int], sq: int, stm: int) -> bool:
    return bool(bbs[stm * 6] & (1 << sq))


class MinalV2MoveOrderingPolicy(MoveOrderingPolicy):
    """Minal v2 tiered ordering with killer + history awareness.

    Tiers (searched in order):
    1. TT best move
    2. SEE >= 0 captures (by victim value)
    3. Queen promotions
    4. Under-promotions
    5. Killer move 0 (most recent beta-cutoff quiet at this ply)
    6. Killer move 1 (second-most recent)
    7. History-scored quiet moves (centralization + dev + history bonus)
    8. SEE < 0 captures (losing captures, MVV-LVA ordered within tier)
    """

    def __init__(self) -> None:
        self._search: object | None = None

    def attach_search(self, search: object) -> None:
        """Register the live search so we can read _ply, killer_table, history_table."""
        self._search = search

    def _get_killer_and_history(
        self,
    ) -> tuple[Move | None, Move | None, HistoryTable | None]:
        s = self._search
        if s is None:
            return None, None, None
        ply = getattr(s, "_ply", 0)
        killers = s.killer_table.get(ply)  # type: ignore[attr-defined]
        history = s.history_table           # type: ignore[attr-defined]
        return killers[0], killers[1], history

    def order_moves(
        self,
        moves: list[Move],
        board,
        tt_best_move: Move | None = None,
    ) -> list[Move]:
        if len(moves) < 2:
            return moves

        tt_key = _move_key(tt_best_move) if tt_best_move is not None else None
        stm = board.side_to_move.value
        killer0, killer1, history = self._get_killer_and_history()
        bbs = board.bitboards

        scored: list[tuple[int, int, Move]] = []
        for idx, move in enumerate(moves):

            if tt_key is not None and _move_key(move) == tt_key:
                score = _TT_SCORE

            elif move.is_capture:
                if see_ge(board, move, 0):
                    vtype = _victim_type(bbs, move.to_square)
                    score = _WINNING_CAP_BASE + (SEE_VALUES[vtype] if vtype else 0)
                else:
                    score = _LOSING_CAP_BASE + score_capture(move, board)

            elif move.promotion_piece is not None:
                if move.promotion_piece == PieceType.QUEEN:
                    score = _QUEEN_PROMO
                else:
                    score = _MINOR_PROMO_BASE + int(move.promotion_piece)

            elif _moves_equal(killer0, move):
                score = _KILLER_0_SCORE

            elif _moves_equal(killer1, move):
                score = _KILLER_1_SCORE

            else:
                # Quiet move: centralization + development + pawn advance + history
                to_file = move.to_square & 7
                to_rank = move.to_square >> 3
                center_dist = abs(2 * to_file - 7) + abs(2 * to_rank - 7)
                score = (32 - center_dist) + _development_bonus(move, stm)
                if _is_pawn(bbs, move.from_square, stm):
                    score += _pawn_advance_bonus(move, stm)

                if history is not None:
                    raw = history.get(board.side_to_move, move.from_square, move.to_square)
                    # Normalize: cap at 6000, scale to 0-60 so history stays
                    # below killers but can still differentiate quiet moves.
                    score += min(raw, 6_000) // 100

            scored.append((score, -idx, move))

        scored.sort(reverse=True)
        return [entry[2] for entry in scored]
