"""Minal's move ordering policy.

Distinct from Frank v3 in three ways:
1. SEE-gated capture splitting — winning/neutral captures (SEE >= 0) rise to
   tier 2; losing captures fall to the last tier and are searched last.
2. Development bonus — quiet moves that lift a piece off the back rank score
   extra, encouraging development without explicit opening logic.
3. Pawn-advance bonus — passed-pawn-style forward incentive for pawn pushes
   into the opponent's half of the board.
"""
from __future__ import annotations

from chesspoint72.engine.core.policies import MoveOrderingPolicy
from chesspoint72.engine.core.types import Move, PieceType
from chesspoint72.engine.ordering.mvv_lva import score_capture
from chesspoint72.engine.ordering.see import SEE_VALUES, see_ge


# ---------------------------------------------------------------------------
# Score tiers (descending priority)
# ---------------------------------------------------------------------------
_TT_SCORE         = 10_000_000
_WINNING_CAP_BASE =  2_000_000   # SEE >= 0, ordered by victim value inside
_QUEEN_PROMO      =  1_500_000
_MINOR_PROMO_BASE =  1_400_000   # + promotion piece ordinal
_LOSING_CAP_BASE  =     -1_000   # SEE < 0; placed last, least-losing first


def _move_key(move: Move) -> tuple[int, int, int]:
    return (
        move.from_square,
        move.to_square,
        0 if move.promotion_piece is None else int(move.promotion_piece),
    )


def _development_bonus(move: Move, side_to_move_value: int) -> int:
    """Reward lifting a piece off the back rank.

    White's back rank is rank 0 (squares 0–7).
    Black's back rank is rank 7 (squares 56–63).
    A move away from the back rank scores +12; moving *onto* the back rank
    scores 0 so we don't incentivise retreating there either.
    """
    from_rank = move.from_square >> 3
    back_rank = 0 if side_to_move_value == 0 else 7
    return 12 if from_rank == back_rank else 0


def _pawn_advance_bonus(move: Move, side_to_move_value: int) -> int:
    """Bonus for pushing pawns into the opponent's half."""
    to_rank = move.to_square >> 3
    if side_to_move_value == 0:   # White — higher ranks are forward
        return max(0, to_rank - 3) * 4
    else:                          # Black — lower ranks are forward
        return max(0, 4 - to_rank) * 4


class MinalV1MoveOrderingPolicy(MoveOrderingPolicy):
    """Minal v1 tiered move ordering.

    Tiers (searched in order):
    1. TT best move
    2. Captures with SEE >= 0, sorted by victim value (best victim first)
    3. Queen promotions
    4. Under-promotions
    5. Quiet moves: centralization + development + pawn-advance bonuses
    6. Captures with SEE < 0 (losing captures), least-losing first
    """

    def order_moves(
        self,
        moves: list[Move],
        board,
        tt_best_move: Move | None = None,
    ) -> list[Move]:
        if len(moves) < 2:
            return moves

        tt_key = _move_key(tt_best_move) if tt_best_move is not None else None
        stm = board.side_to_move.value  # 0=WHITE, 1=BLACK

        scored: list[tuple[int, int, Move]] = []
        for idx, move in enumerate(moves):
            key = _move_key(move)

            if tt_key is not None and key == tt_key:
                score = _TT_SCORE

            elif move.is_capture:
                # Separate winning/neutral from losing captures via SEE.
                if see_ge(board, move, 0):
                    # SEE >= 0: winning or neutral — order by victim value.
                    victim_bb_idx = _victim_piece_type_value(board.bitboards, move.to_square)
                    victim_val = SEE_VALUES[victim_bb_idx] if victim_bb_idx else 0
                    score = _WINNING_CAP_BASE + victim_val
                else:
                    # SEE < 0: losing capture — defer to last tier.
                    score = _LOSING_CAP_BASE + score_capture(move, board)

            elif move.promotion_piece is not None:
                if move.promotion_piece == PieceType.QUEEN:
                    score = _QUEEN_PROMO
                else:
                    score = _MINOR_PROMO_BASE + int(move.promotion_piece)

            else:
                # Quiet move: centralization + development + pawn advance.
                to_file = move.to_square & 7
                to_rank = move.to_square >> 3
                center_dist = abs(2 * to_file - 7) + abs(2 * to_rank - 7)
                score = (32 - center_dist) + _development_bonus(move, stm)
                if _is_pawn(board.bitboards, move.from_square, stm):
                    score += _pawn_advance_bonus(move, stm)

            scored.append((score, -idx, move))

        scored.sort(reverse=True)
        return [entry[2] for entry in scored]


# ---------------------------------------------------------------------------
# Lightweight bitboard helpers
# ---------------------------------------------------------------------------

def _victim_piece_type_value(bbs: list[int], sq: int) -> int:
    """Return the SEE_VALUES index (1–6) for the piece on *sq*, or 0 if empty."""
    bit = 1 << sq
    if bbs[0]  & bit or bbs[6]  & bit: return 1  # PAWN
    if bbs[1]  & bit or bbs[7]  & bit: return 2  # KNIGHT
    if bbs[2]  & bit or bbs[8]  & bit: return 3  # BISHOP
    if bbs[3]  & bit or bbs[9]  & bit: return 4  # ROOK
    if bbs[4]  & bit or bbs[10] & bit: return 5  # QUEEN
    if bbs[5]  & bit or bbs[11] & bit: return 6  # KING
    return 0


def _is_pawn(bbs: list[int], sq: int, stm: int) -> bool:
    """Return True if the piece on *sq* is a pawn for the side to move."""
    bit = 1 << sq
    return bool(bbs[stm * 6] & bit)
