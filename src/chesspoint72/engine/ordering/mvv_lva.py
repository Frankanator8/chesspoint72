"""MVV-LVA (Most Valuable Victim - Least Valuable Aggressor) capture scoring."""
from __future__ import annotations

from typing import TYPE_CHECKING

from chesspoint72.engine.core.types import PieceType

if TYPE_CHECKING:
    from chesspoint72.engine.core.board import Board
    from chesspoint72.engine.core.types import Move

# Piece values indexed by PieceType.value (1=PAWN..6=KING); index 0 unused.
# P=1, N=3, B=3, R=5, Q=9, K=0
_VALUES: list[int] = [0, 1, 3, 3, 5, 9, 0]

# Precomputed 2D scoring matrix: MVV_LVA[victim.value][aggressor.value]
# Score = (victim_value * 10) - aggressor_value
#
# Higher score  → search captures of valuable pieces by cheap pieces first.
# Indices 0 are unused (PieceType values start at 1).
#
#               aggressor →   _  P   N   B   R   Q   K
#                             0  1   2   3   4   5   6
MVV_LVA: list[list[int]] = [
    [(_VALUES[v] * 10) - _VALUES[a] for a in range(7)]
    for v in range(7)
]


def score_capture(move: Move, board: Board) -> int:
    """Return the MVV-LVA score for a capture move.

    Score = (victim_value * 10) - aggressor_value, using the precomputed
    MVV_LVA matrix.  For en-passant captures the victim square is empty on
    the board; the function assumes a pawn victim in that case (always correct
    for legal en-passant).

    Args:
        move: A capture move (move.is_capture == True).
        board: Current position used for piece-type lookups via get_piece_at.

    Returns:
        Integer MVV-LVA score, or 0 when piece types cannot be determined.
    """
    attacker_info = board.get_piece_at(move.from_square)
    if attacker_info is None:
        return 0
    attacker_type, _ = attacker_info

    victim_info = board.get_piece_at(move.to_square)
    # En passant: the captured pawn is not on to_square; default to PAWN.
    victim_type = victim_info[0] if victim_info is not None else PieceType.PAWN

    return MVV_LVA[victim_type.value][attacker_type.value]
