"""Static material evaluation.

Provides a simple centipawn material balance with piece-pair adjustments
(bishop pair bonus, knight/rook pair penalties).  This module is used
directly by legacy callers; the unified HCE in hce.py supersedes it for
full evaluations.

Public API
----------
material_score(board) -> int
    Centipawns, positive = good for White.
"""
# @capability: evaluator
# @capability: material
from __future__ import annotations

import chess

PIECE_VALUES: dict[int, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

# Pair adjustments (centipawns, White's perspective)
BISHOP_PAIR_BONUS = 50    # two bishops cover both square colors
KNIGHT_PAIR_PENALTY = -10  # both short-range; combined coverage is poor
ROOK_PAIR_PENALTY = -15    # two rooks slightly redundant vs rook+bishop


def material_score(board: chess.Board) -> int:
    """Return material balance in centipawns from White's perspective."""
    score = 0

    for piece_type, value in PIECE_VALUES.items():
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        score += (white_count - black_count) * value

    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        bishops = len(board.pieces(chess.BISHOP, color))
        knights = len(board.pieces(chess.KNIGHT, color))
        rooks = len(board.pieces(chess.ROOK, color))

        if bishops >= 2:
            score += sign * BISHOP_PAIR_BONUS
        if knights >= 2:
            score += sign * KNIGHT_PAIR_PENALTY
        if rooks >= 2:
            score += sign * ROOK_PAIR_PENALTY

    return score
