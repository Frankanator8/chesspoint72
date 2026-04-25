from .hce import evaluate, explain, get_game_phase
from .material import (
    BISHOP_PAIR_BONUS,
    KNIGHT_PAIR_PENALTY,
    PIECE_VALUES,
    ROOK_PAIR_PENALTY,
    material_score,
)
from .pst import pst_score

__all__ = [
    # unified evaluator
    "evaluate",
    "explain",
    "get_game_phase",
    # legacy standalone helpers
    "material_score",
    "pst_score",
    "PIECE_VALUES",
    "BISHOP_PAIR_BONUS",
    "KNIGHT_PAIR_PENALTY",
    "ROOK_PAIR_PENALTY",
]
