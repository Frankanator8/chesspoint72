"""Hand-Crafted Evaluation (HCE) package.

The primary entry point is ``evaluate(board)``, which returns a tapered
centipawn score from the side-to-move's perspective.  The legacy helpers
``material_score`` and ``pst_score`` are re-exported for callers that need
individual components.
"""
from .hce import evaluate, explain, get_game_phase
from .material import (
    BISHOP_PAIR_BONUS,
    KNIGHT_PAIR_PENALTY,
    PIECE_VALUES,
    ROOK_PAIR_PENALTY,
    material_score,
)
from .pst import pst_score
from .advanced_features import (
    EWPM, ewpm,
    SRCM, srcm,
    IDAM, idam,
    OTVM, otvm,
    LMDM, lmdm,
    LSCM, lscm,
    CLCM, clcm,
    DESM, desm,
)

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
    # advanced feature classes
    "EWPM", "SRCM", "IDAM", "OTVM", "LMDM", "LSCM", "CLCM", "DESM",
    # advanced feature singletons
    "ewpm", "srcm", "idam", "otvm", "lmdm", "lscm", "clcm", "desm",
]
