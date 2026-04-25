"""Frank v3 engine profile.

This package provides a self-contained UCI entrypoint that assembles the
strongest currently available stack in this repository:
- evaluator: NNUE when available, otherwise HCE fallback
- search: Negamax with TT + forward pruning
- ordering: Frank v3 move ordering policy (TT-first + tactical sorting)
"""

from chesspoint72.aiengines.frank.v3.engine import (
    FrankV3MoveOrderingPolicy,
    build_controller,
    build_frank_v3_evaluator,
    main,
)

__all__ = [
    "FrankV3MoveOrderingPolicy",
    "build_controller",
    "build_frank_v3_evaluator",
    "main",
]

