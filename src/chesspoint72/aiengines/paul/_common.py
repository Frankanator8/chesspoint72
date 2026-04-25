"""Shared utilities for Paul's engine implementations."""
from __future__ import annotations

from pathlib import Path

from chesspoint72.engine.core.policies import MoveOrderingPolicy
from chesspoint72.engine.core.types import Move

# Resolved path to the shared NNUE weight directory.
WEIGHTS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "engine" / "evaluators" / "nnue" / "weights"
)


class PassthroughOrdering(MoveOrderingPolicy):
    """No-op move ordering — returns moves in generation order."""

    def order_moves(
        self,
        moves: list[Move],
        board,
        tt_best_move: Move | None = None,
    ) -> list[Move]:
        return moves
