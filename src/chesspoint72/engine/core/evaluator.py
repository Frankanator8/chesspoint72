from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chesspoint72.engine.core.board import Board
    from chesspoint72.engine.core.types import Move


class Evaluator(ABC):
    """Base class for static position evaluators (HCE or NNUE)."""

    @abstractmethod
    def evaluate_position(self, board: Board) -> int: ...

    def update_accumulator(self, move: Move) -> None:
        # NNUE evaluators override this; HCE leaves it as a no-op.
        return None
