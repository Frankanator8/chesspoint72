from __future__ import annotations

from chesspoint72.engine.core.board import Board
from chesspoint72.engine.core.evaluator import Evaluator
from chesspoint72.engine.factory import _HceEvaluator

# Seven classic terms + CLCM (the only advanced module whose chess content is
# not already captured by mobility/pawns and whose cost is bounded).
_FRANK_HCE_MODULES = "material,pst,pawns,king_safety,mobility,rooks,bishops,clcm"


class FrankEvaluator(Evaluator):
    def __init__(self) -> None:
        self._inner = _HceEvaluator(_FRANK_HCE_MODULES)

    def evaluate_position(self, board: Board) -> int:
        return self._inner.evaluate_position(board)
