"""RandomBeam search — no lookahead, capture-biased random selection.

Why this is ~600 ELO
--------------------
Without any lookahead the engine cannot foresee a piece being recaptured,
cannot avoid forks or pins, and cannot plan more than one half-move ahead.
It takes captures 80 % of the time (better than pure random) but among those
captures it chooses randomly, so it happily walks a queen into a defended
square.  A complete beginner who always takes when they can but otherwise
moves randomly sits roughly in this band.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

from chesspoint72.engine.core.search import Search

if TYPE_CHECKING:
    from chesspoint72.engine.core.board import Board
    from chesspoint72.engine.core.types import Move


class RandomBeamSearch(Search):
    """Pick a random capture (80 % of the time) or a random quiet move."""

    nodes_evaluated: int = 0

    def search_node(self, alpha: int, beta: int, depth: int) -> int:
        return 0

    def quiescence_search(self, alpha: int, beta: int) -> int:
        return 0

    def find_best_move(
        self,
        board: "Board",
        max_depth: int,
        allotted_time: float,
    ) -> "Move | None":
        self.nodes_evaluated = 0
        moves = board.generate_legal_moves()
        if not moves:
            return None
        self.nodes_evaluated = len(moves)

        captures = [m for m in moves if m.is_capture]
        if captures and random.random() < 0.80:
            return random.choice(captures)
        return random.choice(moves)
