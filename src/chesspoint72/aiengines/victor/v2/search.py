"""OnePly search — 1-ply minimax with material evaluation.

Why this is ~800 ELO
--------------------
One full ply of look-ahead means the engine sees every direct capture and
recapture: it will not leave a piece en prise for a single move.  But it is
completely blind to two-move combinations: forks, discovered attacks, skewers,
and anything that requires two sequential moves to set up.  It also has no
positional understanding — it will happily double pawns or trade a bishop for
a knight in the opening if the immediate material balance stays the same.
A beginner who avoids hanging pieces but misses all tactics is in this tier.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from chesspoint72.engine.core.search import Search

if TYPE_CHECKING:
    from chesspoint72.engine.core.board import Board
    from chesspoint72.engine.core.types import Move

_INF = 10_000_000


class OnePlySearch(Search):
    """Full 1-ply minimax: evaluate every legal response and pick the best."""

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

        evaluate = self.evaluator_reference.evaluate_position
        best_score = -_INF
        best_move = moves[0]

        for move in moves:
            board.make_move(move)
            self.nodes_evaluated += 1
            # Negate: opponent's good position = our bad position.
            score = -evaluate(board)
            board.unmake_move()
            if score > best_score:
                best_score = score
                best_move = move

        return best_move
