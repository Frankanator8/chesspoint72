from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chesspoint72.engine.core.board import Board
    from chesspoint72.engine.core.types import Move


class MoveOrderingPolicy(ABC):
    """Strategy for ranking candidate moves before the search loop iterates them.

    Implementations may use any combination of static heuristics (MVV-LVA,
    piece-square tables) and dynamic history (killer moves, history heuristic).
    The optional *tt_best_move* hint should be placed first when provided —
    searching the transposition-table suggestion before all other moves is the
    single highest-value ordering decision available to the engine.
    """

    @abstractmethod
    def order_moves(
        self,
        moves: list[Move],
        board: Board,
        tt_best_move: Move | None = None,
    ) -> list[Move]:
        """Return *moves* sorted from most-promising to least-promising.

        Args:
            moves: Unordered list of legal (or pseudo-legal) moves.
            board: Current position, used for piece-type lookups (MVV-LVA, etc.).
            tt_best_move: Best move stored in the transposition table for this
                position; should be searched first if present.

        Returns:
            A new list containing the same moves in descending priority order.
        """


class PruningPolicy(ABC):
    """Strategy for forward-pruning nodes that are unlikely to affect the result.

    A single ``try_prune`` entry-point covers both the *can-we-prune* decision
    and the score to return when pruning is applied.  Returning ``None`` means
    "do not prune; continue with normal search."  Returning an ``int`` means
    "prune this node; use this value as the node's score."

    Concrete implementations include:

    * **Futility Pruning** — at shallow depths, skip nodes where
      ``static_eval + margin < alpha`` (the position is so far below the
      lower bound that no single move is likely to recover it).
    * **Null-Move Pruning** — skip a move (give the opponent a free turn) and
      search at reduced depth.  If that score still beats beta, the position is
      so strong that a full search is unnecessary.
    """

    @abstractmethod
    def try_prune(
        self,
        board: Board,
        depth: int,
        alpha: int,
        beta: int,
        static_eval: int,
    ) -> int | None:
        """Attempt to prune the current node.

        Args:
            board: Current position.
            depth: Remaining depth to search (plies left, not plies from root).
            alpha: Current lower bound of the search window.
            beta: Current upper bound of the search window.
            static_eval: Static evaluation of *board* from the perspective of
                the side to move (centipawns, positive = better for side to move).

        Returns:
            An integer score to return immediately (pruning fires), or ``None``
            to continue with the normal search (pruning does not fire).
        """
