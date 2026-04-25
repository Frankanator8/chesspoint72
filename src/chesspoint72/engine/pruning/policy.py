# @capability: pruning_policy
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from chesspoint72.engine.core.policies import PruningPolicy
from chesspoint72.engine.pruning import algorithms
from chesspoint72.engine.pruning.config import PruningConfig

if TYPE_CHECKING:
    from chesspoint72.engine.core.board import Board


class ForwardPruningPolicy(PruningPolicy):
    """Concrete PruningPolicy wiring NMP and Razoring into negamax via try_prune.

    Move-level techniques (Futility, LMR) are applied directly by
    NegamaxSearch.search_node when pruning_config is set — they require
    per-move context not available here.

    attach_search() must be called after the search is constructed so that
    NMP can recurse back into search_node / quiescence_search.
    NegamaxSearch.__init__ calls this automatically when it detects the method.
    """

    def __init__(self, config: PruningConfig) -> None:
        self.config = config
        self._search_node: Callable | None = None
        self._quiescence_search: Callable | None = None

    def attach_search(self, search) -> None:
        """Late-bind the search callbacks. Called by NegamaxSearch.__init__."""
        self._search_node = search.search_node
        self._quiescence_search = search.quiescence_search

    # ---- SearchHost duck-type (consumed by algorithms) ---- #

    def search_node(self, alpha: int, beta: int, depth: int) -> int:
        return self._search_node(alpha, beta, depth)  # type: ignore[misc]

    def quiescence_search(self, alpha: int, beta: int) -> int:
        return self._quiescence_search(alpha, beta)  # type: ignore[misc]

    # ---- PruningPolicy ABC ---- #

    def try_prune(
        self,
        board: Board,
        depth: int,
        alpha: int,
        beta: int,
        static_eval: int,
    ) -> int | None:
        """Run node-level pruning: NMP then Razoring.

        Returns None (no-op) if attach_search has not yet been called.
        """
        if self._search_node is None:
            return None

        in_check = board.is_king_in_check()

        nmp_score = algorithms.try_null_move_pruning(
            self, board, depth, beta, in_check, self.config
        )
        if nmp_score is not None:
            return nmp_score

        return algorithms.try_razoring(self, depth, alpha, static_eval, self.config)
