from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from chesspoint72.engine.evaluator import Evaluator
from chesspoint72.engine.policies import MoveOrderingPolicy, PruningPolicy
from chesspoint72.engine.transposition import TranspositionTable
from chesspoint72.engine.types import Move

if TYPE_CHECKING:
    from chesspoint72.engine.board import Board


class Search(ABC):
    """Base class for tree-search algorithms.

    Concrete subclasses choose the actual algorithm (alpha-beta, MTD-f, etc.)
    and receive pluggable heuristics via the Strategy Pattern rather than
    through inheritance.
    """

    evaluator_reference: Evaluator
    transposition_table_reference: TranspositionTable
    move_ordering_policy: MoveOrderingPolicy
    pruning_policy: PruningPolicy
    nodes_evaluated: int
    time_limit: float

    def __init__(
        self,
        evaluator: Evaluator,
        transposition_table: TranspositionTable,
        move_ordering_policy: MoveOrderingPolicy,
        pruning_policy: PruningPolicy,
    ) -> None:
        self.evaluator_reference = evaluator
        self.transposition_table_reference = transposition_table
        self.move_ordering_policy = move_ordering_policy
        self.pruning_policy = pruning_policy
        self.nodes_evaluated = 0
        self.time_limit = 0.0

    @abstractmethod
    def find_best_move(
        self,
        board: Board,
        max_depth: int,
        allotted_time: float,
    ) -> Move: ...

    @abstractmethod
    def search_node(self, alpha: int, beta: int, depth: int) -> int: ...

    @abstractmethod
    def quiescence_search(self, alpha: int, beta: int) -> int: ...
