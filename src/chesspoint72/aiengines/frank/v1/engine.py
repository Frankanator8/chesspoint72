from __future__ import annotations

from typing import Iterable, TextIO

from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.policies import MoveOrderingPolicy
from chesspoint72.engine.core.transposition import TranspositionTable
from chesspoint72.engine.factory import StandardUciController, _StubMoveOrderingPolicy
from chesspoint72.engine.pruning import ForwardPruningPolicy, default_pruning_config
from chesspoint72.engine.search.negamax import NegamaxSearch

from chesspoint72.aiengines.frank.v1.evaluator import FrankEvaluator
from chesspoint72.aiengines.frank.v1.move_ordering import FrankMoveOrdering


def build_frank_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    default_depth: int = 4,
    default_time: float = 5.0,
) -> StandardUciController:
    evaluator = FrankEvaluator()
    board = PyChessBoard()
    pruning_config = default_pruning_config()
    pruning_policy = ForwardPruningPolicy(pruning_config)

    # NegamaxSearch instantiates its own HistoryTable in __init__; build the
    # search first, then bind a real ordering policy that shares that table.
    search = NegamaxSearch(
        evaluator,
        TranspositionTable(),
        _StubMoveOrderingPolicy(),
        pruning_policy,
        pruning_config,
    )
    ordering: MoveOrderingPolicy = FrankMoveOrdering(search.history_table)
    search.move_ordering_policy = ordering

    return StandardUciController(
        board=board,
        search=search,
        input_stream=input_stream,
        output_stream=output_stream,
        evaluator=evaluator,
        default_depth=default_depth,
        default_time=default_time,
    )
