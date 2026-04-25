"""engine_classic.py — Paul's Classic (control/baseline) engine.

Uses the original nnue_weights.pt (256x32 generalist) with standard
NegamaxSearch and the default pruning configuration. This is the reference
engine all others are benchmarked against.

CLI:
    python -m chesspoint72.aiengines.paul.engine_classic
"""
from __future__ import annotations

import sys
from typing import Iterable, TextIO

from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.transposition import TranspositionTable
from chesspoint72.engine.evaluators.nnue import NnueEvaluator
from chesspoint72.engine.factory import StandardUciController
from chesspoint72.engine.pruning import ForwardPruningPolicy, default_pruning_config
from chesspoint72.engine.search.negamax import NegamaxSearch

from ._common import WEIGHTS_DIR, PassthroughOrdering

_WEIGHTS = WEIGHTS_DIR / "nnue_weights.pt"


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    default_depth: int = 5,
    default_time: float = 5.0,
) -> StandardUciController:
    evaluator = NnueEvaluator(_WEIGHTS)
    board = PyChessBoard()
    cfg = default_pruning_config()
    pruning_policy = ForwardPruningPolicy(cfg)
    search = NegamaxSearch(
        evaluator,
        TranspositionTable(),
        PassthroughOrdering(),
        pruning_policy,
        cfg,
    )
    ctrl = StandardUciController(
        board=board,
        search=search,
        input_stream=input_stream,
        output_stream=output_stream,
        evaluator=evaluator,
        default_depth=default_depth,
        default_time=default_time,
    )
    ctrl.engine_name = "Paul-Classic"
    return ctrl


def main() -> int:
    ctrl = build_controller()
    try:
        ctrl.start_listening_loop()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
