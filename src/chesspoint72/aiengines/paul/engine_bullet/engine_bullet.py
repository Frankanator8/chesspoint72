"""engine_bullet.py â€” Paul's Bullet engine.

Pairs the nnue_speedster (64x16, blitz/NPS specialist) with a search tuned
for maximum nodes-per-second throughput. The speedster network is the smallest
in the suite â€” its tiny forward pass makes it the fastest to evaluate. The
pruning config amplifies this by cutting aggressively:

  - Razoring disabled: razoring triggers a quiescence search call per node
    at depths 2-4, which adds latency without meaningful gain at shallow depth.
  - Aggressive LMR: starts from the 2nd move at depth 2, reducing more of the
    tree and enabling the engine to reach depth 4-5 within tight time limits.
  - Aggressive NMP: R=3 even at shallow depths to prune calm branches fast.
  - Higher futility margin: quiet moves 400cp below alpha are pruned at depth 1.

Default depth 4 keeps individual search calls fast; the engine compensates with
raw speed from the small network.

CLI:
    python -m chesspoint72.aiengines.paul.engine_bullet
"""
from __future__ import annotations

import sys
from typing import Iterable, TextIO

from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.transposition import TranspositionTable
from chesspoint72.engine.evaluators.nnue import NnueEvaluator
from chesspoint72.engine.factory import StandardUciController
from chesspoint72.engine.pruning import ForwardPruningPolicy
from chesspoint72.engine.pruning.config import PruningConfig
from chesspoint72.engine.search.negamax import NegamaxSearch

from .._common import WEIGHTS_DIR, PassthroughOrdering

_WEIGHTS = WEIGHTS_DIR / "real_nnue_epoch_2.pt"


def bullet_pruning_config() -> PruningConfig:
    return PruningConfig(
        nmp_enabled=True,
        futility_enabled=True,
        razoring_enabled=False,  # disabled: saves QS overhead at depths 2-4
        lmr_enabled=True,
        nmp_r_shallow=3,         # aggressive: prune calm branches hard
        nmp_r_deep=4,
        futility_margin=400,     # wider: prune more quiet moves at depth 1
        razoring_margins=(350, 450, 550),  # unused (razoring disabled)
        lmr_min_depth=2,         # start LMR earlier than default (3)
        lmr_min_move_index=1,    # apply LMR from 2nd move onward (default 3)
    )


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    default_depth: int = 4,
    default_time: float = 1.0,
) -> StandardUciController:
    evaluator = NnueEvaluator(_WEIGHTS)
    board = PyChessBoard()
    cfg = bullet_pruning_config()
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
    ctrl.engine_name = "Paul-Bullet"
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
