"""Minal v1 engine — assembles all components into a UCI controller.

Design choices vs. Frank v3:
- MinalV1MoveOrderingPolicy: SEE-gated capture splitting + development heuristic.
- Aggressive pruning config: earlier LMR (depth >= 2, move index >= 2) and
  tighter futility margin (250 cp instead of 300 cp).
- Default depth 6 (one ply deeper than Frank v3's 5).
- Engine identity: "Minal v1" / "Minal Sabir".
"""
from __future__ import annotations

import sys
from typing import Iterable, TextIO

from chesspoint72.aiengines.minal.v1.ordering import MinalV1MoveOrderingPolicy
from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.transposition import TranspositionTable
from chesspoint72.engine.factory import StandardUciController, build_evaluator
from chesspoint72.engine.pruning import ForwardPruningPolicy, PruningConfig
from chesspoint72.engine.search.negamax import NegamaxSearch


def _minal_v1_pruning_config() -> PruningConfig:
    """Pruning config tuned for Minal v1.

    Key differences from default:
    - futility_margin: 250 cp  (tighter — prunes more at depth 1)
    - lmr_min_depth: 2         (LMR fires one ply earlier than default 3)
    - lmr_min_move_index: 2    (reduces more moves per node)
    - razoring_margins: (300, 400, 500)  (tighter than default 350/450/550)
    """
    return PruningConfig(
        nmp_enabled=True,
        futility_enabled=True,
        razoring_enabled=True,
        lmr_enabled=True,
        nmp_r_shallow=2,
        nmp_r_deep=3,
        futility_margin=250,
        razoring_margins=(300, 400, 500),
        lmr_min_depth=2,
        lmr_min_move_index=2,
    )


class MinalV1UciController(StandardUciController):
    """StandardUciController with Minal v1 identity."""

    engine_name = "Minal v1"
    engine_author = "Minal Sabir"


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    *,
    hce_modules: str | None = None,
    default_depth: int = 6,
    default_time: float = 0.5,
) -> MinalV1UciController:
    """Assemble Minal v1 as a UCI controller."""
    modules = hce_modules or "classic,advanced"
    evaluator = build_evaluator("hce", modules)
    board = PyChessBoard()

    pruning_config = _minal_v1_pruning_config()
    pruning_policy = ForwardPruningPolicy(pruning_config)
    search = NegamaxSearch(
        evaluator,
        TranspositionTable(),
        MinalV1MoveOrderingPolicy(),
        pruning_policy,
        pruning_config,
    )

    return MinalV1UciController(
        board=board,
        search=search,
        input_stream=input_stream,
        output_stream=output_stream,
        evaluator=evaluator,
        default_depth=default_depth,
        default_time=default_time,
    )


def _parse_cli(argv: list[str]) -> tuple[str | None, int, float]:
    hce_modules: str | None = None
    depth = 6
    move_time = 0.5

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--hce-modules" and i + 1 < len(argv):
            hce_modules = argv[i + 1]; i += 2
        elif arg.startswith("--hce-modules="):
            hce_modules = arg.split("=", 1)[1]; i += 1
        elif arg == "--depth" and i + 1 < len(argv):
            depth = max(int(argv[i + 1]), 1); i += 2
        elif arg.startswith("--depth="):
            depth = max(int(arg.split("=", 1)[1]), 1); i += 1
        elif arg == "--time" and i + 1 < len(argv):
            move_time = max(float(argv[i + 1]), 0.05); i += 2
        elif arg.startswith("--time="):
            move_time = max(float(arg.split("=", 1)[1]), 0.05); i += 1
        else:
            i += 1

    return hce_modules, depth, move_time


def main() -> int:
    hce_modules, depth, move_time = _parse_cli(sys.argv[1:])
    controller = build_controller(
        hce_modules=hce_modules,
        default_depth=depth,
        default_time=move_time,
    )
    try:
        controller.start_listening_loop()
    except KeyboardInterrupt:
        return 130
    return 0
