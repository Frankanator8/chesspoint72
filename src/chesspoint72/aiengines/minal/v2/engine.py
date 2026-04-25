"""Minal v2 engine — aspiration windows + check extensions + killer/history ordering.

What's new vs v1
----------------
Search:
  - Aspiration windows: ID loop uses ±50 cp window around prev score; widens
    exponentially on fail-high/low before accepting.
  - Check extension: when depth reaches 0 while the side to move is in check,
    extend 1 ply to resolve the check before quiescing.

Move ordering:
  - Killer moves at their own tier, above all generic quiet moves.
  - History scores from the search's butterfly table influence quiet-move rank.

Pruning:
  - Slightly less aggressive LMR (min_depth=3, min_move=3) so the extra
    check-extension depth is not immediately reduced away.
"""
from __future__ import annotations

import sys
from typing import Iterable, TextIO

from chesspoint72.aiengines.minal.v2.ordering import MinalV2MoveOrderingPolicy
from chesspoint72.aiengines.minal.v2.search import MinalV2Search
from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.transposition import TranspositionTable
from chesspoint72.engine.factory import StandardUciController, build_evaluator
from chesspoint72.engine.pruning import ForwardPruningPolicy, PruningConfig


def _minal_v2_pruning_config() -> PruningConfig:
    """Pruning config for Minal v2.

    Compared to v1:
    - lmr_min_depth: 3 (back to default — check extensions need those plies)
    - lmr_min_move_index: 3 (back to default)
    - futility_margin: 275 (slightly relaxed vs v1's 250 for same reason)
    - razoring_margins: (325, 425, 525) (tighter than default, looser than v1)
    """
    return PruningConfig(
        nmp_enabled=True,
        futility_enabled=True,
        razoring_enabled=True,
        lmr_enabled=True,
        nmp_r_shallow=2,
        nmp_r_deep=3,
        futility_margin=275,
        razoring_margins=(325, 425, 525),
        lmr_min_depth=3,
        lmr_min_move_index=3,
    )


class MinalV2UciController(StandardUciController):
    engine_name = "Minal v2"
    engine_author = "Minal Sabir"


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    *,
    hce_modules: str | None = None,
    default_depth: int = 6,
    default_time: float = 0.5,
) -> MinalV2UciController:
    """Assemble Minal v2 as a UCI controller."""
    modules = hce_modules or "classic,advanced"
    evaluator = build_evaluator("hce", modules)
    board = PyChessBoard()

    pruning_config = _minal_v2_pruning_config()
    pruning_policy = ForwardPruningPolicy(pruning_config)
    ordering = MinalV2MoveOrderingPolicy()
    search = MinalV2Search(
        evaluator,
        TranspositionTable(),
        ordering,
        pruning_policy,
        pruning_config,
    )

    return MinalV2UciController(
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
