from __future__ import annotations

import os
import sys
from typing import Iterable, TextIO

from chesspoint72.aiengines.frank.v3.ordering import FrankV3MoveOrderingPolicy
from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.evaluator import Evaluator
from chesspoint72.engine.core.transposition import TranspositionTable
from chesspoint72.engine.factory import StandardUciController, build_evaluator
from chesspoint72.engine.pruning import ForwardPruningPolicy, default_pruning_config
from chesspoint72.engine.search.negamax import NegamaxSearch


def _try_build_nnue_evaluator() -> Evaluator | None:
    """Return NNUE evaluator if dependencies/weights are usable, else None."""
    try:
        from chesspoint72.engine.evaluators.nnue import NnueEvaluator
    except Exception:
        return None

    weights = os.environ.get("CHESSPOINT72_NNUE_WEIGHTS")
    try:
        return NnueEvaluator(weights) if weights else NnueEvaluator()
    except Exception:
        return None


def build_frank_v3_evaluator(
    *,
    prefer_nnue: bool = True,
    hce_modules: str | None = None,
) -> Evaluator:
    """Build the strongest evaluator available in this environment.

    Preference chain:
    1) NNUE (if import and weights load succeed)
    2) HCE (default: classic + advanced module bundles)
    """
    if prefer_nnue:
        nnue = _try_build_nnue_evaluator()
        if nnue is not None:
            return nnue

    modules = hce_modules or os.environ.get("CHESSPOINT72_FRANK_HCE_MODULES", "classic,advanced")
    return build_evaluator("hce", modules)


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    *,
    hce_modules: str | None = None,
    default_depth: int = 5,
    default_time: float = 0.5,
    prefer_nnue: bool = True,
) -> StandardUciController:
    """Assemble Frank v3 as a UCI controller."""
    evaluator = build_frank_v3_evaluator(prefer_nnue=prefer_nnue, hce_modules=hce_modules)
    board = PyChessBoard()

    pruning_config = default_pruning_config()
    pruning_policy = ForwardPruningPolicy(pruning_config)
    search = NegamaxSearch(
        evaluator,
        TranspositionTable(),
        FrankV3MoveOrderingPolicy(),
        pruning_policy,
        pruning_config,
    )

    return StandardUciController(
        board=board,
        search=search,
        input_stream=input_stream,
        output_stream=output_stream,
        evaluator=evaluator,
        default_depth=default_depth,
        default_time=default_time,
    )


def _parse_cli(argv: list[str]) -> tuple[str | None, int, float, bool]:
    hce_modules: str | None = None
    depth = 5
    move_time = 0.5
    prefer_nnue = True

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--hce-modules" and i + 1 < len(argv):
            hce_modules = argv[i + 1]
            i += 2
        elif arg.startswith("--hce-modules="):
            hce_modules = arg.split("=", 1)[1]
            i += 1
        elif arg == "--depth" and i + 1 < len(argv):
            depth = max(int(argv[i + 1]), 1)
            i += 2
        elif arg.startswith("--depth="):
            depth = max(int(arg.split("=", 1)[1]), 1)
            i += 1
        elif arg == "--time" and i + 1 < len(argv):
            move_time = max(float(argv[i + 1]), 0.05)
            i += 2
        elif arg.startswith("--time="):
            move_time = max(float(arg.split("=", 1)[1]), 0.05)
            i += 1
        elif arg == "--no-nnue":
            prefer_nnue = False
            i += 1
        else:
            i += 1

    return hce_modules, depth, move_time, prefer_nnue


def main() -> int:
    hce_modules, depth, move_time, prefer_nnue = _parse_cli(sys.argv[1:])
    controller = build_controller(
        hce_modules=hce_modules,
        default_depth=depth,
        default_time=move_time,
        prefer_nnue=prefer_nnue,
    )
    try:
        controller.start_listening_loop()
    except KeyboardInterrupt:
        return 130
    return 0

