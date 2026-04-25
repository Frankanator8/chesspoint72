"""engine_hce.py — Victor's staged Hand-Crafted Evaluation engine.

Uses all HCE modules (classic: material, pst, pawns, king_safety, mobility,
rooks, bishops; advanced: ewpm, srcm, idam, otvm, lmdm, lscm, clcm, desm)
via the shared _HceEvaluator from the engine factory.

CLI:
    python -m chesspoint72.aiengines.victor.engine_hce.engine_hce
"""
from __future__ import annotations

import sys
from typing import Iterable, TextIO

from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.transposition import TranspositionTable
from chesspoint72.engine.factory import StandardUciController, build_evaluator
from chesspoint72.engine.pruning import ForwardPruningPolicy, default_pruning_config
from chesspoint72.engine.search.negamax import NegamaxSearch
from chesspoint72.engine.core.policies import MoveOrderingPolicy
from chesspoint72.engine.core.types import Move


class _PassthroughOrdering(MoveOrderingPolicy):
    def order_moves(
        self,
        moves: list[Move],
        board,
        tt_best_move: Move | None = None,
    ) -> list[Move]:
        return moves


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    default_depth: int = 5,
    default_time: float = 5.0,
) -> StandardUciController:
    evaluator = build_evaluator("hce", hce_modules="all")
    board = PyChessBoard()
    cfg = default_pruning_config()
    pruning_policy = ForwardPruningPolicy(cfg)
    search = NegamaxSearch(
        evaluator,
        TranspositionTable(),
        _PassthroughOrdering(),
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
    ctrl.engine_name = "Victor-HCE"
    ctrl.engine_author = "Victor"
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
