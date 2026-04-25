"""Minal v3 engine.

What's new vs v2
----------------
Search:   PVS, RFP, LMP, IID, SEE-pruned qsearch, countermove tracking.
Eval:     +15 cp tempo bonus for the side to move.
Time:     Phase-aware allocation — more time in the middlegame, less in
          the endgame where positions are simpler and deeper look-aheads
          are cheap.
"""
from __future__ import annotations

import sys
from typing import Iterable, TextIO

import chess

from chesspoint72.aiengines.minal.v3.evaluator import MinalV3Evaluator
from chesspoint72.aiengines.minal.v3.ordering import MinalV3MoveOrderingPolicy
from chesspoint72.aiengines.minal.v3.search import MinalV3Search
from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.transposition import TranspositionTable
from chesspoint72.engine.factory import StandardUciController, build_evaluator
from chesspoint72.engine.pruning import ForwardPruningPolicy, PruningConfig


# ---------------------------------------------------------------------------
# Phase-aware time divisors
# ---------------------------------------------------------------------------
# PHASE_WEIGHTS from the HCE: N=1, B=1, R=2, Q=4 → max = 24
# We coarsely bucket into three bands and adjust how much of our clock we use.
_PHASE_DIVISORS: list[tuple[int, float]] = [
    (20, 25.0),   # opening    (>= 20 phase pts) — more time, harder positions
    (10, 30.0),   # middlegame (>= 10)            — default
    ( 0, 40.0),   # endgame    (<  10)            — less time, simpler trees
]


def _phase_divisor(board: PyChessBoard) -> float:
    """Estimate game phase and return the clock divisor to use."""
    py_board = board.py_board
    phase = 0
    for pt, weight in ((chess.KNIGHT, 1), (chess.BISHOP, 1),
                       (chess.ROOK, 2), (chess.QUEEN, 4)):
        phase += weight * bin(py_board.pieces_mask(pt, chess.WHITE)
                              | py_board.pieces_mask(pt, chess.BLACK)).count("1")
    for threshold, divisor in _PHASE_DIVISORS:
        if phase >= threshold:
            return divisor
    return 40.0   # shouldn't reach here


def _minal_v3_pruning_config() -> PruningConfig:
    """Pruning config for Minal v3.

    RFP and LMP are handled inside MinalV3Search, so the PruningPolicy
    margins here govern NMP, futility, and razoring only.  LMR is kept
    at conservative defaults because PVS already handles the first-move
    re-search; we don't want to reduce critical moves too aggressively.
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
        lmr_min_move_index=4,   # PVS handles the first re-search; LMR starts later
    )


class MinalV3UciController(StandardUciController):
    engine_name = "Minal v3"
    engine_author = "Minal Sabir"

    def _parse_go(self, input_string: str) -> tuple[int, float]:
        """Phase-aware override of the clock-management logic."""
        max_depth, allotted = super()._parse_go(input_string)
        # Adjust time by game-phase divisor relative to the default of 30.
        divisor = _phase_divisor(self._board)
        if divisor != 30.0:
            # Recompute allotted time using the phase-specific divisor.
            # We scale the super()-computed value: if super used /30 and we
            # want /25, multiply by 30/25.  The increment portion is unchanged.
            allotted = allotted * (30.0 / divisor)
        return max_depth, allotted


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    *,
    hce_modules: str | None = None,
    default_depth: int = 6,
    default_time: float = 0.5,
) -> MinalV3UciController:
    """Assemble Minal v3 as a UCI controller."""
    modules = hce_modules or "classic,advanced"
    base_hce = build_evaluator("hce", modules)
    evaluator = MinalV3Evaluator(base_hce)
    board = PyChessBoard()

    pruning_config = _minal_v3_pruning_config()
    pruning_policy = ForwardPruningPolicy(pruning_config)
    ordering = MinalV3MoveOrderingPolicy()
    search = MinalV3Search(
        evaluator,
        TranspositionTable(),
        ordering,
        pruning_policy,
        pruning_config,
    )

    return MinalV3UciController(
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
