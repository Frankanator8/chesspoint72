"""Victor v6 — Strategic (~1650 ELO).

Design rationale
----------------
Depth-5 AspirationNegamaxSearch with the complete HCE suite (all 14 modules),
the full Stockfish 16-style MovePicker ordering (SEE, butterfly history,
capture history, continuation history), and a depth-preferred TT.

Key upgrades over v5
--------------------
1. Aspiration windows: root search starts with a ±50 cp window, narrowing
   the tree dramatically on most moves.  This buys roughly +0.5 depth for free.

2. MovePicker replaces MoveSorter: SEE-filtered captures, continuation history
   wiring, and 13-stage tiered iteration.  Better ordering = more cutoffs =
   effectively deeper search.

3. HCE all (14 modules) adds: EWPM (endgame weak-pawn mobility), SRCM (space),
   IDAM (imbalance delta), OTVM (open-file threats), LMDM (last-move delta),
   LSCM (long-square colour), CLCM (colour-complex), DESM (diagonal escape).
   Together these add ~50 ELO of positional accuracy.

What this engine does poorly
-----------------------------
* Still HCE — misses tactical patterns and positional subtleties that a trained
  NNUE captures implicitly.
* Python speed ceiling: at 1 s/move it reaches depth 6-7, still short of what
  a C++ engine at the same ELO would achieve.

ELO gap to v7
-------------
Replacing HCE with NNUE (v7) is worth +200-300 ELO at comparable depth
because the neural evaluation captures piece interactions that are very hard
to hand-code (e.g., colour-complex weaknesses, piece harmony, king shelter).
"""
from __future__ import annotations

from typing import Iterable, TextIO

from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.transposition import DepthPreferredTT
from chesspoint72.engine.factory import StandardUciController, build_evaluator
from chesspoint72.engine.ordering import KillerMoveTable, HistoryTable
from chesspoint72.engine.ordering.picker_policy import MovePickerPolicy
from chesspoint72.engine.pruning import ForwardPruningPolicy, default_pruning_config
from chesspoint72.engine.search.negamax.aspiration import AspirationNegamaxSearch


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    default_depth: int = 5,
    default_time: float = 1.0,
) -> StandardUciController:
    evaluator = build_evaluator("hce", "all")
    board = PyChessBoard()

    pruning_config = default_pruning_config()
    pruning_policy = ForwardPruningPolicy(pruning_config)

    ordering_policy = MovePickerPolicy()
    killer_table = KillerMoveTable()
    history_table = HistoryTable()

    search = AspirationNegamaxSearch(
        evaluator,
        DepthPreferredTT(),
        ordering_policy,
        pruning_policy,
        pruning_config,
    )
    search.killer_table = killer_table
    search.history_table = history_table

    return StandardUciController(
        board=board,
        search=search,
        input_stream=input_stream,
        output_stream=output_stream,
        evaluator=evaluator,
        default_depth=default_depth,
        default_time=default_time,
    )
