"""Victor v7 — Neural (~1900 ELO).

Design rationale
----------------
GMSearch (PVS + check extensions + SEE/delta QSearch pruning + continuation
history fully wired) combined with the nnue_speedster evaluator (64-hidden-
unit NNUE, fastest inference in the suite), full Stockfish 16-style
MovePickerPolicy, and a 256 MB depth-preferred TT.

This is the strongest configuration that fits in a 1 s/move budget in
CPython.  At 5+ min/move (or in PyPy) it would approach IM strength.

Key upgrades over v6
--------------------
1. NNUE evaluator: a trained 64-hidden-unit network replaces all 14 HCE
   modules.  The network has learned positional patterns that would require
   thousands of lines of HCE code to approximate (colour-complex weaknesses,
   piece harmony, king-shelter nuances, endgame technique).

2. GMSearch: adds Principal Variation Search (null-window proof for non-PV
   moves, ~20-30 % fewer nodes), check extensions (depth +1 in check to
   prevent horizon-effect cutoffs), and SEE/delta pruning in quiescence.

3. Continuation history fully wired: GMSearch maintains a 6-ply move-context
   stack and passes it to MovePickerPolicy, enabling the full Stockfish 16
   continuation-history bonus that was dormant in v5/v6.

Absolute ceiling
----------------
At 1 s/move in CPython this engine reaches depth 5-6 with NNUE.
True GM strength (2500+ FIDE) requires C++/PyPy + multi-threading + a much
larger NNUE trained on millions of master games.
"""
from __future__ import annotations

import os
from typing import Iterable, TextIO

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.transposition import DepthPreferredTT
from chesspoint72.engine.factory import StandardUciController, build_evaluator
from chesspoint72.engine.ordering import KillerMoveTable, HistoryTable
from chesspoint72.engine.ordering.picker_policy import MovePickerPolicy
from chesspoint72.engine.pruning import ForwardPruningPolicy, default_pruning_config
from chesspoint72.engine.search.negamax.gm_search import GMSearch


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    default_depth: int = 30,
    default_time: float = 1.0,
) -> StandardUciController:
    evaluator = build_evaluator("nnue_speedster")
    board = PyChessBoard()

    pruning_config = default_pruning_config()
    pruning_policy = ForwardPruningPolicy(pruning_config)
    ordering_policy = MovePickerPolicy()

    search = GMSearch(
        evaluator,
        DepthPreferredTT(max_memory_size=256),
        ordering_policy,
        pruning_policy,
        pruning_config,
    )
    search.killer_table = KillerMoveTable()
    search.history_table = HistoryTable()

    return StandardUciController(
        board=board,
        search=search,
        input_stream=input_stream,
        output_stream=output_stream,
        evaluator=evaluator,
        default_depth=default_depth,
        default_time=default_time,
    )
