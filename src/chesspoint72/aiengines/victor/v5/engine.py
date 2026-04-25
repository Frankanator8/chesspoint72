"""Victor v5 — Tactical (~1500 ELO).

Design rationale
----------------
Depth-4 iterative-deepening alpha-beta with the full classical engine stack:
HCE classic (7 modules), MoveSorter ordering (TT + MVV-LVA + killers +
history), depth-preferred TT, and all four forward pruning techniques
(NMP, LMR, razoring, futility).  This matches the published BASELINE_B
configuration from the eval pipeline.

What this engine does well
--------------------------
* Full HCE classic: material, PST, pawn structure, king safety, mobility,
  rooks, bishops.
* Pruning lets it reach effective depth 5-6 in 1 s.
* Killer moves + history ordering finds beta-cutoffs quickly.
* Quiescence search resolves all capture chains.

What this engine does poorly
-----------------------------
* HCE misses subtle positional patterns that NNUE would catch.
* No aspiration windows: root search always uses a full [-INF, INF] window.
* MoveSorter lacks continuation history and SEE-filtered captures (MovePicker).

ELO gap to v6
-------------
Adding aspiration windows and upgrading to the full Stockfish-style
MovePicker (SEE, continuation history, butterfly history) is worth ~150 ELO
at this depth.  Adding the remaining 7 HCE modules is worth another ~50 ELO.
"""
from __future__ import annotations

from typing import Iterable, TextIO

from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.transposition import DepthPreferredTT
from chesspoint72.engine.factory import (
    StandardUciController,
    MoveSorterPolicy,
    build_evaluator,
)
from chesspoint72.engine.ordering import KillerMoveTable, HistoryTable
from chesspoint72.engine.pruning import ForwardPruningPolicy, default_pruning_config
from chesspoint72.engine.search.negamax import NegamaxSearch


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    default_depth: int = 4,
    default_time: float = 1.0,
) -> StandardUciController:
    evaluator = build_evaluator("hce", "classic")
    board = PyChessBoard()

    pruning_config = default_pruning_config()
    pruning_policy = ForwardPruningPolicy(pruning_config)

    killer_table = KillerMoveTable()
    history_table = HistoryTable()
    ordering_policy = MoveSorterPolicy(killer_table, history_table)

    search = NegamaxSearch(
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
