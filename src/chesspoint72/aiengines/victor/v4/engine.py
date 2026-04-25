"""Victor v4 — Positional (~1200 ELO).

Design rationale
----------------
Depth-3 alpha-beta with HCE material + PST evaluation, TT, and basic
MVV-LVA + history move ordering.  Forward pruning is disabled (stub policy)
so the tree is searched more honestly but slowly — at depth 3 the lack of
pruning is still manageable.

What this engine does well
--------------------------
* PST bonuses mean it develops pieces, controls the centre, and castles.
* Depth 3 catches most simple tactical motifs (forks, skewers, 1-move threats).
* TT avoids re-evaluating transposed positions.
* History ordering improves move selection within the search.

What this engine does poorly
-----------------------------
* No null-move pruning / LMR means it cannot reach depth 5+ in time.
* Misses deeper combinations (depth 4+ required).
* King safety, mobility, and pawn structure not yet evaluated.

ELO gap to v5
-------------
Enabling the full pruning suite (NMP + LMR + futility + razoring) in v5
effectively doubles the depth the engine can search in the same time budget,
and adding HCE classic (7 modules) adds king safety, mobility, and rook
bonuses.  Together these are worth ~300 ELO.
"""
from __future__ import annotations

from typing import Iterable, TextIO

from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.transposition import TranspositionTable
from chesspoint72.engine.factory import (
    StandardUciController,
    _StubPruningPolicy,
    MoveSorterPolicy,
    build_evaluator,
)
from chesspoint72.engine.ordering import KillerMoveTable, HistoryTable
from chesspoint72.engine.search.negamax import NegamaxSearch


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    default_depth: int = 3,
    default_time: float = 1.0,
) -> StandardUciController:
    # Material + piece-square tables only — no king safety, mobility, etc.
    evaluator = build_evaluator("hce", "material,pst")
    board = PyChessBoard()

    killer_table = KillerMoveTable()
    history_table = HistoryTable()
    ordering_policy = MoveSorterPolicy(killer_table, history_table)

    search = NegamaxSearch(
        evaluator,
        TranspositionTable(),
        ordering_policy,
        _StubPruningPolicy(),   # no forward pruning
        pruning_config=None,
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
