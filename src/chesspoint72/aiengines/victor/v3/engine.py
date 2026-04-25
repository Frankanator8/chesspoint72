"""Victor v3 — Shallow (~1000 ELO).

Design rationale
----------------
Depth-2 alpha-beta negamax with pure material evaluation.  No forward
pruning (NMP / LMR / razoring / futility all disabled), no move ordering,
no transposition table.  The engine evaluates all 2-ply subtrees exhaustively.

What this engine does well
--------------------------
* Sees all simple captures and immediate recaptures.
* Avoids 1-move blunders reliably.
* Will spot a free piece two half-moves away.

What this engine does poorly
-----------------------------
* Misses anything that requires 3+ half-moves to set up.
* No positional understanding — will misplace pieces if material is equal.
* No ordering means the search is slow and the effective depth is low.
* No TT means identical positions are re-evaluated.

ELO gap to v4
-------------
Adding PST-based positional evaluation (v4) closes the gap between a purely
tactical engine and one that understands development and piece activity.
"""
from __future__ import annotations

from typing import Iterable, TextIO

from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.transposition import NullTranspositionTable
from chesspoint72.engine.factory import (
    StandardUciController,
    _StubPruningPolicy,
    _StubMoveOrderingPolicy,
    build_evaluator,
)
from chesspoint72.engine.search.negamax import NegamaxSearch


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    default_depth: int = 2,
    default_time: float = 1.0,
) -> StandardUciController:
    evaluator = build_evaluator("material")
    board = PyChessBoard()
    pruning_policy = _StubPruningPolicy()
    search = NegamaxSearch(
        evaluator,
        NullTranspositionTable(),   # no caching
        _StubMoveOrderingPolicy(),  # no ordering
        pruning_policy,
        pruning_config=None,        # all pruning off
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
