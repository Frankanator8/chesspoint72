"""Victor v2 — OnePly (~800 ELO).

Design rationale
----------------
1-ply full minimax with a pure material evaluator.  For every legal move the
engine makes the move, evaluates the resulting material balance from the
opponent's perspective (negated), and picks the move that minimises the
opponent's advantage.

What this engine does well
--------------------------
* Never leaves a piece en prise for a single move.
* Will take any undefended piece.

What this engine does poorly
-----------------------------
* Cannot see forks, discovered checks, or any 2-move combination.
* No positional understanding: will double pawns, misplace pieces, etc.
* Ignores king safety completely.

ELO gap to v3
-------------
Extending to depth 2 (v3) catches all simple 2-move tactical motifs
(immediate recaptures, one-move threats) and is the second-largest jump
in the suite.
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
from chesspoint72.aiengines.victor.v2.search import OnePlySearch


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    default_depth: int = 1,
    default_time: float = 0.5,
) -> StandardUciController:
    evaluator = build_evaluator("material")
    board = PyChessBoard()
    search = OnePlySearch(
        evaluator,
        NullTranspositionTable(),
        _StubMoveOrderingPolicy(),
        _StubPruningPolicy(),
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
