"""Victor v1 — RandomBeam (~600 ELO).

Design rationale
----------------
No search whatsoever.  The engine generates all legal moves and picks a
random capture 80 % of the time, otherwise a random quiet move.

What this engine does well
--------------------------
* Always plays legal moves.
* Takes hanging pieces most of the time.

What this engine does poorly
-----------------------------
* Cannot avoid forks, pins, or skewers.
* Will sacrifice its queen for a pawn if luck goes against it.
* Has no positional understanding at all.

ELO gap to v2
-------------
Adding a single ply of look-ahead (v2) is the single largest jump in the
suite: it eliminates all single-move blunders, which are the dominant source
of losses at this level.
"""
from __future__ import annotations

from typing import Iterable, TextIO

from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.transposition import NullTranspositionTable
from chesspoint72.engine.factory import StandardUciController, _StubEvaluator, _StubPruningPolicy, _StubMoveOrderingPolicy
from chesspoint72.aiengines.victor.v1.search import RandomBeamSearch


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    default_depth: int = 1,
    default_time: float = 0.1,
) -> StandardUciController:
    evaluator = _StubEvaluator()
    board = PyChessBoard()
    search = RandomBeamSearch(
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
