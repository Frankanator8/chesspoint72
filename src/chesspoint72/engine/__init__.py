"""Chesspoint72 chess engine package.

Public surface re-exports the core interfaces (the ABCs every implementation
plugs into) plus the headline registered implementations. Concrete modules
live in subpackages: ``core``, ``evaluators``, ``search``, ``ordering``,
``uci``. The ``factory`` module is the single dispatch point for selecting
implementations at runtime.
"""

from chesspoint72.engine.core import (
    Board,
    CastlingRights,
    Color,
    Evaluator,
    Move,
    MoveOrderingPolicy,
    NodeType,
    PieceType,
    PruningPolicy,
    Search,
    TranspositionEntry,
    TranspositionTable,
    square_to_algebraic,
)
from chesspoint72.engine.evaluators.nnue import (
    NnueEvaluator,
    NnueNetwork,
    fen_to_tensor,
)
from chesspoint72.engine.ordering import (
    HistoryTable,
    KillerMoveTable,
    MVV_LVA,
    MoveSorter,
    score_capture,
)
from chesspoint72.engine.search.negamax import NegamaxSearch
from chesspoint72.engine.uci import UciController

__all__ = [
    "Board",
    "CastlingRights",
    "Color",
    "Evaluator",
    "HistoryTable",
    "KillerMoveTable",
    "MVV_LVA",
    "Move",
    "MoveOrderingPolicy",
    "MoveSorter",
    "NegamaxSearch",
    "NnueEvaluator",
    "NnueNetwork",
    "NodeType",
    "PieceType",
    "PruningPolicy",
    "Search",
    "TranspositionEntry",
    "TranspositionTable",
    "UciController",
    "fen_to_tensor",
    "score_capture",
    "square_to_algebraic",
]
