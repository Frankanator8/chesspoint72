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
try:  # NNUE pulls in torch; skip the re-export when torch isn't installed.
    from chesspoint72.engine.evaluators.nnue import (
        NnueEvaluator,
        NnueNetwork,
        fen_to_tensor,
    )
except ImportError:
    NnueEvaluator = None  # type: ignore[assignment]
    NnueNetwork = None  # type: ignore[assignment]
    fen_to_tensor = None  # type: ignore[assignment]
from chesspoint72.engine.ordering import (
    HistoryTable,
    KillerMoveTable,
    MVV_LVA,
    MoveSorter,
    score_capture,
)
from chesspoint72.engine.pruning import (
    ForwardPruningPolicy,
    PruningConfig,
    default_pruning_config,
    disable_futility,
    disable_lmr,
    disable_nmp,
    disable_razoring,
)
from chesspoint72.engine.search.negamax import NegamaxSearch
from chesspoint72.engine.uci import UciController

__all__ = [
    "Board",
    "CastlingRights",
    "Color",
    "Evaluator",
    "ForwardPruningPolicy",
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
    "PruningConfig",
    "PruningPolicy",
    "Search",
    "TranspositionEntry",
    "TranspositionTable",
    "UciController",
    "default_pruning_config",
    "disable_futility",
    "disable_lmr",
    "disable_nmp",
    "disable_razoring",
    "fen_to_tensor",
    "score_capture",
    "square_to_algebraic",
]
