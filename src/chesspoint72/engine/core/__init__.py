"""Core interfaces and shared primitives for the chess engine.

Implementations of these ABCs live in sibling packages:
``engine.evaluators``, ``engine.search``, ``engine.ordering``.
"""

from chesspoint72.engine.core.board import Board
from chesspoint72.engine.core.evaluator import Evaluator
from chesspoint72.engine.core.policies import MoveOrderingPolicy, PruningPolicy
from chesspoint72.engine.core.search import Search
from chesspoint72.engine.core.transposition import TranspositionEntry, TranspositionTable
from chesspoint72.engine.core.types import (
    CastlingRights,
    Color,
    Move,
    NodeType,
    PieceType,
    square_to_algebraic,
)

__all__ = [
    "Board",
    "CastlingRights",
    "Color",
    "Evaluator",
    "Move",
    "MoveOrderingPolicy",
    "NodeType",
    "PieceType",
    "PruningPolicy",
    "Search",
    "TranspositionEntry",
    "TranspositionTable",
    "square_to_algebraic",
]
