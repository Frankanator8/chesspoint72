"""Minal v3 evaluator — wraps HCE and adds a tempo bonus.

The tempo bonus (+TEMPO_CP centipawns for the side to move) captures the
initiative: having the move is worth a small but real positional advantage
because you can threaten something before the opponent can respond.

Stockfish uses a ~28 cp tempo; here we use 15 cp, a more conservative value
that still makes the engine prefer active play over passive defence.
"""
from __future__ import annotations

from chesspoint72.engine.core.board import Board
from chesspoint72.engine.core.evaluator import Evaluator

TEMPO_CP: int = 15


class MinalV3Evaluator(Evaluator):
    """HCE + tempo bonus."""

    def __init__(self, hce: Evaluator) -> None:
        self._hce = hce

    def evaluate_position(self, board: Board) -> int:
        return self._hce.evaluate_position(board) + TEMPO_CP
