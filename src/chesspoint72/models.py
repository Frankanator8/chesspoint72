from __future__ import annotations

from dataclasses import dataclass, field

import chess


@dataclass
class GameState:
    """Tracks board state and move history."""

    board: chess.Board = field(default_factory=chess.Board)
    selected_square: int | None = None
    move_history: list[str] = field(default_factory=list)

    @classmethod
    def from_fen(cls, fen: str) -> "GameState":
        return cls(board=chess.Board(fen=fen))

    def reset(self) -> None:
        self.board.reset()
        self.selected_square = None
        self.move_history.clear()

    def legal_moves_from(self, square: int) -> list[chess.Move]:
        return [move for move in self.board.legal_moves if move.from_square == square]

    def push_move(self, move: chess.Move) -> bool:
        if move not in self.board.legal_moves:
            return False
        self.board.push(move)
        self.move_history.append(move.uci())
        self.selected_square = None
        return True

    def push_uci(self, move_uci: str) -> bool:
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            return False
        return self.push_move(move)

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def result(self) -> str:
        if not self.is_game_over():
            return "*"
        return self.board.result(claim_draw=True)

