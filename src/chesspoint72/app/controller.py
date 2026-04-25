from __future__ import annotations

from dataclasses import dataclass

import chess
import pygame

from chesspoint72.engine.uci.client import UciEngineClient
from chesspoint72.models import GameState
from chesspoint72.ui.renderer import BoardRenderer


@dataclass
class GameConfig:
    engine_path: str | None = None
    evaluator: str | None = None
    hce_modules: str | None = None
    depth: int = 4
    engine_color: bool = chess.BLACK
    think_time: float = 0.2
    square_size: int = 96
    initial_fen: str | None = None


class GameController:
    def __init__(self, config: GameConfig) -> None:
        self.config = config
        self.game_state = (
            GameState.from_fen(config.initial_fen)
            if config.initial_fen
            else GameState()
        )
        self.renderer: BoardRenderer | None = None
        self.engine: object | None = None
        self.last_move: chess.Move | None = None
        self.running = True

    def run(self) -> None:
        pygame.init()
        screen = pygame.display.set_mode((self.config.square_size*8, self.config.square_size*8))
        pygame.display.set_caption("Chesspoint72")
        self.renderer = BoardRenderer(square_size=self.config.square_size)

        if self.config.engine_path:
            self.engine = UciEngineClient(
                engine_path=self.config.engine_path,
                think_time=self.config.think_time,
            )
            self.engine.start()
        elif self.config.evaluator is not None:
            from chesspoint72.app.builtin_engine import BuiltinEngineClient
            self.engine = BuiltinEngineClient(
                evaluator=self.config.evaluator,
                hce_modules=self.config.hce_modules,
                depth=self.config.depth,
                think_time=self.config.think_time,
            )
            self.engine.start()

        try:
            clock = pygame.time.Clock()
            while self.running:
                self._handle_events()
                if self.engine and self._is_engine_turn() and not self.game_state.is_game_over():
                    self._play_engine_move()
                legal_targets = self._legal_targets_for_selection()
                self.renderer.draw(
                    screen,
                    self.game_state.board,
                    selected_square=self.game_state.selected_square,
                    legal_targets=legal_targets,
                    last_move=self.last_move,
                )
                pygame.display.flip()
                clock.tick(60)
        finally:
            if self.engine:
                self.engine.stop()
            pygame.quit()

    def _handle_events(self) -> None:
        assert self.renderer is not None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._on_mouse_click(event.pos)

    def _on_mouse_click(self, pos: tuple[int, int]) -> None:
        assert self.renderer is not None
        if not self._is_human_turn() or self.game_state.is_game_over():
            return

        square = self.renderer.square_from_pixel(pos)
        if square is None:
            return

        piece = self.game_state.board.piece_at(square)
        selected = self.game_state.selected_square

        if selected is None:
            if piece and piece.color == self.game_state.board.turn:
                self.game_state.selected_square = square
            return

        if square == selected:
            self.game_state.selected_square = None
            return

        move = self.build_move(selected, square, self.game_state.board)
        if self.game_state.push_move(move):
            self.last_move = move
            return

        if piece and piece.color == self.game_state.board.turn:
            self.game_state.selected_square = square
        else:
            self.game_state.selected_square = None

    def _play_engine_move(self) -> None:
        if self.engine is None:
            return
        move = self.engine.request_best_move(self.game_state.board)
        if self.game_state.push_move(move):
            self.last_move = move

    def _is_human_turn(self) -> bool:
        if not self.config.engine_path:
            return True
        return self.game_state.board.turn != self.config.engine_color

    def _is_engine_turn(self) -> bool:
        return self.game_state.board.turn == self.config.engine_color

    def _legal_targets_for_selection(self) -> set[int]:
        selected = self.game_state.selected_square
        if selected is None:
            return set()
        return {move.to_square for move in self.game_state.legal_moves_from(selected)}

    @staticmethod
    def build_move(from_square: int, to_square: int, board: chess.Board) -> chess.Move:
        piece = board.piece_at(from_square)
        promotion_rank = chess.square_rank(to_square)

        # Auto-promote to queen to keep mouse-only input lightweight.
        if piece and piece.piece_type == chess.PAWN and promotion_rank in (0, 7):
            return chess.Move(from_square, to_square, promotion=chess.QUEEN)
        return chess.Move(from_square, to_square)

