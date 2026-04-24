from __future__ import annotations

import pygame
import chess
from .sprite_atlas import PieceSpriteAtlas

LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
SELECTED_SQUARE = (242, 246, 130)
LEGAL_TARGET = (138, 197, 100)
LAST_MOVE = (224, 194, 123)
WHITE_PIECE = (245, 245, 245)
BLACK_PIECE = (40, 40, 40)
TEXT_DARK = (15, 15, 15)
TEXT_LIGHT = (250, 250, 250)

PIECE_TEXT = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}


class BoardRenderer:
    def __init__(self, square_size: int = 96) -> None:
        self.square_size = square_size
        self.board_px = square_size * 8
        self.font = pygame.font.SysFont("arial", int(square_size * 0.45), bold=True)
        self.sprites = PieceSpriteAtlas(square_size)

    def square_from_pixel(self, pos: tuple[int, int]) -> int | None:
        x, y = pos
        if not (0 <= x < self.board_px and 0 <= y < self.board_px):
            return None
        file_idx = x // self.square_size
        rank_idx = 7 - (y // self.square_size)
        return chess.square(file_idx, rank_idx)

    def draw(
        self,
        screen: pygame.Surface,
        board: chess.Board,
        selected_square: int | None = None,
        legal_targets: set[int] | None = None,
        last_move: chess.Move | None = None,
    ) -> None:
        legal_targets = legal_targets or set()
        self._draw_board(screen, selected_square, legal_targets, last_move)
        self._draw_pieces(screen, board)

    def _draw_board(
        self,
        screen: pygame.Surface,
        selected_square: int | None,
        legal_targets: set[int],
        last_move: chess.Move | None,
    ) -> None:
        for rank in range(8):
            for file_idx in range(8):
                square = chess.square(file_idx, 7 - rank)
                base = LIGHT_SQUARE if (file_idx + rank) % 2 == 0 else DARK_SQUARE
                if last_move and square in (last_move.from_square, last_move.to_square):
                    base = LAST_MOVE
                if square in legal_targets:
                    base = LEGAL_TARGET
                if selected_square == square:
                    base = SELECTED_SQUARE
                rect = pygame.Rect(
                    file_idx * self.square_size,
                    rank * self.square_size,
                    self.square_size,
                    self.square_size,
                )
                pygame.draw.rect(screen, base, rect)

    def _draw_pieces(self, screen: pygame.Surface, board: chess.Board) -> None:
        radius = int(self.square_size * 0.32)
        for square, piece in board.piece_map().items():
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            x = file_idx * self.square_size
            y = (7 - rank_idx) * self.square_size
            sprite = self.sprites.get(piece.color, piece.piece_type)
            if sprite:
                screen.blit(sprite, (x, y))
            else:
                # fallback: draw circle and letter
                cx = x + self.square_size // 2
                cy = y + self.square_size // 2
                piece_color = WHITE_PIECE if piece.color == chess.WHITE else BLACK_PIECE
                text_color = TEXT_DARK if piece.color == chess.WHITE else TEXT_LIGHT
                pygame.draw.circle(screen, piece_color, (cx, cy), radius)
                glyph = PIECE_TEXT[piece.piece_type]
                text = self.font.render(glyph, True, text_color)
                text_rect = text.get_rect(center=(cx, cy))
                screen.blit(text, text_rect)

