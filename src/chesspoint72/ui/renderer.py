from __future__ import annotations

from dataclasses import dataclass, field

import pygame
import chess
from .sprite_atlas import PieceSpriteAtlas

# Board colours
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
SELECTED_SQUARE = (242, 246, 130)
LEGAL_TARGET = (138, 197, 100)
LAST_MOVE = (224, 194, 123)
WHITE_PIECE = (245, 245, 245)
BLACK_PIECE = (40, 40, 40)
TEXT_DARK = (15, 15, 15)
TEXT_LIGHT = (250, 250, 250)

# Sidebar colours
SIDEBAR_WIDTH = 320
SIDEBAR_BG = (20, 20, 25)
SIDEBAR_ACCENT = (100, 180, 255)
SIDEBAR_TEXT = (220, 220, 220)
SIDEBAR_DIM = (130, 130, 140)
SIDEBAR_DIVIDER = (55, 55, 65)
SCORE_POS = (90, 200, 110)
SCORE_NEG = (210, 80, 80)

PIECE_TEXT = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}


@dataclass
class SidebarData:
    move_san: str = ""
    move_number: int = 0
    is_white: bool = True
    score_cp: int | None = None     # None for human moves
    depth: int = 0
    nodes: int = 0
    pv_san: list[str] = field(default_factory=list)
    explanation: str = ""
    thinking: bool = False          # True while waiting for Claude


def _wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        if font.size(test)[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


class BoardRenderer:
    def __init__(self, square_size: int = 96) -> None:
        self.square_size = square_size
        self.board_px = square_size * 8
        self.font = pygame.font.SysFont("arial", int(square_size * 0.45), bold=True)
        self.sprites = PieceSpriteAtlas(square_size)
        # Sidebar fonts
        self._title_font = pygame.font.SysFont("arial", 16, bold=True)
        self._move_font = pygame.font.SysFont("arial", 30, bold=True)
        self._score_font = pygame.font.SysFont("arial", 24, bold=True)
        self._label_font = pygame.font.SysFont("arial", 13)
        self._body_font = pygame.font.SysFont("arial", 13)

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
        sidebar_data: SidebarData | None = None,
    ) -> None:
        legal_targets = legal_targets or set()
        self._draw_board(screen, selected_square, legal_targets, last_move)
        self._draw_pieces(screen, board)
        self._draw_sidebar(screen, sidebar_data or SidebarData())

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
                cx = x + self.square_size // 2
                cy = y + self.square_size // 2
                piece_color = WHITE_PIECE if piece.color == chess.WHITE else BLACK_PIECE
                text_color = TEXT_DARK if piece.color == chess.WHITE else TEXT_LIGHT
                pygame.draw.circle(screen, piece_color, (cx, cy), radius)
                glyph = PIECE_TEXT[piece.piece_type]
                text = self.font.render(glyph, True, text_color)
                text_rect = text.get_rect(center=(cx, cy))
                screen.blit(text, text_rect)

    def _draw_sidebar(self, screen: pygame.Surface, data: SidebarData) -> None:
        sx = self.board_px
        sw = SIDEBAR_WIDTH
        sh = self.board_px
        pygame.draw.rect(screen, SIDEBAR_BG, pygame.Rect(sx, 0, sw, sh))

        pad = 16
        x = sx + pad
        y = 18
        line_gap = 4

        # ── Title ────────────────────────────────────────────────────────── #
        self._blit(screen, self._title_font, "ANALYSIS", SIDEBAR_ACCENT, x, y)
        y += self._title_font.get_height() + 10
        self._hline(screen, sx, y, sw)
        y += 10

        if not data.move_san:
            self._blit(screen, self._body_font, "Make a move to begin.", SIDEBAR_DIM, x, y)
            return

        # ── Move ─────────────────────────────────────────────────────────── #
        color_str = "White" if data.is_white else "Black"
        self._blit(screen, self._label_font, f"Move {data.move_number}  ·  {color_str}", SIDEBAR_DIM, x, y)
        y += self._label_font.get_height() + 3
        self._blit(screen, self._move_font, data.move_san, SIDEBAR_TEXT, x, y)
        y += self._move_font.get_height() + 12

        if data.score_cp is not None:
            # ── Score ─────────────────────────────────────────────────────── #
            score_val = data.score_cp / 100.0
            score_str = f"{score_val:+.2f}"
            score_color = SCORE_POS if data.score_cp >= 0 else SCORE_NEG
            self._blit(screen, self._score_font, score_str, score_color, x, y)
            y += self._score_font.get_height() + 4

            nodes_k = data.nodes // 1000
            stats = f"depth {data.depth}    nodes {nodes_k}k"
            self._blit(screen, self._label_font, stats, SIDEBAR_DIM, x, y)
            y += self._label_font.get_height() + 12

            # ── PV line ───────────────────────────────────────────────────── #
            if data.pv_san:
                self._blit(screen, self._label_font, "Principal variation", SIDEBAR_DIM, x, y)
                y += self._label_font.get_height() + 3
                pv_text = "  ".join(data.pv_san[:6])
                for line in _wrap_text(pv_text, self._body_font, sw - pad * 2):
                    self._blit(screen, self._body_font, line, SIDEBAR_TEXT, x, y)
                    y += self._body_font.get_height() + line_gap
                y += 8

        # ── Explanation ───────────────────────────────────────────────────── #
        self._hline(screen, sx, y, sw)
        y += 10
        self._blit(screen, self._title_font, "WHY THIS MOVE", SIDEBAR_ACCENT, x, y)
        y += self._title_font.get_height() + 6

        if data.thinking:
            self._blit(screen, self._body_font, "Analyzing...", SIDEBAR_DIM, x, y)
        elif data.explanation:
            for line in _wrap_text(data.explanation, self._body_font, sw - pad * 2):
                if y + self._body_font.get_height() > sh - 10:
                    break
                self._blit(screen, self._body_font, line, SIDEBAR_TEXT, x, y)
                y += self._body_font.get_height() + line_gap
        elif data.score_cp is None:
            self._blit(screen, self._body_font, "Human move — no analysis.", SIDEBAR_DIM, x, y)

    @staticmethod
    def _blit(
        screen: pygame.Surface,
        font: pygame.font.Font,
        text: str,
        color: tuple[int, int, int],
        x: int,
        y: int,
    ) -> None:
        screen.blit(font.render(text, True, color), (x, y))

    @staticmethod
    def _hline(screen: pygame.Surface, sx: int, y: int, width: int) -> None:
        pygame.draw.line(screen, SIDEBAR_DIVIDER, (sx, y), (sx + width, y))
