import os
import pygame
import chess

ASSET_DIR = os.path.join(os.path.dirname(__file__), "../assets")

PIECE_FILENAMES = {
    (chess.WHITE, chess.KING): "wK.png",
    (chess.WHITE, chess.QUEEN): "wQ.png",
    (chess.WHITE, chess.ROOK): "wR.png",
    (chess.WHITE, chess.BISHOP): "wB.png",
    (chess.WHITE, chess.KNIGHT): "wN.png",
    (chess.WHITE, chess.PAWN): "wP.png",
    (chess.BLACK, chess.KING): "bK.png",
    (chess.BLACK, chess.QUEEN): "bQ.png",
    (chess.BLACK, chess.ROOK): "bR.png",
    (chess.BLACK, chess.BISHOP): "bB.png",
    (chess.BLACK, chess.KNIGHT): "bN.png",
    (chess.BLACK, chess.PAWN): "bP.png",
}

class PieceSpriteAtlas:
    def __init__(self, square_size: int = 96):
        self.square_size = square_size
        self.sprites = {}
        self._load_sprites()

    def _load_sprites(self):
        for (color, piece_type), filename in PIECE_FILENAMES.items():
            path = os.path.join(ASSET_DIR, filename)
            if os.path.exists(path):
                image = pygame.image.load(path).convert_alpha()
                image = pygame.transform.smoothscale(image, (self.square_size, self.square_size))
                self.sprites[(color, piece_type)] = image
            else:
                self.sprites[(color, piece_type)] = None

    def get(self, color, piece_type):
        return self.sprites.get((color, piece_type))

