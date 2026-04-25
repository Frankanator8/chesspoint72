"""NNUE-based evaluator backed by a small fully-connected PyTorch model.

Architecture is dynamic: 768 -> h1 -> h2 -> 1, where h1 and h2 are read
from the checkpoint dictionary. Legacy checkpoints (plain state_dict) default
to h1=256, h2=32 to stay backward-compatible with nnue_weights.pt.

Input encoding: 12 piece-type planes (P,N,B,R,Q,K,p,n,b,r,q,k) of 64 squares,
flattened to a 768-element float tensor with 1.0 where a piece sits.
"""
# @capability: evaluator
# @capability: nnue
from __future__ import annotations

from pathlib import Path

import chess
import torch
import torch.nn as nn

from chesspoint72.engine.core.board import Board
from chesspoint72.engine.core.evaluator import Evaluator


_PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
}


def fen_to_tensor(fen: str) -> torch.Tensor:
    """Convert a FEN string into a flattened 768-element float tensor."""
    tensor = torch.zeros(768, dtype=torch.float32)
    board = chess.Board(fen)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            idx = _PIECE_TO_INDEX[piece.symbol()]
            tensor[idx * 64 + square] = 1.0
    return tensor


class NnueNetwork(nn.Module):
    """Fully-connected NNUE: 768 -> h1 -> h2 -> 1."""

    def __init__(self, h1: int = 256, h2: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "nnue_weights.pt"


class NnueEvaluator(Evaluator):
    """Evaluator that runs a trained NNUE on the current position.

    Supports two checkpoint formats:
    - New format: dict with keys ``state_dict``, ``h1``, ``h2``
    - Legacy format: raw OrderedDict (plain state_dict); assumes 256x32
    """

    def __init__(self, weights_path: str | Path = _DEFAULT_WEIGHTS_PATH) -> None:
        self._device = torch.device("cpu")
        checkpoint = torch.load(str(weights_path), map_location=self._device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            h1 = int(checkpoint.get("h1", 256))
            h2 = int(checkpoint.get("h2", 32))
            state_dict = checkpoint["state_dict"]
        else:
            h1, h2 = 256, 32
            state_dict = checkpoint
        self._model = NnueNetwork(h1, h2).to(self._device)
        self._model.load_state_dict(state_dict)
        self._model.eval()

    def evaluate_position(self, board: Board) -> int:
        get_fen = getattr(board, "get_current_fen", None)
        fen = get_fen() if callable(get_fen) else board.fen()
        x = fen_to_tensor(fen).unsqueeze(0).to(self._device)
        with torch.no_grad():
            score = self._model(x).item()
        return int(round(score))
