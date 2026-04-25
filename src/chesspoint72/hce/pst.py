from __future__ import annotations

import chess

# Tables are written rank-8-at-top, a-file-at-left (how you'd draw a board).
#
# Index mapping:
#   White piece on square sq  →  pst[sq ^ 56]   (flip rank, keep file)
#   Black piece on square sq  →  pst[sq]         (vertical mirror is free)
#
# python-chess square encoding: a1=0 … h1=7, a2=8 … h8=63
# XOR-56 trick: bit-pattern (rank, file) → (7-rank, file)

# fmt: off

_PAWN_MG: list[int] = [
     0,  0,  0,  0,  0,  0,  0,  0,   # rank 8
    50, 50, 50, 50, 50, 50, 50, 50,   # rank 7 – nearly promoted
    10, 10, 20, 30, 30, 20, 10, 10,   # rank 6
     5,  5, 10, 25, 25, 10,  5,  5,   # rank 5
     0,  0,  0, 20, 20,  0,  0,  0,   # rank 4 – center pawns
     5, -5,-10,  0,  0,-10, -5,  5,   # rank 3
     5, 10, 10,-20,-20, 10, 10,  5,   # rank 2 – penalty for blocking d/e
     0,  0,  0,  0,  0,  0,  0,  0,   # rank 1
]

_PAWN_EG: list[int] = [
     0,  0,  0,  0,  0,  0,  0,  0,
    80, 80, 80, 80, 80, 80, 80, 80,   # advancement matters most in EG
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
    20, 20, 20, 20, 20, 20, 20, 20,
    10, 10, 10, 10, 10, 10, 10, 10,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
]

_KNIGHT_MG: list[int] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,   # corners/edges heavily penalised
]

_KNIGHT_EG: list[int] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

_BISHOP_MG: list[int] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,   # long diagonals rewarded
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

_BISHOP_EG: list[int] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

_ROOK_MG: list[int] = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,   # 7th-rank invasion bonus
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,   # central open-file nudge
]

_ROOK_EG: list[int] = [
    10, 10, 10, 10, 10, 10, 10, 10,   # any 8th-rank control good in EG
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  5,  5,  5,  5,  0,  0,
     0,  0,  5,  5,  5,  5,  0,  0,
     0,  0,  5,  5,  5,  5,  0,  0,
     0,  0,  5,  5,  5,  5,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

_QUEEN_MG: list[int] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,   # d1/e1 penalised – avoid early queen out
    -20,-10,-10, -5, -5,-10,-10,-20,
]

_QUEEN_EG: list[int] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -10,  5, 10, 10, 10, 10,  5,-10,
      0,  0, 10, 15, 15, 10,  0, -5,
     -5,  0, 10, 15, 15, 10,  0, -5,
    -10,  5, 10, 10, 10, 10,  5,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

_KING_MG: list[int] = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,   # castled positions rewarded
     20, 30, 10,  0,  0, 10, 30, 20,   # g1/b1 ideal castled squares
]

_KING_EG: list[int] = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,   # king must centralise in EG
]

# fmt: on

_PST_MG: dict[int, list[int]] = {
    chess.PAWN:   _PAWN_MG,
    chess.KNIGHT: _KNIGHT_MG,
    chess.BISHOP: _BISHOP_MG,
    chess.ROOK:   _ROOK_MG,
    chess.QUEEN:  _QUEEN_MG,
    chess.KING:   _KING_MG,
}

_PST_EG: dict[int, list[int]] = {
    chess.PAWN:   _PAWN_EG,
    chess.KNIGHT: _KNIGHT_EG,
    chess.BISHOP: _BISHOP_EG,
    chess.ROOK:   _ROOK_EG,
    chess.QUEEN:  _QUEEN_EG,
    chess.KING:   _KING_EG,
}

# Phase weight per piece type (both sides combined max = 24).
_PHASE_WEIGHTS: dict[int, int] = {
    chess.QUEEN:  4,
    chess.ROOK:   2,
    chess.BISHOP: 1,
    chess.KNIGHT: 1,
}
_MAX_PHASE = 24  # 2Q×4 + 4R×2 + 4B×1 + 4N×1


def _game_phase(board: chess.Board) -> float:
    """Return midgame fraction: 1.0 = opening, 0.0 = pure endgame."""
    phase = sum(
        len(board.pieces(pt, chess.WHITE)) * w + len(board.pieces(pt, chess.BLACK)) * w
        for pt, w in _PHASE_WEIGHTS.items()
    )
    return min(phase, _MAX_PHASE) / _MAX_PHASE


def pst_score(board: chess.Board) -> int:
    """Return tapered PST positional bonus in centipawns from White's perspective."""
    mg = eg = 0
    for sq, piece in board.piece_map().items():
        idx = sq ^ 56 if piece.color == chess.WHITE else sq
        sign = 1 if piece.color == chess.WHITE else -1
        mg += sign * _PST_MG[piece.piece_type][idx]
        eg += sign * _PST_EG[piece.piece_type][idx]

    phase = _game_phase(board)
    return round(mg * phase + eg * (1.0 - phase))
