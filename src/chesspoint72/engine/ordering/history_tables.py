"""
Stockfish 16+ history tables for move ordering.

Tables implemented
------------------
ButterflyHistory       [color][from_sq][to_sq]         cap = 7 183
CaptureHistory         [piece_idx][to_sq][cap_type]    cap = 10 692
ContinuationHistory    [prev_ctx][curr_ctx]            cap = 30 000
    where ctx = piece_idx * 64 + sq  (0-767 per colour-piece-square triple)

All tables use the same *gravity / aging* update rule from Stockfish:

    new_val = old_val + clamped_bonus - old_val * |clamped_bonus| / cap

The multiplicative decay keeps values in [-cap, +cap] and ages stale entries:
large bonuses pull the value strongly *and* wipe out history from earlier plies.

Piece index convention (matches board.bitboards layout):
    piece_idx = color.value * 6 + piece_type.value - 1
    0=wP 1=wN 2=wB 3=wR 4=wQ 5=wK  6=bP ... 11=bK

Storage: all tables use ``array.array`` of signed 16-bit integers ('h') to
minimise memory footprint and avoid per-element Python object overhead.
"""
from __future__ import annotations

from array import array

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BUTTERFLY_CAP:   int = 7_183
CAPTURE_CAP:     int = 10_692
CONT_HIST_CAP:   int = 30_000

PIECE_NB:        int = 12   # 6 piece types × 2 colours
SQUARE_NB:       int = 64
PIECE_TYPE_NB:   int = 6    # PAWN..KING (PieceType.value 1-6)

# Context key for "no previous move" (sentinel inserted into search stacks
# before any real move has been made).
CONT_HIST_SENTINEL: int = -1

# ---------------------------------------------------------------------------
# Shared gravity update (inline-able via manual inlining in hot paths)
# ---------------------------------------------------------------------------

def gravity_update(val: int, bonus: int, cap: int) -> int:
    """Stockfish gravity / aging formula.

    Clamps *bonus* to [-cap, +cap], then:
        val + clamped - val * |clamped| / cap

    The ``val * |clamped| / cap`` term is a multiplicative decay: large updates
    strongly age stale values; small updates preserve existing history.
    """
    clamped = bonus if bonus <= cap else cap
    if clamped < -cap:
        clamped = -cap
    return val + clamped - val * (clamped if clamped >= 0 else -clamped) // cap


# ---------------------------------------------------------------------------
# ButterflyHistory  [color: 2][from_sq: 64][to_sq: 64]
# ---------------------------------------------------------------------------

class ButterflyHistory:
    """Main quiet-move history indexed by (color, from_square, to_square).

    Stored as a flat array of 8 192 signed int16 values (2 × 64 × 64).
    """

    __slots__ = ("_data",)
    _STRIDE_COLOR: int = 64 * 64   # 4 096

    def __init__(self) -> None:
        self._data: array = array("h", [0] * (2 * 64 * 64))

    # Inline-able index computation
    @staticmethod
    def _idx(color: int, from_sq: int, to_sq: int) -> int:
        return color * 4096 + from_sq * 64 + to_sq

    def get(self, color: int, from_sq: int, to_sq: int) -> int:
        return self._data[color * 4096 + from_sq * 64 + to_sq]

    def update(self, color: int, from_sq: int, to_sq: int, bonus: int) -> None:
        idx = color * 4096 + from_sq * 64 + to_sq
        v = self._data[idx]
        new_v = gravity_update(v, bonus, BUTTERFLY_CAP)
        # Clamp to int16 range before storing
        self._data[idx] = max(-32768, min(32767, new_v))

    def clear(self) -> None:
        for i in range(len(self._data)):
            self._data[i] = 0


# ---------------------------------------------------------------------------
# CaptureHistory  [piece_idx: 12][to_sq: 64][cap_type: 6]
# ---------------------------------------------------------------------------

class CaptureHistory:
    """Capture history indexed by (moving piece, destination, captured piece type).

    Scoring formula (from movepick.cpp):
        score = capture_history[piece][to_sq][cap_type] + 7 * material[cap_type]

    Stored as 4 608 signed int16 values (12 × 64 × 6).
    """

    __slots__ = ("_data",)
    _SIZE: int = PIECE_NB * SQUARE_NB * PIECE_TYPE_NB  # 4 608

    def __init__(self) -> None:
        self._data: array = array("h", [0] * self._SIZE)

    @staticmethod
    def _idx(piece_idx: int, to_sq: int, cap_type: int) -> int:
        # cap_type = piece_type.value - 1  (0-5)
        return (piece_idx * SQUARE_NB + to_sq) * PIECE_TYPE_NB + cap_type

    def get(self, piece_idx: int, to_sq: int, cap_type: int) -> int:
        return self._data[(piece_idx * SQUARE_NB + to_sq) * PIECE_TYPE_NB + cap_type]

    def update(self, piece_idx: int, to_sq: int, cap_type: int, bonus: int) -> None:
        idx = (piece_idx * SQUARE_NB + to_sq) * PIECE_TYPE_NB + cap_type
        v = self._data[idx]
        new_v = gravity_update(v, bonus, CAPTURE_CAP)
        self._data[idx] = max(-32768, min(32767, new_v))

    def clear(self) -> None:
        for i in range(len(self._data)):
            self._data[i] = 0


# ---------------------------------------------------------------------------
# ContinuationHistory  [prev_ctx: 768][curr_ctx: 768]
# ---------------------------------------------------------------------------
# ctx = piece_idx * SQUARE_NB + sq  →  range [0, 768)
# Full table: 768 × 768 = 589 824 int16 values ≈ 1.2 MB

_CTX_NB: int = PIECE_NB * SQUARE_NB   # 768


class ContinuationHistory:
    """Continuation history (follow-up / counter-move history).

    Maps (previous piece+square context, current piece+square) to a score.
    During search, each node stores a *context key*:
        key = piece_idx * 64 + sq      (range 0..767)
    CONT_HIST_SENTINEL (-1) is used for uninitialised stack frames.

    The search passes up to 6 context keys (plies 1-4, 6 back) to MovePicker.
    ``lookup(key, curr_piece, curr_sq)`` returns 0 for the sentinel.

    Stored flat as 589 824 signed int16 values ≈ 1.2 MB.
    """

    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: array = array("h", [0] * (_CTX_NB * _CTX_NB))

    # ---- Key helpers -------------------------------------------------------

    @staticmethod
    def make_key(piece_idx: int, sq: int) -> int:
        """Pack (piece_idx, sq) into a single integer context key."""
        return piece_idx * SQUARE_NB + sq

    # ---- Lookup / update ----------------------------------------------------

    def lookup(self, key: int, curr_piece: int, curr_sq: int) -> int:
        """Return the score for *key* → (curr_piece, curr_sq).

        Returns 0 for the sentinel key.
        """
        if key == CONT_HIST_SENTINEL:
            return 0
        return self._data[key * _CTX_NB + curr_piece * SQUARE_NB + curr_sq]

    def update(self, key: int, curr_piece: int, curr_sq: int, bonus: int) -> None:
        """Apply a gravity update for *key* → (curr_piece, curr_sq)."""
        if key == CONT_HIST_SENTINEL:
            return
        idx = key * _CTX_NB + curr_piece * SQUARE_NB + curr_sq
        v = self._data[idx]
        new_v = gravity_update(v, bonus, CONT_HIST_CAP)
        self._data[idx] = max(-32768, min(32767, new_v))

    def clear(self) -> None:
        for i in range(len(self._data)):
            self._data[i] = 0
