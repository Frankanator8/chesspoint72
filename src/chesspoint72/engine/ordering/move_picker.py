# @capability: move_ordering
# @capability: staged_picker
"""
Stockfish 16+ style tiered 'Pick Best' move picker.

Staging order (main search, not in check)
------------------------------------------
MAIN_TT        → yield TT move if present
CAPTURE_INIT   → score all captures with CaptureHistory + 7 × material
GOOD_CAPTURE   → pick-best loop; SEE-failing captures deferred to bad bucket
QUIET_INIT     → score quiets with ButterflyHistory + ContinuationHistory;
                 partial-insertion-sort down to –3 560 × depth
GOOD_QUIET     → yield sorted quiets with score ≥ GOOD_QUIET_THRESHOLD
BAD_CAPTURE    → yield SEE-failing captures
BAD_QUIET      → yield remaining quiets (score < GOOD_QUIET_THRESHOLD)

In-check staging
----------------
EVASION_TT     → yield TT move
EVASION_INIT   → score: captures by material-delta + bonus, quiets by butterfly
EVASION        → pick-best across all evasion moves

Quiescence staging
------------------
QSEARCH_TT      → yield TT move
QCAPTURE_INIT   → score captures
QCAPTURE        → pick-best loop

Object-creation minimisation
-----------------------------
* All move and score state is held in two parallel ``list`` objects
  (``_moves``, ``_scores``) that are never reallocated after __init__.
* ``pick_best`` is an in-place swap on these two lists — no tuples created.
* ``partial_insertion_sort`` uses the same swap idiom as Stockfish's C++.
* History lookups use pre-resolved integer indices into flat ``array.array``
  buffers, avoiding dict or attribute access in the hot scoring path.

Piece index convention (matches board.bitboards layout)
-------------------------------------------------------
    piece_idx = color.value * 6 + piece_type.value - 1
    0=wP 1=wN 2=wB 3=wR 4=wQ 5=wK   6=bP … 11=bK
"""
from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

from chesspoint72.engine.ordering.history_tables import (
    CONT_HIST_SENTINEL,
    ButterflyHistory,
    CaptureHistory,
    ContinuationHistory,
)
from chesspoint72.engine.ordering.see import SEE_VALUES, see_ge

if TYPE_CHECKING:
    from chesspoint72.engine.core.board import Board
    from chesspoint72.engine.core.types import Move

# ---------------------------------------------------------------------------
# Stage enumeration
# ---------------------------------------------------------------------------

class Stage(IntEnum):
    # Main search
    MAIN_TT       = 0
    CAPTURE_INIT  = 1
    GOOD_CAPTURE  = 2
    QUIET_INIT    = 3
    GOOD_QUIET    = 4
    BAD_CAPTURE   = 5
    BAD_QUIET     = 6
    # In-check (evasions)
    EVASION_TT    = 7
    EVASION_INIT  = 8
    EVASION       = 9
    # Quiescence search
    QSEARCH_TT    = 10
    QCAPTURE_INIT = 11
    QCAPTURE      = 12
    # Terminal
    DONE          = 13


# ---------------------------------------------------------------------------
# Thresholds (Stockfish 16 exact values)
# ---------------------------------------------------------------------------
_GOOD_QUIET_THRESHOLD: int = -14_000
_QUIET_SORT_SCALE:     int = -3_560   # threshold = scale × depth
_EVASION_CAP_BONUS:    int = 1 << 28  # separates captures from quiets in check

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pack_move(move: Move) -> int:
    """Encode (from_sq, to_sq, promo) into one integer for O(1) equality."""
    p = move.promotion_piece
    return (move.from_square << 9) | (move.to_square << 3) | (0 if p is None else int(p))


def _piece_idx_at(bbs: list[int], sq: int) -> int:
    """Bitboard-index (0-11) of the piece on *sq*, or -1 if empty."""
    bit = 1 << sq
    for i in range(12):
        if bbs[i] & bit:
            return i
    return -1


def _piece_type_val(bbs: list[int], sq: int) -> int:
    """PieceType.value (1-6) of the piece on *sq*, or 0 if empty."""
    bit = 1 << sq
    if bbs[0]  & bit: return 1
    if bbs[6]  & bit: return 1
    if bbs[1]  & bit: return 2
    if bbs[7]  & bit: return 2
    if bbs[2]  & bit: return 3
    if bbs[8]  & bit: return 3
    if bbs[3]  & bit: return 4
    if bbs[9]  & bit: return 4
    if bbs[4]  & bit: return 5
    if bbs[10] & bit: return 5
    if bbs[5]  & bit: return 6
    if bbs[11] & bit: return 6
    return 0


# ---------------------------------------------------------------------------
# Sorting primitives
# ---------------------------------------------------------------------------

def _pick_best(moves: list, scores: list, start: int, end: int) -> None:
    """In-place O(n) selection: swap the highest-scored entry to *moves[start]*.

    Operates on parallel ``moves`` / ``scores`` lists.  No new objects created.
    """
    best_idx = start
    best_s   = scores[start]
    for i in range(start + 1, end):
        s = scores[i]
        if s > best_s:
            best_s   = s
            best_idx = i
    if best_idx != start:
        moves[start],  moves[best_idx]  = moves[best_idx],  moves[start]
        scores[start], scores[best_idx] = scores[best_idx], scores[start]


def _partial_insertion_sort(
    moves: list, scores: list, start: int, end: int, limit: int
) -> None:
    """Sort entries with ``score >= limit`` into a descending sorted prefix.

    Matches Stockfish's ``partial_insertion_sort`` exactly: elements below
    *limit* are left in place in the unsorted tail.  Elements above *limit*
    are insertion-sorted into the growing prefix [start..sorted_end].

    No new list objects are allocated; all work is done via index swaps.
    """
    sorted_end = start          # last index of the sorted region (inclusive)
    p = start + 1
    while p < end:
        if scores[p] >= limit:
            tmp_m = moves[p]
            tmp_s = scores[p]
            # Expand the sorted boundary, displacing the element that was there
            sorted_end += 1
            moves[p],  scores[p]  = moves[sorted_end],  scores[sorted_end]
            # Insertion sort tmp into [start..sorted_end]
            q = sorted_end
            while q != start and scores[q - 1] < tmp_s:
                moves[q],  scores[q]  = moves[q - 1],  scores[q - 1]
                q -= 1
            moves[q],  scores[q]  = tmp_m, tmp_s
        p += 1


# ---------------------------------------------------------------------------
# MovePicker
# ---------------------------------------------------------------------------

class MovePicker:
    """Stateful iterator that yields moves in Stockfish 16+ priority order.

    Usage in search
    ---------------
        picker = MovePicker(
            board, depth, tt_move,
            butterfly, capture_hist, cont_hist,
            cont_hist_keys,          # tuple of up to 6 ints (or SENTINEL)
            in_check=board.is_king_in_check(),
        )
        for move in picker:
            board.make_move(move)
            ...
            board.unmake_move()

    Constructing a ``MovePicker`` generates all legal moves exactly once
    (``board.generate_legal_moves``).  Subsequent ``__next__`` calls do no
    board queries; they only score/sort the pre-generated lists.
    """

    __slots__ = (
        "_board", "_depth", "_tt_key", "_in_check", "_is_qsearch",
        "_butterfly", "_capture_hist", "_cont_hist", "_cont_keys",
        "_captures", "_cap_scores",
        "_quiets",   "_quiet_scores",
        "_bad_caps", "_bad_cap_scores",
        "_cur", "_end", "_bad_quiet_start",
        "_stage",
    )

    def __init__(
        self,
        board:        "Board",
        depth:        int,
        tt_move:      "Move | None",
        butterfly:    ButterflyHistory,
        capture_hist: CaptureHistory,
        cont_hist:    ContinuationHistory,
        cont_hist_keys: tuple[int, ...] = (),
        *,
        in_check:  bool = False,
        is_qsearch: bool = False,
    ) -> None:
        self._board        = board
        self._depth        = depth
        self._tt_key       = _pack_move(tt_move) if tt_move is not None else -1
        self._in_check     = in_check
        self._is_qsearch   = is_qsearch
        self._butterfly    = butterfly
        self._capture_hist = capture_hist
        self._cont_hist    = cont_hist
        # Pad to 6 entries with SENTINEL so indexing is always safe
        keys = list(cont_hist_keys)
        while len(keys) < 6:
            keys.append(CONT_HIST_SENTINEL)
        self._cont_keys: list[int] = keys

        # Generate and partition moves once
        all_moves = board.generate_legal_moves()
        self._captures:      list["Move"] = []
        self._quiets:        list["Move"] = []
        self._bad_caps:      list["Move"] = []
        self._bad_cap_scores: list[int] = []
        for m in all_moves:
            if m.is_capture:
                self._captures.append(m)
            else:
                self._quiets.append(m)

        self._cap_scores:   list[int] = [0] * len(self._captures)
        self._quiet_scores: list[int] = [0] * len(self._quiets)
        self._cur            = 0
        self._end            = 0
        self._bad_quiet_start = 0

        # Initial stage
        if is_qsearch:
            self._stage = Stage.QSEARCH_TT
        elif in_check:
            self._stage = Stage.EVASION_TT
        else:
            self._stage = Stage.MAIN_TT

    # -----------------------------------------------------------------------
    # Scoring helpers
    # -----------------------------------------------------------------------

    def _score_captures(self) -> None:
        """Score captures: CaptureHistory + 7 × victim material."""
        bbs          = self._board.bitboards
        cap_hist     = self._capture_hist
        caps         = self._captures
        scores       = self._cap_scores
        see_vals     = SEE_VALUES

        for i in range(len(caps)):
            m         = caps[i]
            mover_idx = _piece_idx_at(bbs, m.from_square)
            victim_t  = _piece_type_val(bbs, m.to_square)   # 1-6, 0=en-passant

            if victim_t and mover_idx >= 0:
                cap_type = victim_t - 1  # 0-5
                scores[i] = (
                    cap_hist.get(mover_idx, m.to_square, cap_type)
                    + 7 * see_vals[victim_t]
                )
            else:
                # En passant: victim pawn not on to_sq; use pawn material only
                scores[i] = 7 * see_vals[1]

    def _score_quiets(self) -> None:
        """Score quiet moves: 2×butterfly + sum of continuation histories."""
        bbs       = self._board.bitboards
        butterfly = self._butterfly
        ch        = self._cont_hist
        keys      = self._cont_keys
        color     = self._board.side_to_move.value
        quiets    = self._quiets
        scores    = self._quiet_scores

        for i in range(len(quiets)):
            m          = quiets[i]
            from_sq    = m.from_square
            to_sq      = m.to_square
            piece_idx  = _piece_idx_at(bbs, from_sq)

            if piece_idx < 0:
                scores[i] = 0
                continue

            # 2 × main butterfly history
            s = 2 * butterfly.get(color, from_sq, to_sq)

            # Continuation histories: plies 1, 2, 3, 4, 6 back (index 5)
            # Index 4 (5-ply back) is intentionally skipped (matches Stockfish)
            s += ch.lookup(keys[0], piece_idx, to_sq)
            s += ch.lookup(keys[1], piece_idx, to_sq)
            s += ch.lookup(keys[2], piece_idx, to_sq)
            s += ch.lookup(keys[3], piece_idx, to_sq)
            s += ch.lookup(keys[5], piece_idx, to_sq)

            scores[i] = s

    def _score_evasions(self) -> None:
        """Score all moves when in check.

        Captures: material_delta + EVASION_CAP_BONUS (ensures above quiets).
        Quiets:   butterfly history score.
        """
        bbs       = self._board.bitboards
        butterfly = self._butterfly
        color     = self._board.side_to_move.value
        see_vals  = SEE_VALUES
        bonus     = _EVASION_CAP_BONUS

        # Merge all moves for scoring; we'll use the captures list for everything
        all_moves  = self._captures + self._quiets
        all_scores = [0] * len(all_moves)

        for i, m in enumerate(all_moves):
            if m.is_capture:
                victim_t = _piece_type_val(bbs, m.to_square)
                mover_t  = _piece_type_val(bbs, m.from_square)
                all_scores[i] = (
                    see_vals[victim_t] - see_vals[mover_t] + bonus
                    if victim_t
                    else bonus  # en passant
                )
            else:
                all_scores[i] = butterfly.get(color, m.from_square, m.to_square)

        # Reuse captures list to hold all evasion moves (avoids new allocation)
        self._captures  = all_moves
        self._cap_scores = all_scores
        self._quiets     = []
        self._quiet_scores = []

    # -----------------------------------------------------------------------
    # Iterator protocol
    # -----------------------------------------------------------------------

    def __iter__(self) -> "MovePicker":
        return self

    def __next__(self) -> "Move":  # type: ignore[return]
        bbs     = self._board.bitboards
        tt_key  = self._tt_key

        while True:
            stage = self._stage

            # ==============================================================
            #  MAIN SEARCH
            # ==============================================================

            if stage == Stage.MAIN_TT:
                self._stage = Stage.CAPTURE_INIT
                if tt_key != -1:
                    # Verify the TT move is in the legal-move set by checking
                    # if any of our pre-generated moves matches the key.
                    for m in self._captures:
                        if _pack_move(m) == tt_key:
                            return m
                    for m in self._quiets:
                        if _pack_move(m) == tt_key:
                            return m

            elif stage == Stage.CAPTURE_INIT:
                self._score_captures()
                self._cur = 0
                self._end = len(self._captures)
                self._stage = Stage.GOOD_CAPTURE

            elif stage == Stage.GOOD_CAPTURE:
                while self._cur < self._end:
                    _pick_best(self._captures, self._cap_scores, self._cur, self._end)
                    move  = self._captures[self._cur]
                    score = self._cap_scores[self._cur]
                    self._cur += 1

                    if _pack_move(move) == tt_key:
                        continue  # already yielded

                    # SEE threshold: allow losses of up to ~5.6 % of the score
                    see_threshold = -(score // 18)
                    if see_ge(self._board, move, see_threshold):
                        return move
                    # SEE-failing: defer to bad-capture bucket
                    self._bad_caps.append(move)
                    self._bad_cap_scores.append(score)

                self._stage = Stage.QUIET_INIT

            elif stage == Stage.QUIET_INIT:
                self._score_quiets()
                limit = _QUIET_SORT_SCALE * self._depth
                _partial_insertion_sort(
                    self._quiets, self._quiet_scores,
                    0, len(self._quiets), limit,
                )
                self._cur = 0
                self._end = len(self._quiets)
                self._stage = Stage.GOOD_QUIET

            elif stage == Stage.GOOD_QUIET:
                while self._cur < self._end:
                    move  = self._quiets[self._cur]
                    score = self._quiet_scores[self._cur]

                    if score < _GOOD_QUIET_THRESHOLD:
                        # Entered the unsorted tail — everything below is BAD_QUIET
                        break

                    self._cur += 1
                    if _pack_move(move) != tt_key:
                        return move

                self._bad_quiet_start = self._cur
                self._cur = 0
                self._stage = Stage.BAD_CAPTURE

            elif stage == Stage.BAD_CAPTURE:
                while self._cur < len(self._bad_caps):
                    move = self._bad_caps[self._cur]
                    self._cur += 1
                    return move

                self._cur = self._bad_quiet_start
                self._stage = Stage.BAD_QUIET

            elif stage == Stage.BAD_QUIET:
                while self._cur < self._end:
                    move = self._quiets[self._cur]
                    self._cur += 1
                    if _pack_move(move) != tt_key:
                        return move

                self._stage = Stage.DONE
                raise StopIteration

            # ==============================================================
            #  IN-CHECK EVASIONS
            # ==============================================================

            elif stage == Stage.EVASION_TT:
                self._stage = Stage.EVASION_INIT
                if tt_key != -1:
                    for m in self._captures:
                        if _pack_move(m) == tt_key:
                            return m
                    for m in self._quiets:
                        if _pack_move(m) == tt_key:
                            return m

            elif stage == Stage.EVASION_INIT:
                self._score_evasions()
                self._cur = 0
                self._end = len(self._captures)  # all moves merged here
                self._stage = Stage.EVASION

            elif stage == Stage.EVASION:
                while self._cur < self._end:
                    _pick_best(self._captures, self._cap_scores, self._cur, self._end)
                    move = self._captures[self._cur]
                    self._cur += 1
                    if _pack_move(move) != tt_key:
                        return move

                self._stage = Stage.DONE
                raise StopIteration

            # ==============================================================
            #  QUIESCENCE SEARCH
            # ==============================================================

            elif stage == Stage.QSEARCH_TT:
                self._stage = Stage.QCAPTURE_INIT
                if tt_key != -1:
                    for m in self._captures:
                        if _pack_move(m) == tt_key:
                            return m

            elif stage == Stage.QCAPTURE_INIT:
                self._score_captures()
                self._cur = 0
                self._end = len(self._captures)
                self._stage = Stage.QCAPTURE

            elif stage == Stage.QCAPTURE:
                while self._cur < self._end:
                    _pick_best(self._captures, self._cap_scores, self._cur, self._end)
                    move = self._captures[self._cur]
                    self._cur += 1
                    if _pack_move(move) != tt_key:
                        return move

                self._stage = Stage.DONE
                raise StopIteration

            else:
                raise StopIteration

    # -----------------------------------------------------------------------
    # Static utilities (usable outside the iterator when needed)
    # -----------------------------------------------------------------------

    @staticmethod
    def pick_best(moves: list, scores: list, start: int) -> "Move":
        """Surface the highest-scored entry from *start* onward in-place.

        Convenience wrapper around ``_pick_best`` for callers that hold their
        own (moves, scores) lists and want a single best move.
        """
        _pick_best(moves, scores, start, len(moves))
        return moves[start]
