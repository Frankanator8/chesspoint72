"""MoveSorter: tiered move scoring and lazy pick-best iteration.

Score tiers (highest to lowest priority):
    TT move        1_000_000        searched first — best known refutation
    Captures      ~10_000–90_000   ranked by MVV-LVA within the tier
    Killers        5_000 / 4_999   quiet moves that caused beta-cutoffs here
    History        0+               accumulated from prior beta-cutoffs (depth^2)

Typical search usage (pick-best, not full sort):

    for move in sorter.iter_moves(board, moves, tt_move, depth):
        board.make_move(move)
        score = -search_node(...)
        board.unmake_move()
        if alpha >= beta:
            break   # paid O(k·n) not O(n log n); k is almost always 1-3
"""
# @capability: move_ordering
# @capability: killers
# @capability: history
from __future__ import annotations

from typing import TYPE_CHECKING

from chesspoint72.engine.ordering.heuristics import HistoryTable, KillerMoveTable

if TYPE_CHECKING:
    from chesspoint72.engine.core.board import Board
    from chesspoint72.engine.core.types import Move

# ---------------------------------------------------------------------------
# Tier constants
# ---------------------------------------------------------------------------
_TT_SCORE: int = 1_000_000
_KILLER_SCORE_0: int = 5_000   # most recent killer at this depth
_KILLER_SCORE_1: int = 4_999   # second killer
_CAPTURE_BASE: int = 10_000    # en-passant fallback (victim not on to_square)

# ---------------------------------------------------------------------------
# Capture scoring table
#
# Victim score drives the broad 10k-90k range; aggressor cost is a small
# tiebreaker (max 900) so every legal capture stays well above the 5k killer
# floor.  All values are precomputed — zero arithmetic inside the hot loop.
#
# _VICTIM_SCORE and _AGGRESSOR_COST are indexed by PieceType.value (1-6);
# index 0 is unused (no piece on that square), kept as 0 for safe fallback.
# ---------------------------------------------------------------------------
_VICTIM_SCORE:    list[int] = [0, 10_000, 30_000, 30_000, 50_000, 90_000,      0]
_AGGRESSOR_COST:  list[int] = [0,    100,    300,    300,    500,    900,      0]

# _CAPTURE_SCORE[victim_piece_type_val][aggressor_piece_type_val]
_CAPTURE_SCORE: list[list[int]] = [
    [_VICTIM_SCORE[v] - _AGGRESSOR_COST[a] for a in range(7)]
    for v in range(7)
]

# ---------------------------------------------------------------------------
# Helpers — module-level, not called per-iteration after the loop begins
# ---------------------------------------------------------------------------

def _pack_move(move: Move) -> int:
    """Encode (from_sq, to_sq, promo) into one integer for O(1) comparison.

    Bit layout (15 bits total):
        bits 14-9  from_square  (6 bits, 0-63)
        bits  8-3  to_square    (6 bits, 0-63)
        bits  2-0  promotion    (3 bits, 0 = none, 1-6 = PieceType.value)
    """
    p = move.promotion_piece
    return (move.from_square << 9) | (move.to_square << 3) | (0 if p is None else p)


def _piece_type_value_at(bitboards: list[int], square: int) -> int:
    """Return the PieceType value (1-6) of the piece on *square*, or 0 if empty.

    Uses only bitwise operations; creates no Python objects.
    Board bitboard layout: index = color.value * 6 + (piece_type.value - 1),
    so piece_type.value = (index % 6) + 1.
    """
    bit = 1 << square
    if bitboards[0]  & bit: return 1  # WHITE PAWN
    if bitboards[6]  & bit: return 1  # BLACK PAWN
    if bitboards[1]  & bit: return 2  # WHITE KNIGHT
    if bitboards[7]  & bit: return 2  # BLACK KNIGHT
    if bitboards[2]  & bit: return 3  # WHITE BISHOP
    if bitboards[8]  & bit: return 3  # BLACK BISHOP
    if bitboards[3]  & bit: return 4  # WHITE ROOK
    if bitboards[9]  & bit: return 4  # BLACK ROOK
    if bitboards[4]  & bit: return 5  # WHITE QUEEN
    if bitboards[10] & bit: return 5  # BLACK QUEEN
    if bitboards[5]  & bit: return 6  # WHITE KING
    if bitboards[11] & bit: return 6  # BLACK KING
    return 0


# ---------------------------------------------------------------------------
# MoveSorter
# ---------------------------------------------------------------------------

class MoveSorter:
    """Scores moves and serves them in best-first order via lazy selection.

    get_scored_moves  — assigns a priority integer to every move in O(n).
    pick_best         — one O(n) linear scan to surface the best remaining
                        move; swaps it in-place so the next call starts after
                        it.  Never sorts the full list.
    iter_moves        — generator combining both; use it in the search loop
                        so the engine pays O(k·n) rather than O(n log n),
                        where k is the number of moves tried before a cutoff.

    Optimisations:
    * All expensive lookups (TT key, killer keys, bitboard pointer, history
      colour-slice) are resolved once before the scoring loop.
    * Move identity uses a packed integer key rather than dataclass equality,
      avoiding per-field Python attribute lookups.
    * Capture piece-type lookup uses unrolled bitwise checks — no new objects
      created, no tuple returned.
    * list.append is aliased to a local to skip one attribute lookup per move.
    * pick_best avoids a second allocation: it mutates the existing list with
      a single parallel swap and returns the winning move directly.
    """

    __slots__ = ("_killers", "_history")

    def __init__(
        self,
        killer_table: KillerMoveTable,
        history_table: HistoryTable,
    ) -> None:
        self._killers = killer_table
        self._history = history_table

    def get_scored_moves(
        self,
        board: Board,
        moves: list[Move],
        tt_move: Move | None,
        depth: int,
    ) -> list[tuple[Move, int]]:
        """Score every move in *moves* and return (move, score) pairs.

        Args:
            board:   Current position. Must have populated bitboards for
                     accurate capture scoring (shim boards score captures as
                     _CAPTURE_BASE instead).
            moves:   Legal (or pseudo-legal) moves to score.
            tt_move: Best move from the transposition table, or None.
            depth:   Remaining search depth, used to index the killer table.

        Returns:
            Unsorted list of (move, score) tuples.  Pass to iter_moves (via
            pick_best) for lazy best-first iteration, or sort explicitly when
            the full ordered list is required.
        """
        # ------------------------------------------------------------------ #
        # Resolve all external state once — nothing re-looked-up in the loop  #
        # ------------------------------------------------------------------ #
        tt_key: int = _pack_move(tt_move) if tt_move is not None else -1

        killers = self._killers.get(depth)
        k0, k1 = killers[0], killers[1]
        k0_key: int = _pack_move(k0) if k0 is not None else -2
        k1_key: int = _pack_move(k1) if k1 is not None else -3

        bitboards: list[int] = board.bitboards
        # Pre-index the colour slice: history[from_sq][to_sq] is a plain int.
        history: list[list[int]] = self._history.scores[board.side_to_move.value]

        result: list[tuple[Move, int]] = []
        append = result.append  # local alias avoids one attribute lookup/move

        # ------------------------------------------------------------------ #
        # Hot loop — one branch per tier, cheapest checks first               #
        # ------------------------------------------------------------------ #
        for move in moves:
            from_sq: int = move.from_square
            to_sq:   int = move.to_square
            promo         = move.promotion_piece

            # Pack into one int; no new Python object, single comparison per tier.
            move_key: int = (from_sq << 9) | (to_sq << 3) | (0 if promo is None else promo)

            if move_key == tt_key:
                score = _TT_SCORE

            elif move.is_capture:
                v = _piece_type_value_at(bitboards, to_sq)
                a = _piece_type_value_at(bitboards, from_sq)
                # v == 0 only for en passant (captured pawn not on to_sq).
                score = _CAPTURE_SCORE[v][a] if v else _CAPTURE_BASE

            elif move_key == k0_key:
                score = _KILLER_SCORE_0

            elif move_key == k1_key:
                score = _KILLER_SCORE_1

            else:
                score = history[from_sq][to_sq]

            append((move, score))

        return result

    # ---------------------------------------------------------------------- #
    # Pick-best lazy iteration
    # ---------------------------------------------------------------------- #

    @staticmethod
    def pick_best(scored: list[tuple[Move, int]], start: int) -> Move:
        """Find the highest-scored entry from *start* onward, swap it to
        *scored[start]*, and return the move.

        One linear O(n − start) pass; mutates *scored* in-place with a single
        parallel swap.  No new list or tuple is allocated.

        Args:
            scored: List of (move, score) pairs produced by get_scored_moves.
                    Modified in-place so repeated calls with increasing *start*
                    values implement a full selection sort lazily.
            start:  Index of the current search position in *scored*.

        Returns:
            The move with the highest score at or after *scored[start]*.
        """
        best_idx = start
        best_score = scored[start][1]
        for i in range(start + 1, len(scored)):
            s = scored[i][1]
            if s > best_score:
                best_score = s
                best_idx = i
        if best_idx != start:
            scored[start], scored[best_idx] = scored[best_idx], scored[start]
        return scored[start][0]

    def iter_moves(
        self,
        board: Board,
        moves: list[Move],
        tt_move: Move | None,
        depth: int,
    ):
        """Yield moves in best-first order using lazy pick-best selection.

        Scores all moves once (O(n)), then for each position does a single
        linear scan to surface the next best move (O(n − i) per step).
        Total cost when k moves are tried before a beta-cutoff: O(k·n).
        A full list.sort() would always pay O(n log n) regardless of k.

        Args:
            board:   Current position (same as get_scored_moves).
            moves:   Legal moves to iterate.
            tt_move: Transposition-table best move hint, or None.
            depth:   Remaining search depth for killer-table lookup.

        Yields:
            Move objects in descending score order, one at a time.
        """
        scored = self.get_scored_moves(board, moves, tt_move, depth)
        for i in range(len(scored)):
            yield MoveSorter.pick_best(scored, i)
