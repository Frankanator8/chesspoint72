"""Move-ordering heuristic tables: KillerMoveTable and HistoryTable."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chesspoint72.engine.core.types import Color, Move

# Upper bound on search depth; both tables pre-allocate to this size.
MAX_DEPTH: int = 64


class KillerMoveTable:
    """Stores the two most recent quiet moves that caused a beta-cutoff at each depth.

    Layout: killers[depth][0] is the most recent killer, [1] is the second-most.
    Captures are still handled by MVV-LVA; killers order the quiet moves that
    follow them.
    """

    def __init__(self, max_depth: int = MAX_DEPTH) -> None:
        self.max_depth = max_depth
        # [depth][slot]: slot 0 = most recent, slot 1 = second-most recent
        self.killers: list[list[Move | None]] = [
            [None, None] for _ in range(max_depth)
        ]

    def update(self, move: Move, depth: int) -> None:
        """Record *move* as the latest killer at *depth*.

        If *move* is already in slot 0, nothing changes (avoid duplicate
        entries that would evict the second killer without adding information).
        The previous slot-0 move is demoted to slot 1.
        """
        if depth >= self.max_depth:
            return
        slot = self.killers[depth]
        if move != slot[0]:
            slot[1] = slot[0]
            slot[0] = move

    def get(self, depth: int) -> list[Move | None]:
        """Return [killer0, killer1] for *depth*, or [None, None] if out of range."""
        if depth >= self.max_depth:
            return [None, None]
        return self.killers[depth]

    def clear(self) -> None:
        """Reset all entries; called at the start of each root search."""
        for row in self.killers:
            row[0] = None
            row[1] = None


class HistoryTable:
    """Accumulates move quality scores indexed by [color][from_square][to_square].

    Scores are incremented by depth_remaining ** 2 each time a move causes a
    beta-cutoff.  Deeper cutoffs receive quadratically larger bonuses, reflecting
    the greater search impact of a move that prunes a higher-depth subtree.
    """

    def __init__(self) -> None:
        # scores[color.value][from_square][to_square]
        self.scores: list[list[list[int]]] = [
            [[0] * 64 for _ in range(64)] for _ in range(2)
        ]

    def update(self, color: Color, move: Move, depth_remaining: int) -> None:
        """Increment the score for (color, from, to) by depth_remaining ** 2."""
        self.scores[color.value][move.from_square][move.to_square] += depth_remaining ** 2

    def get(self, color: Color, from_square: int, to_square: int) -> int:
        """Return the accumulated history score for the given (color, from, to)."""
        return self.scores[color.value][from_square][to_square]

    def clear(self) -> None:
        """Reset all scores; called at the start of each root search."""
        self.scores = [[[0] * 64 for _ in range(64)] for _ in range(2)]
