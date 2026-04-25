"""MovePickerPolicy — MoveOrderingPolicy backed by the Stockfish-style MovePicker.

Wraps ``MovePicker`` (SEE + ButterflyHistory + CaptureHistory + ContinuationHistory)
as a ``MoveOrderingPolicy``. The eval pipeline Stage 5-A A/B test uses this to
measure the Elo gain of the full Stockfish 16+ ordering pipeline over the simpler
MoveSorter (MVV-LVA + killers + history).

Limitations vs. a native search integration
--------------------------------------------
* Continuation history requires the two most-recent moves made on the stack.
  ``order_moves`` has no access to this context, so CONT_HIST_SENTINEL is used
  for all six continuation-history keys — effectively disabling that component.
* ButterflyHistory and CaptureHistory ARE populated via ``record_cutoff``, which
  ``NegamaxSearch`` calls through the ``set_depth`` / ``record_cutoff`` protocol.
  Until NegamaxSearch exposes a richer callback, history bonuses are accumulated
  but the ordering gain from continuation history is absent.
* Killer moves are NOT used — MovePicker uses ButterflyHistory for quiet ordering.
  The Stage 5-A A/B test therefore measures SEE capture ordering + ButterflyHistory
  vs. MVV-LVA + killers + HistoryTable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from chesspoint72.engine.core.policies import MoveOrderingPolicy
from chesspoint72.engine.ordering.history_tables import (
    CONT_HIST_SENTINEL,
    ButterflyHistory,
    CaptureHistory,
    ContinuationHistory,
)
from chesspoint72.engine.ordering.move_picker import MovePicker

if TYPE_CHECKING:
    from chesspoint72.engine.core.board import Board
    from chesspoint72.engine.core.types import Move


class MovePickerPolicy(MoveOrderingPolicy):
    """MoveOrderingPolicy backed by the full Stockfish 16+ MovePicker pipeline."""

    def __init__(self) -> None:
        self._butterfly    = ButterflyHistory()
        self._cap_hist     = CaptureHistory()
        self._cont_hist    = ContinuationHistory()
        self._current_depth: int = 0

    def set_depth(self, depth: int) -> None:
        """Called by NegamaxSearch before each order_moves to pass current depth."""
        self._current_depth = depth

    def order_moves(
        self,
        moves: list[Move],
        board: Board,
        tt_best_move: Move | None = None,
    ) -> list[Move]:
        in_check = board.is_king_in_check()
        picker = MovePicker(
            board=board,
            depth=self._current_depth,
            tt_move=tt_best_move,
            butterfly=self._butterfly,
            capture_hist=self._cap_hist,
            cont_hist=self._cont_hist,
            cont_hist_keys=(),  # continuation history disabled (no stack context)
            in_check=in_check,
        )
        return list(picker)

    def clear(self) -> None:
        """Reset all history tables; call at the start of each root search."""
        self._butterfly    = ButterflyHistory()
        self._cap_hist     = CaptureHistory()
        self._cont_hist    = ContinuationHistory()
