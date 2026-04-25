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
        self._cont_keys: tuple[int, ...] = ()

    def set_depth(self, depth: int) -> None:
        """Called by NegamaxSearch before each order_moves to pass current depth."""
        self._current_depth = depth

    def set_cont_keys(self, keys: tuple[int, ...]) -> None:
        """Side-channel: pass continuation-history context keys before order_moves.

        Called by GMSearch before each node to wire in the move stack so that
        MovePicker can use continuation history for quiet-move ordering.
        """
        self._cont_keys = keys

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
            cont_hist_keys=self._cont_keys,
            in_check=in_check,
        )
        return list(picker)

    def record_quiet_cutoff(
        self,
        color: int,
        from_sq: int,
        to_sq: int,
        piece_idx: int,
        depth: int,
        cont_keys: tuple[int, ...],
    ) -> None:
        """Update butterfly and continuation history on a quiet-move beta cutoff."""
        bonus = min(depth * depth, 2048)
        self._butterfly.update(color, from_sq, to_sq, bonus)
        for key in cont_keys:
            if key != CONT_HIST_SENTINEL:
                self._cont_hist.update(key, piece_idx, to_sq, bonus)

    def record_capture_cutoff(
        self,
        piece_idx: int,
        to_sq: int,
        cap_type: int,
        depth: int,
    ) -> None:
        """Update capture history on a capture beta cutoff."""
        bonus = min(depth * depth, 2048)
        self._cap_hist.update(piece_idx, to_sq, cap_type, bonus)

    def clear(self) -> None:
        """Reset all history tables; call at the start of each root search."""
        self._butterfly    = ButterflyHistory()
        self._cap_hist     = CaptureHistory()
        self._cont_hist    = ContinuationHistory()
        self._cont_keys    = ()
