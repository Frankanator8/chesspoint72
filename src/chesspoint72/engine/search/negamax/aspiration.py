"""AspirationNegamaxSearch — iterative deepening with aspiration windows.

Aspiration windows narrow the root alpha/beta window to
    [prev_score - delta, prev_score + delta]
on every depth iteration after the first, provoking far more beta-cutoffs
than a full-window search. When the score falls outside the window the window
widens exponentially until the score is inside it, then the search continues.

Stage 5-B of the eval pipeline A/B tests this variant against the baseline
full-window NegamaxSearch.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from chesspoint72.engine.search.negamax.negamax import (
    NegamaxSearch,
    _INF,
    _SearchAborted,
)

if TYPE_CHECKING:
    from chesspoint72.engine.core.board import Board
    from chesspoint72.engine.core.types import Move

_INITIAL_DELTA: int = 50   # centipawns; tune via Stage 5-B results


class AspirationNegamaxSearch(NegamaxSearch):
    """NegamaxSearch with aspiration windows at the root.

    Overrides ``find_best_move`` to use a narrowed [prev-delta, prev+delta]
    window after the first depth iteration. The window widens by 2× on each
    fail-low or fail-high until the score is captured, matching the standard
    Stockfish aspiration approach.
    """

    def find_best_move(
        self,
        board: Board,
        max_depth: int,
        allotted_time: float,
    ) -> Move:
        self._board = board
        self._allotted_time = allotted_time
        self._start_time = time.monotonic()
        self.nodes_evaluated = 0
        self.beta_cutoffs = 0
        self.tt_lookups = 0
        self.tt_hits = 0
        self.depth_reached = 0
        self._last_root_score = 0
        self._ply = 0
        self.killer_table.clear()
        self.history_table.clear()

        best_move: Move | None = None
        prev_score: int | None = None

        for depth in range(1, max_depth + 1):
            if self._time_exceeded():
                break
            try:
                if prev_score is None or depth <= 1:
                    candidate = self._root_search(depth)
                    prev_score = self._last_root_score
                else:
                    candidate = self._aspiration_root_search(depth, prev_score)
                    prev_score = self._last_root_score
            except _SearchAborted:
                break
            if candidate is not None:
                best_move = candidate
            self.depth_reached = depth

        if best_move is None:
            legal = board.generate_legal_moves()
            if legal:
                best_move = legal[0]

        return best_move  # type: ignore[return-value]

    def _aspiration_root_search(self, depth: int, prev_score: int) -> Move | None:
        """Root search with an exponentially-widening aspiration window."""
        delta = _INITIAL_DELTA
        alpha = prev_score - delta
        beta  = prev_score + delta

        while True:
            candidate = self._root_search_windowed(depth, alpha, beta)
            score = self._last_root_score

            if score <= alpha:
                alpha = max(alpha - delta, -_INF)
                delta *= 2
            elif score >= beta:
                beta = min(beta + delta, _INF)
                delta *= 2
            else:
                return candidate

    def _root_search_windowed(self, depth: int, alpha: int, beta: int) -> Move | None:
        """Root search over an arbitrary [alpha, beta] window.

        Mirrors _root_search but accepts explicit bounds so aspiration logic
        can widen them on fail-low/fail-high without duplicating search code.
        """
        best_move: Move | None = None

        board = self._board
        make_move = board.make_move
        unmake_move = board.unmake_move
        search_node = self.search_node
        order_moves = self.move_ordering_policy.order_moves

        zobrist = board.calculate_zobrist_hash()
        tt_entry = self.transposition_table_reference.retrieve_position(zobrist)
        tt_best_move = tt_entry.best_move if tt_entry is not None else None

        moves = board.generate_legal_moves()
        if self._set_ordering_depth is not None:
            self._set_ordering_depth(depth)
        moves = order_moves(moves, board, tt_best_move)

        current_alpha = alpha
        for move in moves:
            make_move(move)
            self._ply += 1
            score = -search_node(-beta, -current_alpha, depth - 1)
            self._ply -= 1
            unmake_move()

            if score > current_alpha:
                current_alpha = score
                best_move = move

        self._last_root_score = current_alpha
        return best_move
