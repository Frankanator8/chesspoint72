"""Minal v2 custom search.

Extends NegamaxSearch with two improvements:

1. Aspiration windows — iterative deepening uses a ±DELTA window around the
   previous depth's score instead of [-INF, +INF].  On a fail-high/low the
   window is widened exponentially until it encompasses the true score, then
   the result is accepted.  This cuts the number of nodes searched at each
   depth significantly when the score is stable.

2. Check extensions — when the side to move is in check at the normally
   terminal depth (depth == 0 inside search_node), we extend 1 extra ply
   rather than dropping into quiescence immediately.  This prevents the
   horizon effect from hiding forced-checkmate sequences.
"""
from __future__ import annotations

import time

from chesspoint72.engine.search.negamax.negamax import (
    NegamaxSearch,
    _INF,
    _MATE_SCORE,
    _SearchAborted,
    _TIME_CHECK_INTERVAL,
)
from chesspoint72.engine.core.types import Move, NodeType
from chesspoint72.engine.pruning import algorithms as _pruning

# Initial aspiration window (centipawns).  Widened by ×GROW on each failure.
_ASP_DELTA: int = 50
_ASP_GROW: float = 4.0      # multiply delta on fail-high or fail-low
_ASP_MAX: int = _INF        # don't widen past full window


class MinalV2Search(NegamaxSearch):
    """NegamaxSearch + aspiration windows + check extensions."""

    # Maximum ply at which check extension fires.  Caps runaway recursion in
    # forced-check chains (rare but theoretically possible).
    _CHECK_EXT_MAX_PLY: int = 48

    def find_best_move(
        self,
        board,
        max_depth: int,
        allotted_time: float,
    ) -> Move:
        self._board = board
        self._allotted_time = allotted_time
        self._start_time = time.monotonic()
        self.nodes_evaluated = 0
        self._ply = 0
        self.killer_table.clear()
        self.history_table.clear()

        # Let the ordering policy read our live tables by ply.
        if hasattr(self.move_ordering_policy, "attach_search"):
            self.move_ordering_policy.attach_search(self)

        best_move: Move | None = None
        prev_score: int = 0

        for depth in range(1, max_depth + 1):
            if self._time_exceeded():
                break

            # Aspiration windows from depth 2 onward.
            if depth >= 2:
                delta = _ASP_DELTA
                alpha = max(-_INF, prev_score - delta)
                beta  = min( _INF, prev_score + delta)
            else:
                alpha, beta = -_INF, _INF

            try:
                candidate, score = self._root_search_windowed(depth, alpha, beta)
            except _SearchAborted:
                break

            if candidate is not None:
                best_move = candidate
                prev_score = score

        if best_move is None:
            legal = board.generate_legal_moves()
            if legal:
                best_move = legal[0]

        return best_move  # type: ignore[return-value]

    def _root_search_windowed(
        self,
        depth: int,
        alpha: int,
        beta: int,
    ) -> tuple[Move | None, int]:
        """Search at *depth* with aspiration window [alpha, beta].

        If the result falls outside the window, widens and re-searches.
        Returns (best_move, score).
        """
        delta = max(beta - alpha, _ASP_DELTA)

        while True:
            move, score = self._root_search_scored(depth, alpha, beta)

            if score <= alpha:
                # Fail-low: widen downward
                alpha = max(-_INF, alpha - delta)
                delta = min(int(delta * _ASP_GROW), _ASP_MAX)
            elif score >= beta:
                # Fail-high: widen upward
                beta = min(_INF, beta + delta)
                delta = min(int(delta * _ASP_GROW), _ASP_MAX)
            else:
                return move, score

            # Full-window search never needs a retry
            if alpha <= -_INF and beta >= _INF:
                return move, score

    def _root_search_scored(
        self,
        depth: int,
        alpha: int,
        beta: int,
    ) -> tuple[Move | None, int]:
        """Root search with explicit window; returns (best_move, best_score)."""
        best_move: Move | None = None
        best_score = -_INF

        board = self._board
        order_moves = self.move_ordering_policy.order_moves

        zobrist = board.calculate_zobrist_hash()
        tt_entry = self.transposition_table_reference.retrieve_position(zobrist)
        tt_best = tt_entry.best_move if tt_entry is not None else None

        moves = board.generate_legal_moves()
        if not moves:
            return None, 0

        moves = order_moves(moves, board, tt_best)

        for move in moves:
            board.make_move(move)
            self._ply += 1
            score = -self.search_node(-beta, -alpha, depth - 1)
            self._ply -= 1
            board.unmake_move()

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        return best_move, best_score

    def search_node(self, alpha: int, beta: int, depth: int) -> int:
        """Alpha-beta negamax with check extension at the horizon.

        When depth reaches 0 while the side to move is in check we extend
        by 1 ply (capped at _CHECK_EXT_MAX_PLY) so that we resolve the check
        before falling into quiescence.
        """
        if depth == 0:
            if (self._ply <= self._CHECK_EXT_MAX_PLY
                    and self._board.is_king_in_check()):
                depth = 1   # extend: search 1 more ply to escape the check
            else:
                return self.quiescence_search(alpha, beta)

        # ------------------------------------------------------------------ #
        # From here: identical to NegamaxSearch.search_node (depth >= 1)
        # ------------------------------------------------------------------ #
        self.nodes_evaluated += 1
        if self.nodes_evaluated & (_TIME_CHECK_INTERVAL - 1) == 0 and self._time_exceeded():
            raise _SearchAborted()

        board = self._board
        make_move = board.make_move
        unmake_move = board.unmake_move
        generate_legal_moves = board.generate_legal_moves
        is_king_in_check = board.is_king_in_check
        calculate_zobrist_hash = board.calculate_zobrist_hash

        evaluate_position = self.evaluator_reference.evaluate_position
        try_prune = self.pruning_policy.try_prune
        order_moves = self.move_ordering_policy.order_moves

        tt = self.transposition_table_reference
        tt_retrieve = tt.retrieve_position
        tt_store = tt.store_position

        EXACT = NodeType.EXACT
        LOWER = NodeType.LOWER_BOUND
        UPPER = NodeType.UPPER_BOUND

        search_node = self.search_node   # picks up our override for child nodes

        # TT probe
        zobrist = calculate_zobrist_hash()
        tt_entry = tt_retrieve(zobrist)
        tt_best_move: Move | None = None
        original_alpha = alpha

        if tt_entry is not None and tt_entry.depth >= depth:
            tt_best_move = tt_entry.best_move
            if tt_entry.node_type == EXACT:
                return tt_entry.score
            elif tt_entry.node_type == LOWER:
                s = tt_entry.score
                if s > alpha:
                    alpha = s
            else:
                s = tt_entry.score
                if s < beta:
                    beta = s
            if alpha >= beta:
                return tt_entry.score

        # Forward pruning
        in_check = is_king_in_check()
        static_eval = evaluate_position(board)
        prune_score = try_prune(board, depth, alpha, beta, static_eval)
        if prune_score is not None:
            return prune_score

        # Move generation + ordering
        moves = generate_legal_moves()
        if not moves:
            if in_check:
                return -(_MATE_SCORE - self._ply)
            return 0

        moves = order_moves(moves, board, tt_best_move)

        # Main search loop
        best_score = -_INF
        best_move: Move | None = None
        cfg = self.pruning_config

        for move_index, move in enumerate(moves):
            move_is_quiet = (not move.is_capture) and (move.promotion_piece is None)

            if cfg is not None and move_is_quiet and _pruning.is_futile(
                depth, alpha, static_eval, in_check, move_is_quiet, cfg
            ):
                continue

            make_move(move)
            self._ply += 1
            gives_check = is_king_in_check()

            if cfg is not None and _pruning.should_apply_lmr(
                depth, move_index, move_is_quiet, in_check, gives_check, cfg
            ):
                r = _pruning.lmr_reduction(depth, move_index)
                reduced_depth = max(1, depth - 1 - r)
                score = -search_node(-alpha - 1, -alpha, reduced_depth)
                if score > alpha:
                    score = -search_node(-beta, -alpha, depth - 1)
            else:
                score = -search_node(-beta, -alpha, depth - 1)

            self._ply -= 1
            unmake_move()

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                self.killer_table.update(move, depth)
                self.update_history(move, depth)
                break

        # TT store
        if best_score <= original_alpha:
            node_type = UPPER
        elif best_score >= beta:
            node_type = LOWER
        else:
            node_type = EXACT

        tt_store(zobrist, depth, best_score, node_type, best_move)
        return best_score
