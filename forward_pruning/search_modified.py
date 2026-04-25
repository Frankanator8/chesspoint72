"""NegamaxSearch with the forward-pruning module wired in.

This file is the *separate-folder* counterpart to
``src/chesspoint72/engine/negamax.py``. It is a copy of that search with
the pruning calls inserted in the prescribed order:

    NMP  →  Razoring  →  (enter move loop)  →  Futility  →  LMR

The diff between this file and the upstream ``negamax.py`` is the Phase-4
integration patch (see ``SEARCH_DIFF.md``).

Wiring choice (matches codebase convention): ``PruningConfig`` is passed
into the constructor and stored on ``self`` — same Strategy-Pattern style
as ``MoveOrderingPolicy`` and ``PruningPolicy``. We do NOT replace the
existing ``PruningPolicy`` hook; the upstream ``try_prune`` call is left
intact so users can layer custom policies on top if they want.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from chesspoint72.engine.evaluator import Evaluator
from chesspoint72.engine.policies import MoveOrderingPolicy, PruningPolicy
from chesspoint72.engine.search import Search
from chesspoint72.engine.transposition import TranspositionTable
from chesspoint72.engine.types import Move, NodeType

from forward_pruning import pruning

if TYPE_CHECKING:
    from chesspoint72.engine.board import Board

_INF: int = 10_000_000
_MATE_SCORE: int = 9_000_000
_TIME_CHECK_INTERVAL: int = 1024


class _SearchAborted(Exception):
    pass


class PrunedNegamaxSearch(Search):
    """Iterative-deepening negamax with the forward-pruning module enabled."""

    def __init__(
        self,
        evaluator: Evaluator,
        transposition_table: TranspositionTable,
        move_ordering_policy: MoveOrderingPolicy,
        pruning_policy: PruningPolicy,
        pruning_config,
    ) -> None:
        super().__init__(evaluator, transposition_table, move_ordering_policy, pruning_policy)
        self.pruning_config = pruning_config
        self._board: Board
        self._start_time: float = 0.0
        self._allotted_time: float = 0.0
        self._ply: int = 0

    # ---------------------------------------------------------------------- #
    # Public interface
    # ---------------------------------------------------------------------- #

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
        self._ply = 0

        best_move: Move | None = None
        for depth in range(1, max_depth + 1):
            if self._time_exceeded():
                break
            try:
                candidate = self._root_search(depth)
            except _SearchAborted:
                break
            if candidate is not None:
                best_move = candidate

        if best_move is None:
            legal = board.generate_legal_moves()
            if legal:
                best_move = legal[0]
        return best_move  # type: ignore[return-value]

    def search_node(self, alpha: int, beta: int, depth: int) -> int:
        if depth == 0:
            return self.quiescence_search(alpha, beta)

        self.nodes_evaluated += 1
        if self.nodes_evaluated & (_TIME_CHECK_INTERVAL - 1) == 0 and self._time_exceeded():
            raise _SearchAborted()

        # ---------------- Transposition table probe ---------------- #
        zobrist = self._board.calculate_zobrist_hash()
        tt_entry = self.transposition_table_reference.retrieve_position(zobrist)
        tt_best_move: Move | None = None
        original_alpha = alpha
        if tt_entry is not None and tt_entry.depth >= depth:
            tt_best_move = tt_entry.best_move
            if tt_entry.node_type == NodeType.EXACT:
                return tt_entry.score
            elif tt_entry.node_type == NodeType.LOWER_BOUND:
                alpha = max(alpha, tt_entry.score)
            else:
                beta = min(beta, tt_entry.score)
            if alpha >= beta:
                return tt_entry.score

        in_check = self._board.is_king_in_check()
        static_eval = self.evaluator_reference.evaluate_position(self._board)

        # ----------------- (1) Null-move pruning ----------------- #
        nmp_score = pruning.try_null_move_pruning(
            self, self._board, depth, beta, in_check, self.pruning_config
        )
        if nmp_score is not None:
            return nmp_score

        # --------------------- (2) Razoring --------------------- #
        razor_score = pruning.try_razoring(
            self, depth, alpha, static_eval, self.pruning_config
        )
        if razor_score is not None:
            return razor_score

        # ---------- Existing pluggable PruningPolicy hook ---------- #
        # Left in place to preserve the original extension point.
        prune_score = self.pruning_policy.try_prune(
            self._board, depth, alpha, beta, static_eval
        )
        if prune_score is not None:
            return prune_score

        # ---------------- Move generation + ordering ---------------- #
        moves = self._board.generate_legal_moves()
        if not moves:
            if in_check:
                return -(_MATE_SCORE - self._ply)
            return 0  # stalemate

        moves = self.move_ordering_policy.order_moves(moves, self._board, tt_best_move)

        # ---------------------- Main search loop ---------------------- #
        best_score = -_INF
        best_move: Move | None = None
        for move_index, move in enumerate(moves):
            move_is_quiet = (not move.is_capture) and (move.promotion_piece is None)

            # ---- (3) Futility pruning (move-level, depth==1, quiet) ---- #
            if move_is_quiet and pruning.is_futile(
                depth, alpha, static_eval, in_check, move_is_quiet, self.pruning_config
            ):
                continue

            self._board.make_move(move)
            self._ply += 1

            gives_check = self._board.is_king_in_check()

            # --------------------- (4) LMR --------------------- #
            if pruning.should_apply_lmr(
                depth, move_index, move_is_quiet, in_check, gives_check, self.pruning_config
            ):
                r = pruning.lmr_reduction(depth, move_index)
                reduced_depth = max(1, depth - 1 - r)
                score = -self.search_node(-alpha - 1, -alpha, reduced_depth)
                if score > alpha:
                    # Re-search at full depth.
                    score = -self.search_node(-beta, -alpha, depth - 1)
            else:
                score = -self.search_node(-beta, -alpha, depth - 1)

            self._ply -= 1
            self._board.unmake_move()

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        # -------------------- Transposition store -------------------- #
        if best_score <= original_alpha:
            node_type = NodeType.UPPER_BOUND
        elif best_score >= beta:
            node_type = NodeType.LOWER_BOUND
        else:
            node_type = NodeType.EXACT
        self.transposition_table_reference.store_position(
            zobrist, depth, best_score, node_type, best_move
        )
        return best_score

    def quiescence_search(self, alpha: int, beta: int) -> int:
        # Untouched copy of the upstream QS — Phase 4 says do not modify QS.
        self.nodes_evaluated += 1
        static_eval = self.evaluator_reference.evaluate_position(self._board)
        if static_eval >= beta:
            return beta
        if static_eval > alpha:
            alpha = static_eval

        captures = [m for m in self._board.generate_legal_moves() if m.is_capture]
        if not captures:
            return alpha
        captures = self.move_ordering_policy.order_moves(captures, self._board)

        for move in captures:
            self._board.make_move(move)
            self._ply += 1
            score = -self.quiescence_search(-beta, -alpha)
            self._ply -= 1
            self._board.unmake_move()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _root_search(self, depth: int) -> Move | None:
        alpha = -_INF
        beta = _INF
        best_move: Move | None = None

        zobrist = self._board.calculate_zobrist_hash()
        tt_entry = self.transposition_table_reference.retrieve_position(zobrist)
        tt_best_move = tt_entry.best_move if tt_entry is not None else None

        moves = self._board.generate_legal_moves()
        moves = self.move_ordering_policy.order_moves(moves, self._board, tt_best_move)

        for move in moves:
            self._board.make_move(move)
            self._ply += 1
            score = -self.search_node(-beta, -alpha, depth - 1)
            self._ply -= 1
            self._board.unmake_move()
            if score > alpha:
                alpha = score
                best_move = move
        return best_move

    def _time_exceeded(self) -> bool:
        return time.monotonic() - self._start_time >= self._allotted_time
