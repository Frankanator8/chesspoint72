from __future__ import annotations

import time
from typing import TYPE_CHECKING

from chesspoint72.engine.evaluator import Evaluator
from chesspoint72.engine.policies import MoveOrderingPolicy, PruningPolicy
from chesspoint72.engine.search import Search
from chesspoint72.engine.transposition import TranspositionTable
from chesspoint72.engine.types import Move, NodeType

if TYPE_CHECKING:
    from chesspoint72.engine.board import Board

# Sentinel score larger than any reachable centipawn value.
_INF: int = 10_000_000
# Base mate score; subtract ply so the engine prefers shorter mates.
_MATE_SCORE: int = 9_000_000
# Check time every this many nodes to avoid expensive monotonic() calls.
_TIME_CHECK_INTERVAL: int = 1024


class _SearchAborted(Exception):
    """Raised internally to unwind the call stack when time is exhausted."""


class NegamaxSearch(Search):
    """Iterative-deepening negamax with fail-soft alpha-beta and quiescence search.

    All heuristic behaviour is injected:
    - move ordering via ``MoveOrderingPolicy``
    - forward pruning via ``PruningPolicy``
    - position scoring via ``Evaluator``
    - position caching via ``TranspositionTable``
    """

    def __init__(
        self,
        evaluator: Evaluator,
        transposition_table: TranspositionTable,
        move_ordering_policy: MoveOrderingPolicy,
        pruning_policy: PruningPolicy,
    ) -> None:
        super().__init__(evaluator, transposition_table, move_ordering_policy, pruning_policy)
        # Mutable search state — reset at the start of every find_best_move call.
        self._board: Board
        self._start_time: float = 0.0
        self._allotted_time: float = 0.0
        # Tracks the number of half-moves made from the root position so that
        # mate scores can be adjusted per-ply (shorter mates score higher).
        self._ply: int = 0

    # ---------------------------------------------------------------------- #
    # Public interface (implements Search ABC)
    # ---------------------------------------------------------------------- #

    def find_best_move(
        self,
        board: Board,
        max_depth: int,
        allotted_time: float,
    ) -> Move:
        """Iterative deepening loop with time-based abort.

        Searches depth 1, 2, …, *max_depth* in succession.  If the clock
        expires before a depth iteration finishes, the best move from the last
        *fully completed* depth is returned.  A fallback to the first legal
        move is used only when even depth-1 could not be completed.
        """
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
                # Depth was not completed — keep the result from the previous depth.
                break
            if candidate is not None:
                best_move = candidate

        if best_move is None:
            # Absolute last resort: return the first legal move so we never
            # hand back an illegal null move to the caller.
            legal = board.generate_legal_moves()
            if legal:
                best_move = legal[0]

        return best_move  # type: ignore[return-value]  # caller guarantees a non-terminal position

    def search_node(self, alpha: int, beta: int, depth: int) -> int:
        """Fail-soft alpha-beta negamax node.

        Queries the transposition table first, optionally forward-prunes via
        the injected ``PruningPolicy``, then iterates over moves ordered by
        ``MoveOrderingPolicy``.  Writes the result back to the TT on exit.

        Returns:
            Score of the position from the perspective of the side to move.
            Positive means the side to move is winning.
        """
        if depth == 0:
            return self.quiescence_search(alpha, beta)

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

        search_node = self.search_node

        # ------------------------------------------------------------------ #
        # Transposition table probe
        # ------------------------------------------------------------------ #
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
            else:  # UPPER_BOUND
                s = tt_entry.score
                if s < beta:
                    beta = s
            if alpha >= beta:
                return tt_entry.score

        # ------------------------------------------------------------------ #
        # Forward pruning (Futility, Null-Move, etc.)
        # ------------------------------------------------------------------ #
        static_eval = evaluate_position(board)
        prune_score = try_prune(board, depth, alpha, beta, static_eval)
        if prune_score is not None:
            return prune_score

        # ------------------------------------------------------------------ #
        # Move generation and ordering
        # ------------------------------------------------------------------ #
        moves = generate_legal_moves()

        if not moves:
            # Terminal node: checkmate or stalemate.
            if is_king_in_check():
                # Being mated is the worst outcome; subtract ply so the engine
                # seeks mates faster (shorter-ply mates produce larger values
                # at the parent after negation).
                return -(_MATE_SCORE - self._ply)
            return 0  # Stalemate

        moves = order_moves(moves, board, tt_best_move)

        # ------------------------------------------------------------------ #
        # Main search loop (fail-soft alpha-beta)
        # ------------------------------------------------------------------ #
        best_score = -_INF
        best_move: Move | None = None

        for move in moves:
            make_move(move)
            self._ply += 1
            score = -search_node(-beta, -alpha, depth - 1)
            self._ply -= 1
            unmake_move()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                break  # Beta cut-off

        # ------------------------------------------------------------------ #
        # Transposition table store
        # ------------------------------------------------------------------ #
        if best_score <= original_alpha:
            node_type = UPPER  # Failed low — score is an upper bound
        elif best_score >= beta:
            node_type = LOWER  # Failed high — score is a lower bound
        else:
            node_type = EXACT

        tt_store(zobrist, depth, best_score, node_type, best_move)

        return best_score

    def quiescence_search(self, alpha: int, beta: int) -> int:
        """Extend the search at depth 0 by evaluating forcing sequences.

        Only captures are generated.  A "stand-pat" evaluation provides a
        lower bound: if the static score already beats beta the position is
        good enough that further capture analysis is unnecessary.

        Returns:
            Score of the position from the perspective of the side to move.
        """
        self.nodes_evaluated += 1

        board = self._board
        make_move = board.make_move
        unmake_move = board.unmake_move
        generate_legal_moves = board.generate_legal_moves
        evaluate_position = self.evaluator_reference.evaluate_position
        order_moves = self.move_ordering_policy.order_moves
        quiescence_search = self.quiescence_search

        static_eval = evaluate_position(board)

        # Stand-pat pruning — the side to move can choose *not* to capture.
        if static_eval >= beta:
            return beta

        if static_eval > alpha:
            alpha = static_eval

        captures = [m for m in generate_legal_moves() if m.is_capture]
        if not captures:
            return alpha

        captures = order_moves(captures, board)

        for move in captures:
            make_move(move)
            self._ply += 1
            score = -quiescence_search(-beta, -alpha)
            self._ply -= 1
            unmake_move()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _root_search(self, depth: int) -> Move | None:
        """Search all root moves at *depth* and return the best one found.

        Raises:
            _SearchAborted: if the time budget is exhausted mid-iteration;
                the caller discards the partial result and keeps the previous depth's answer.
        """
        alpha = -_INF
        beta = _INF
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
        moves = order_moves(moves, board, tt_best_move)

        for move in moves:
            make_move(move)
            self._ply += 1
            score = -search_node(-beta, -alpha, depth - 1)
            self._ply -= 1
            unmake_move()

            if score > alpha:
                alpha = score
                best_move = move

        return best_move

    def _time_exceeded(self) -> bool:
        return time.monotonic() - self._start_time >= self._allotted_time
