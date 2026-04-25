"""Minal v3 search.

Inherits aspiration windows from MinalV2Search and adds:

PVS (Principal Variation Search)
    Every move after the first is tried with a zero-width window [-alpha-1, -alpha].
    Only when the zero-window probe returns score > alpha do we re-search with
    the full window.  This cuts ~20-40% of nodes on positions with a clear PV.

Reverse Futility Pruning (RFP / static null move)
    At depth 1-3, if static_eval - MARGIN[depth] >= beta, return static_eval.
    The position is so clearly above beta that spending time to confirm it
    wastes more than it gains.

Late Move Pruning (LMP)
    At depth 1-4, after LMP_COUNTS[depth] quiet moves have been tried without
    raising alpha, skip the rest.  Ordered-later moves almost never rescue a
    losing position; this frees time for deeper searches elsewhere.

Internal Iterative Deepening (IID)
    When depth >= 4 and no TT move is available, do a quick depth-2 reduced
    search to populate the TT.  The TT move obtained is then used to order
    the real search's first move — often a 10-20% node reduction.

SEE pruning in quiescence
    In quiescence search, captures where SEE < 0 (the initial exchange is
    losing for us) are skipped.  The stand-pat score already accounts for
    not capturing, so we lose nothing by omitting losing captures.

Countermove updates
    On every beta-cutoff from a quiet move, we record the refuting move as
    the counter to the opponent's previous move (_prev_move).  The ordering
    policy (MinalV3MoveOrderingPolicy) retrieves this at ordering time via
    the search reference.
"""
from __future__ import annotations

import time

from chesspoint72.aiengines.minal.v2.search import MinalV2Search, _ASP_DELTA, _ASP_GROW, _ASP_MAX
from chesspoint72.engine.search.negamax.negamax import (
    _INF,
    _MATE_SCORE,
    _SearchAborted,
    _TIME_CHECK_INTERVAL,
)
from chesspoint72.engine.core.types import Move, NodeType
from chesspoint72.engine.ordering.see import see_ge
from chesspoint72.engine.pruning import algorithms as _pruning

# ---------------------------------------------------------------------------
# Reverse Futility Pruning margins (indexed by depth: 1, 2, 3)
# ---------------------------------------------------------------------------
_RFP_MARGIN: dict[int, int] = {1: 100, 2: 200, 3: 300}

# ---------------------------------------------------------------------------
# Late Move Pruning: max quiet moves to try before pruning (by depth 1-4)
# ---------------------------------------------------------------------------
_LMP_COUNTS: dict[int, int] = {1: 5, 2: 10, 3: 18, 4: 28}


class MinalV3Search(MinalV2Search):
    """MinalV2Search + PVS + RFP + LMP + IID + SEE qsearch + countermoves."""

    _CHECK_EXT_MAX_PLY: int = 48

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def find_best_move(self, board, max_depth: int, allotted_time: float) -> Move:
        # V3-specific state
        self._prev_move: Move | None = None
        # Countermove table: [from_sq][to_sq] → Move | None
        self.countermove_table: list[list[Move | None]] = [
            [None] * 64 for _ in range(64)
        ]
        # Delegate to v2's aspiration-window iterative deepening.
        return super().find_best_move(board, max_depth, allotted_time)

    # ------------------------------------------------------------------ #
    # Root search — override to manage _prev_move at the root level
    # ------------------------------------------------------------------ #

    def _root_search_scored(
        self, depth: int, alpha: int, beta: int
    ) -> tuple[Move | None, int]:
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

        for move_index, move in enumerate(moves):
            # Expose move to the recursive search_node so the child sees the
            # correct _prev_move (= the move we just played, i.e. the move
            # the opponent will want to counter at the next level).
            self._prev_move = move
            board.make_move(move)
            self._ply += 1

            if move_index == 0:
                score = -self.search_node(-beta, -alpha, depth - 1)
            else:
                # PVS zero-window probe at root too.
                score = -self.search_node(-alpha - 1, -alpha, depth - 1)
                if alpha < score < beta:
                    score = -self.search_node(-beta, -alpha, depth - 1)

            self._ply -= 1
            board.unmake_move()
            self._prev_move = None

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        return best_move, best_score

    # ------------------------------------------------------------------ #
    # Main search node — all v3 features
    # ------------------------------------------------------------------ #

    def search_node(self, alpha: int, beta: int, depth: int) -> int:  # noqa: C901 (complex but intentional)
        # ---- Check extension (inherited from v2) -------------------------
        if depth == 0:
            if (self._ply <= self._CHECK_EXT_MAX_PLY
                    and self._board.is_king_in_check()):
                depth = 1
            else:
                return self.quiescence_search(alpha, beta)

        # ---- Bookkeeping -------------------------------------------------
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
        EXACT = NodeType.EXACT
        LOWER = NodeType.LOWER_BOUND
        UPPER = NodeType.UPPER_BOUND
        search_node = self.search_node   # self-reference so overrides propagate

        # ---- Transposition table probe -----------------------------------
        zobrist = calculate_zobrist_hash()
        tt_entry = tt.retrieve_position(zobrist)
        tt_best_move: Move | None = None
        original_alpha = alpha

        if tt_entry is not None and tt_entry.depth >= depth:
            tt_best_move = tt_entry.best_move
            if tt_entry.node_type == EXACT:
                return tt_entry.score
            elif tt_entry.node_type == LOWER:
                s = tt_entry.score
                if s > alpha: alpha = s
            else:
                s = tt_entry.score
                if s < beta: beta = s
            if alpha >= beta:
                return tt_entry.score

        in_check = is_king_in_check()
        static_eval = evaluate_position(board)

        # ---- Reverse Futility Pruning (static null move) -----------------
        # If our static eval is so far above beta that even giving away a
        # tempo can't change the outcome, prune immediately.
        if (not in_check
                and depth in _RFP_MARGIN
                and static_eval - _RFP_MARGIN[depth] >= beta
                and abs(static_eval) < _MATE_SCORE - 1000):
            return static_eval

        # ---- Forward pruning (NMP, futility, razoring) -------------------
        prune_score = try_prune(board, depth, alpha, beta, static_eval)
        if prune_score is not None:
            return prune_score

        # ---- Internal Iterative Deepening (IID) --------------------------
        # When we have no TT move to guide ordering, a shallow probe fills
        # the TT so the full search can use a good first move.
        if depth >= 4 and tt_best_move is None and not in_check:
            search_node(alpha, beta, depth - 2)
            iid_entry = tt.retrieve_position(zobrist)
            if iid_entry is not None:
                tt_best_move = iid_entry.best_move

        # ---- Move generation + ordering ----------------------------------
        moves = generate_legal_moves()
        if not moves:
            return -(_MATE_SCORE - self._ply) if in_check else 0

        moves = order_moves(moves, board, tt_best_move)

        # ---- Main search loop --------------------------------------------
        best_score = -_INF
        best_move: Move | None = None
        cfg = self.pruning_config
        quiet_count = 0
        lmp_limit = _LMP_COUNTS.get(depth)   # None if depth > 4

        for move_index, move in enumerate(moves):
            move_is_quiet = (not move.is_capture) and (move.promotion_piece is None)

            # Futility pruning (depth 1, quiet moves)
            if cfg is not None and move_is_quiet and _pruning.is_futile(
                depth, alpha, static_eval, in_check, move_is_quiet, cfg
            ):
                continue

            # Late Move Pruning: at shallow depth, skip quiet moves past the limit
            if (move_is_quiet
                    and lmp_limit is not None
                    and not in_check
                    and quiet_count >= lmp_limit
                    and alpha > -(_MATE_SCORE - 1000)):
                quiet_count += 1
                continue

            if move_is_quiet:
                quiet_count += 1

            # Set _prev_move so child nodes see the correct countermove context.
            saved_prev = self._prev_move
            self._prev_move = move

            make_move(move)
            self._ply += 1
            gives_check = is_king_in_check()

            # ---- PVS + LMR (combined) ------------------------------------
            if move_index == 0:
                # First (best-ordered) move: full window, no reduction
                score = -search_node(-beta, -alpha, depth - 1)
            else:
                # All other moves: try a zero-window search first.
                if cfg is not None and _pruning.should_apply_lmr(
                    depth, move_index, move_is_quiet, in_check, gives_check, cfg
                ):
                    # Reduced depth zero-window
                    r = _pruning.lmr_reduction(depth, move_index)
                    score = -search_node(-alpha - 1, -alpha, max(1, depth - 1 - r))
                    # If LMR succeeded (> alpha), re-search at full depth ZW
                    if score > alpha:
                        score = -search_node(-alpha - 1, -alpha, depth - 1)
                else:
                    # Full depth zero-window
                    score = -search_node(-alpha - 1, -alpha, depth - 1)

                # PVS re-search: if ZW failed high, prove with full window
                if alpha < score < beta:
                    score = -search_node(-beta, -alpha, depth - 1)

            self._ply -= 1
            unmake_move()
            self._prev_move = saved_prev

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                # Killer + history
                self.killer_table.update(move, depth)
                self.update_history(move, depth)
                # Countermove: record the refutation of saved_prev
                if move_is_quiet and saved_prev is not None:
                    self.countermove_table[saved_prev.from_square][saved_prev.to_square] = move
                break

        # ---- Transposition table store -----------------------------------
        if best_score <= original_alpha:
            node_type = UPPER
        elif best_score >= beta:
            node_type = LOWER
        else:
            node_type = EXACT

        tt.store_position(zobrist, depth, best_score, node_type, best_move)
        return best_score

    # ------------------------------------------------------------------ #
    # Quiescence search — SEE pruning
    # ------------------------------------------------------------------ #

    def quiescence_search(self, alpha: int, beta: int) -> int:
        self.nodes_evaluated += 1

        board = self._board
        evaluate_position = self.evaluator_reference.evaluate_position
        order_moves = self.move_ordering_policy.order_moves

        static_eval = evaluate_position(board)

        if static_eval >= beta:
            return beta
        if static_eval > alpha:
            alpha = static_eval

        # Generate captures only; filter out losing ones via SEE.
        all_captures = [m for m in board.generate_legal_moves() if m.is_capture]
        captures = [m for m in all_captures if see_ge(board, m, 0)]

        if not captures:
            return alpha

        captures = order_moves(captures, board)

        for move in captures:
            board.make_move(move)
            self._ply += 1
            score = -self.quiescence_search(-beta, -alpha)
            self._ply -= 1
            board.unmake_move()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha
