"""GMSearch — grandmaster-strength alpha-beta with five tactical upgrades.

Built on top of AspirationNegamaxSearch, adding:

1. Check extensions
   Every node where the side to move is in check gets depth += 1.
   Prevents the horizon effect cutting off forcing lines mid-sequence.
   Sacrifices: unbounded depth growth in forced-check ladders (rare;
   handled by the time limit).

2. Principal Variation Search (PVS)
   First move at each node is searched with the full [alpha, beta] window.
   All subsequent moves use a zero-width null window [-alpha-1, -alpha]
   to prove they can't raise alpha cheaply, then re-search with full
   window only on failure. Cuts roughly 20-30 % of searched nodes with
   no accuracy loss on the PV.

3. Continuation history wiring
   GMSearch maintains a move-context stack and passes the last 6 plies'
   (piece, square) keys to MovePickerPolicy before each order_moves call,
   enabling the full Stockfish 16 continuation-history bonus (which was
   previously always zero because the keys were empty).
   Also updates continuation history and butterfly table on beta cutoffs
   so the tables improve throughout the search.

4. SEE pruning in quiescence search
   Captures that Static Exchange Evaluation says lose material (SEE < 0)
   are skipped entirely. This removes the "garbage capture" sequences that
   inflate the QSearch tree without providing useful information.

5. Delta pruning in quiescence search
   If the static evaluation plus the maximum possible gain (queen value
   + a 200 cp safety margin = 2738 cp) still can't raise alpha, the
   entire QSearch subtree is pruned. Avoids exploring positions that are
   already hopeless regardless of captures.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from chesspoint72.engine.core.types import NodeType
from chesspoint72.engine.ordering.history_tables import CONT_HIST_SENTINEL
from chesspoint72.engine.ordering.see import SEE_VALUES, see_ge
from chesspoint72.engine.pruning import algorithms as _pruning
from chesspoint72.engine.search.negamax.aspiration import AspirationNegamaxSearch
from chesspoint72.engine.search.negamax.negamax import (
    _INF,
    _MATE_SCORE,
    _SearchAborted,
    _TIME_CHECK_INTERVAL,
)

if TYPE_CHECKING:
    pass

# Centipawn safety margin added to the queen value in delta pruning.
# A 200 cp margin avoids pruning positions that might have a promotion bonus.
_DELTA_MARGIN: int = 200
# Upper bound on the gain from any single capture (queen SEE value + margin).
_MAX_CAPTURE_GAIN: int = SEE_VALUES[5] + _DELTA_MARGIN  # 2538 + 200 = 2738


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_piece_idx(bbs: list[int], sq: int) -> int:
    """Return the 0-11 bitboard index of the piece on *sq*, or -1 if empty."""
    bit = 1 << sq
    for i in range(12):
        if bbs[i] & bit:
            return i
    return -1


def _get_cap_type(bbs: list[int], sq: int) -> int:
    """Return (piece_type.value - 1) of the piece on *sq* for CaptureHistory."""
    bit = 1 << sq
    # Piece types: P=0 N=1 B=2 R=3 Q=4 K=5 (0-indexed)
    for pt_minus1, (wi, bi) in enumerate([(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)]):
        if bbs[wi] & bit or bbs[bi] & bit:
            return pt_minus1
    return 0  # en passant captures return 0 (pawn)


# ---------------------------------------------------------------------------
# GMSearch
# ---------------------------------------------------------------------------

class GMSearch(AspirationNegamaxSearch):
    """Grandmaster-strength search: check extensions + PVS + cont history +
    SEE/delta pruning in QSearch.

    Designed to use with MovePickerPolicy (Stockfish-style ordering) and
    DepthPreferredTT. Use build_gm_engine() in gm_engine.py to assemble
    the full stack.
    """

    def __init__(
        self,
        evaluator,
        transposition_table,
        move_ordering_policy,
        pruning_policy,
        pruning_config=None,
    ) -> None:
        super().__init__(
            evaluator, transposition_table,
            move_ordering_policy, pruning_policy, pruning_config,
        )
        # Move-context stack: stores (piece_idx * 64 + to_sq) for each ply.
        # Index is self._ply at the time of the move. Pre-filled with sentinel.
        self._move_stack: list[int] = [CONT_HIST_SENTINEL] * 128

        # Cache optional side-channel methods from the ordering policy.
        pol = move_ordering_policy
        self._set_cont_keys      = getattr(pol, "set_cont_keys",         None)
        self._record_quiet_cut   = getattr(pol, "record_quiet_cutoff",   None)
        self._record_capture_cut = getattr(pol, "record_capture_cutoff", None)

    # ---------------------------------------------------------------------- #
    # find_best_move — reset move stack at the root
    # ---------------------------------------------------------------------- #

    def find_best_move(self, board, max_depth: int, allotted_time: float):
        for i in range(len(self._move_stack)):
            self._move_stack[i] = CONT_HIST_SENTINEL
        return super().find_best_move(board, max_depth, allotted_time)

    # ---------------------------------------------------------------------- #
    # Continuation history key helpers
    # ---------------------------------------------------------------------- #

    def _build_cont_keys(self, ply: int) -> tuple[int, ...]:
        """Build the 6-entry continuation-history key tuple for the current ply.

        Follows Stockfish 16: plies 1,2,3,4 back + sentinel at slot 4 + ply 6 back.
        (Ply 5 is intentionally skipped as in the Stockfish source.)
        """
        ms = self._move_stack
        def get(k: int) -> int:
            return ms[ply - k] if ply >= k else CONT_HIST_SENTINEL
        return (get(1), get(2), get(3), get(4), CONT_HIST_SENTINEL, get(6))

    # ---------------------------------------------------------------------- #
    # search_node — full replacement with PVS + check extension + cont hist
    # ---------------------------------------------------------------------- #

    def search_node(self, alpha: int, beta: int, depth: int) -> int:
        if depth == 0:
            return self.quiescence_search(alpha, beta)

        self.nodes_evaluated += 1
        if self.nodes_evaluated & (_TIME_CHECK_INTERVAL - 1) == 0 and self._time_exceeded():
            raise _SearchAborted()

        board       = self._board
        make_move   = board.make_move
        unmake_move = board.unmake_move
        is_in_check = board.is_king_in_check
        evaluate    = self.evaluator_reference.evaluate_position
        try_prune   = self.pruning_policy.try_prune
        order_moves = self.move_ordering_policy.order_moves
        tt          = self.transposition_table_reference
        cfg         = self.pruning_config

        EXACT = NodeType.EXACT
        LOWER = NodeType.LOWER_BOUND
        UPPER = NodeType.UPPER_BOUND

        # ── 1. CHECK EXTENSION ─────────────────────────────────────────────
        in_check = is_in_check()
        if in_check:
            depth += 1

        # ── 2. TRANSPOSITION TABLE PROBE ───────────────────────────────────
        zobrist       = board.calculate_zobrist_hash()
        original_alpha = alpha
        self.tt_lookups += 1
        tt_entry = tt.retrieve_position(zobrist)
        tt_best_move: object = None

        if tt_entry is not None and tt_entry.depth >= depth:
            self.tt_hits += 1
            tt_best_move = tt_entry.best_move
            if tt_entry.node_type == EXACT:
                return tt_entry.score
            elif tt_entry.node_type == LOWER:
                if tt_entry.score > alpha:
                    alpha = tt_entry.score
            else:
                if tt_entry.score < beta:
                    beta = tt_entry.score
            if alpha >= beta:
                return tt_entry.score

        # ── 3. FORWARD PRUNING ─────────────────────────────────────────────
        static_eval = evaluate(board)
        prune_score = try_prune(board, depth, alpha, beta, static_eval)
        if prune_score is not None:
            return prune_score

        # ── 4. MOVE GENERATION + ORDERING ──────────────────────────────────
        moves = board.generate_legal_moves()
        if not moves:
            return -(_MATE_SCORE - self._ply) if in_check else 0

        if self._set_ordering_depth is not None:
            self._set_ordering_depth(depth)
        if self._set_cont_keys is not None:
            self._set_cont_keys(self._build_cont_keys(self._ply))
        moves = order_moves(moves, board, tt_best_move)

        # ── 5. MAIN LOOP: fail-soft alpha-beta with PVS ────────────────────
        best_score: int = -_INF
        best_move        = None
        pv_searched      = False   # has the PV node been searched yet?

        for move_index, move in enumerate(moves):
            move_is_quiet = (not move.is_capture) and (move.promotion_piece is None)

            # Futility pruning (quiet moves at depth 1)
            if cfg is not None and move_is_quiet and _pruning.is_futile(
                depth, alpha, static_eval, in_check, move_is_quiet, cfg
            ):
                continue

            # Compute the continuation-history context key for this move BEFORE
            # making it (piece is still at from_square in the bitboards).
            bbs       = board.bitboards
            piece_idx = _get_piece_idx(bbs, move.from_square)
            cont_key  = (
                piece_idx * 64 + move.to_square
                if piece_idx >= 0
                else CONT_HIST_SENTINEL
            )
            self._move_stack[self._ply] = cont_key

            make_move(move)
            self._ply += 1
            gives_check = is_in_check()

            # ── PVS + LMR ──────────────────────────────────────────────────
            if not pv_searched:
                # First move: full-window search (the PV node).
                score = -self.search_node(-beta, -alpha, depth - 1)
                pv_searched = True

            elif cfg is not None and _pruning.should_apply_lmr(
                depth, move_index, move_is_quiet, in_check, gives_check, cfg
            ):
                # Late-move reduction: try a zero-window at reduced depth.
                r = _pruning.lmr_reduction(depth, move_index)
                score = -self.search_node(-alpha - 1, -alpha, max(1, depth - 1 - r))
                # Failed high at reduced depth → confirm at full depth (null window).
                if score > alpha:
                    score = -self.search_node(-alpha - 1, -alpha, depth - 1)
                # Still failed high → full-window re-search.
                if score > alpha and score < beta:
                    score = -self.search_node(-beta, -alpha, depth - 1)

            else:
                # PVS non-first move: zero-window probe, then full window on improvement.
                score = -self.search_node(-alpha - 1, -alpha, depth - 1)
                if score > alpha and score < beta:
                    score = -self.search_node(-beta, -alpha, depth - 1)

            self._ply -= 1
            unmake_move()

            if score > best_score:
                best_score = score
                best_move  = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                self.beta_cutoffs += 1
                self.killer_table.update(move, depth)
                self.update_history(move, depth)
                # Update continuation history and butterfly on beta cutoffs.
                if move_is_quiet:
                    if self._record_quiet_cut is not None:
                        self._record_quiet_cut(
                            board.side_to_move.value,
                            move.from_square,
                            move.to_square,
                            piece_idx,
                            depth,
                            self._build_cont_keys(self._ply),
                        )
                else:
                    if self._record_capture_cut is not None:
                        cap_type = _get_cap_type(bbs, move.to_square)
                        self._record_capture_cut(piece_idx, move.to_square, cap_type, depth)
                break

        # ── 6. TRANSPOSITION TABLE STORE ───────────────────────────────────
        node_type = (
            UPPER if best_score <= original_alpha else
            LOWER if best_score >= beta           else
            EXACT
        )
        tt.store_position(zobrist, depth, best_score, node_type, best_move)
        return best_score

    # ---------------------------------------------------------------------- #
    # quiescence_search — SEE pruning + delta pruning
    # ---------------------------------------------------------------------- #

    def quiescence_search(self, alpha: int, beta: int) -> int:
        self.nodes_evaluated += 1

        # Time guard in QSearch: NNUE evaluation is expensive; without this check
        # the QSearch can run for minutes on tactical positions.
        if self.nodes_evaluated & (_TIME_CHECK_INTERVAL - 1) == 0 and self._time_exceeded():
            raise _SearchAborted()

        board       = self._board
        make_move   = board.make_move
        unmake_move = board.unmake_move
        evaluate    = self.evaluator_reference.evaluate_position

        static_eval = evaluate(board)

        # Stand-pat: if the static score already beats beta, no need to search.
        if static_eval >= beta:
            return beta

        # ── DELTA PRUNING ──────────────────────────────────────────────────
        # If even capturing the queen + safety margin can't raise alpha, abort.
        if static_eval + _MAX_CAPTURE_GAIN < alpha:
            return alpha

        if static_eval > alpha:
            alpha = static_eval

        # Generate captures only, sorted by victim value (MVV) descending.
        # We sort directly rather than going through order_moves because
        # MovePickerPolicy.order_moves creates a fresh MovePicker that generates
        # ALL legal moves (including quiets), which would pollute the QSearch
        # and cause unbounded recursion.
        bbs = board.bitboards
        captures = sorted(
            (m for m in board.generate_legal_moves() if m.is_capture),
            key=lambda m: _get_cap_type(bbs, m.to_square),
            reverse=True,
        )
        if not captures:
            return alpha

        for move in captures:
            # ── SEE PRUNING ────────────────────────────────────────────────
            # Skip captures that lose material (SEE < 0) — they can't help.
            if not see_ge(board, move, 0):
                continue

            make_move(move)
            self._ply += 1
            score = -self.quiescence_search(-beta, -alpha)
            self._ply -= 1
            unmake_move()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha
