"""engine_cannon.py â€” Paul's Glass Cannon hybrid engine.

Pairs the nnue_tactician (256x32 tactical specialist) with a search built
for tactical depth at the cost of positional safety. The "glass" half of the
name comes from the aggression: any single calculation error is unrecoverable.

Distinguishing features (each unique to this engine in Paul's suite):

  - **CaptureFirstOrdering**: captures sorted by victim piece value
    (queen > rook > bishop/knight > pawn) using direct lookup on
    board.py_board, then promotions, then quiet. Maximises beta-cutoffs in
    forcing sequences where the tactician has been trained to excel.
  - **CannonSearch**: two architectural pieces unique to this engine â€”
      1. **Narrow aspiration windows (Î”=30)**: tighter than Chaos's Î”=50.
         The tactician's eval is sharply peaked on tactical positions, so
         a tighter window gives more aggressive cut-offs and re-searches
         only when the score genuinely diverges.
      2. **Quiescence-with-checks (first ply only)**: standard qsearch only
         considers captures, missing forced-mate sequences. We extend the
         first qsearch ply to also include checking moves, then revert to
         captures-only at deeper plies (depth tracked via _qs_depth counter).
  - **cannon_pruning_config**: ultra-aggressive â€” razoring disabled (its
    QS calls are too slow for a deep aspiration search), NMP R=4/5 (vs the
    default 2/3), futility margin 500cp, LMR from the second move at depth 2.

Default depth 6 / time 3s â€” tuned for short-clock tactical knockouts.

CLI:
    python -m chesspoint72.aiengines.paul.engine_cannon
"""
from __future__ import annotations

import sys
import time
from typing import Iterable, TextIO

import chess

from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.policies import MoveOrderingPolicy
from chesspoint72.engine.core.transposition import TranspositionTable
from chesspoint72.engine.core.types import Move
from chesspoint72.engine.evaluators.nnue import NnueEvaluator
from chesspoint72.engine.factory import StandardUciController
from chesspoint72.engine.pruning import ForwardPruningPolicy
from chesspoint72.engine.pruning.config import PruningConfig
from chesspoint72.engine.search.negamax import NegamaxSearch
from chesspoint72.engine.search.negamax.negamax import _SearchAborted

from .._common import WEIGHTS_DIR

_WEIGHTS = WEIGHTS_DIR / "real_nnue_epoch_2.pt"
_INF = 10_000_000
_ASPIRATION_DELTA = 30  # tighter than Chaos's 50 â€” the tactician justifies it

_PIECE_VALUE = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   10_000,  # captures of kings shouldn't appear; keeps ordering monotone
}


def _py_board(board) -> chess.Board | None:
    py = getattr(board, "py_board", None)
    return py if isinstance(py, chess.Board) else None


class CaptureFirstOrdering(MoveOrderingPolicy):
    """TT-best > captures (by victim value desc) > promotions > quiet."""

    def order_moves(
        self,
        moves: list[Move],
        board,
        tt_best_move: Move | None = None,
    ) -> list[Move]:
        py = _py_board(board)
        captures: list[tuple[int, Move]] = []
        promotions: list[Move] = []
        quiet: list[Move] = []
        tt_first: Move | None = None

        for move in moves:
            if (
                tt_best_move is not None
                and move.from_square == tt_best_move.from_square
                and move.to_square == tt_best_move.to_square
                and move.promotion_piece == tt_best_move.promotion_piece
            ):
                tt_first = move
                continue
            if move.is_capture:
                victim_value = 100  # default for en-passant pawn or unknown board
                if py is not None:
                    victim = py.piece_at(move.to_square)
                    if victim is not None:
                        victim_value = _PIECE_VALUE.get(victim.piece_type, 100)
                captures.append((victim_value, move))
            elif move.promotion_piece is not None:
                promotions.append(move)
            else:
                quiet.append(move)

        captures.sort(key=lambda kv: kv[0], reverse=True)
        ordered: list[Move] = []
        if tt_first is not None:
            ordered.append(tt_first)
        ordered.extend(m for _, m in captures)
        ordered.extend(promotions)
        ordered.extend(quiet)
        return ordered


def cannon_pruning_config() -> PruningConfig:
    return PruningConfig(
        nmp_enabled=True,
        futility_enabled=True,
        razoring_enabled=False,
        lmr_enabled=True,
        nmp_r_shallow=4,
        nmp_r_deep=5,
        futility_margin=500,
        razoring_margins=(0, 0, 0),  # unused
        lmr_min_depth=2,
        lmr_min_move_index=1,
    )


class CannonSearch(NegamaxSearch):
    """Tight aspiration windows + first-ply qsearch checks.

    Aspiration: depth 1 is full-window (establishes a baseline score). For
    depth >= 2 the root opens with [prev - 30, prev + 30]. Failed-low/high
    doubles delta and re-searches up to 4 times before falling back to a
    full window.

    Qsearch-with-checks: a per-call counter (_qs_depth) tracks how many
    levels deep we are inside qsearch. At the top level we generate captures
    AND checking moves; deeper levels are captures-only (preventing
    pathological check-chains).
    """

    _last_score: int = 0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Tracks recursion depth INSIDE quiescence_search. 0 means we are
        # entering qsearch fresh from search_node (first ply).
        self._qs_depth = 0

    # ------------------------------------------------------------------ #
    # find_best_move + aspiration root
    # ------------------------------------------------------------------ #

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
        self._qs_depth = 0

        best_move: Move | None = None

        for depth in range(1, max_depth + 1):
            if self._time_exceeded():
                break
            try:
                if depth == 1:
                    candidate = self._root_search_scored(depth)
                else:
                    candidate = self._aspiration_root(depth, self._last_score)
            except _SearchAborted:
                break
            if candidate is not None:
                best_move = candidate

        if best_move is None:
            legal = board.generate_legal_moves()
            if legal:
                best_move = legal[0]
        return best_move  # type: ignore[return-value]

    def _root_search_scored(self, depth: int) -> Move | None:
        alpha = -_INF
        beta = _INF
        best_move: Move | None = None
        board = self._board

        zobrist = board.calculate_zobrist_hash()
        tt_entry = self.transposition_table_reference.retrieve_position(zobrist)
        tt_best_move = tt_entry.best_move if tt_entry is not None else None

        moves = board.generate_legal_moves()
        moves = self.move_ordering_policy.order_moves(moves, board, tt_best_move)

        for move in moves:
            board.make_move(move)
            self._ply += 1
            score = -self.search_node(-beta, -alpha, depth - 1)
            self._ply -= 1
            board.unmake_move()
            if score > alpha:
                alpha = score
                best_move = move

        self._last_score = alpha
        return best_move

    def _aspiration_root(self, depth: int, center: int) -> Move | None:
        delta = _ASPIRATION_DELTA
        lo = center - delta
        hi = center + delta
        board = self._board
        best_move: Move | None = None
        best_alpha = lo

        for _ in range(4):
            alpha = lo
            candidate: Move | None = None

            zobrist = board.calculate_zobrist_hash()
            tt_entry = self.transposition_table_reference.retrieve_position(zobrist)
            tt_best_move = tt_entry.best_move if tt_entry is not None else None

            moves = board.generate_legal_moves()
            moves = self.move_ordering_policy.order_moves(moves, board, tt_best_move)

            for move in moves:
                board.make_move(move)
                self._ply += 1
                score = -self.search_node(-hi, -alpha, depth - 1)
                self._ply -= 1
                board.unmake_move()
                if score > alpha:
                    alpha = score
                    candidate = move
                if alpha >= hi:
                    break

            if candidate is not None:
                best_move = candidate
                best_alpha = alpha

            if lo < alpha < hi:
                self._last_score = alpha
                return best_move

            if alpha <= lo:
                delta *= 2
                lo = max(-_INF, center - delta)
            else:
                delta *= 2
                hi = min(_INF, center + delta)

            if lo <= -_INF and hi >= _INF:
                break

        self._last_score = best_alpha
        return best_move

    # ------------------------------------------------------------------ #
    # quiescence_search override (with checks at first ply)
    # ------------------------------------------------------------------ #

    def quiescence_search(self, alpha: int, beta: int) -> int:
        self.nodes_evaluated += 1
        is_first_ply = (self._qs_depth == 0)
        self._qs_depth += 1
        try:
            board = self._board
            py = _py_board(board)

            static_eval = self.evaluator_reference.evaluate_position(board)
            if static_eval >= beta:
                return beta
            if static_eval > alpha:
                alpha = static_eval

            all_moves = board.generate_legal_moves()
            forcing: list[Move] = []
            if is_first_ply and py is not None:
                for move in all_moves:
                    if move.is_capture:
                        forcing.append(move)
                        continue
                    py_move = chess.Move(
                        move.from_square,
                        move.to_square,
                        promotion=int(move.promotion_piece) if move.promotion_piece is not None else None,
                    )
                    if py.gives_check(py_move):
                        forcing.append(move)
            else:
                forcing = [m for m in all_moves if m.is_capture]

            if not forcing:
                return alpha

            forcing = self.move_ordering_policy.order_moves(forcing, board)

            for move in forcing:
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
        finally:
            self._qs_depth -= 1


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    default_depth: int = 6,
    default_time: float = 3.0,
) -> StandardUciController:
    evaluator = NnueEvaluator(_WEIGHTS)
    board = PyChessBoard()
    cfg = cannon_pruning_config()
    pruning_policy = ForwardPruningPolicy(cfg)
    search = CannonSearch(
        evaluator,
        TranspositionTable(),
        CaptureFirstOrdering(),
        pruning_policy,
        cfg,
    )
    ctrl = StandardUciController(
        board=board,
        search=search,
        input_stream=input_stream,
        output_stream=output_stream,
        evaluator=evaluator,
        default_depth=default_depth,
        default_time=default_time,
    )
    ctrl.engine_name = "Paul-Cannon"
    return ctrl


def main() -> int:
    ctrl = build_controller()
    try:
        ctrl.start_listening_loop()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
