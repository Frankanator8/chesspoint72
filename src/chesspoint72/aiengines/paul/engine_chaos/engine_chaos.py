"""engine_chaos.py â€” Paul's Chaos engine.

Pairs the nnue_tactician (256x32, tactical specialist) with aspiration windows
and aggressive Null Move Pruning. Aspiration windows exploit the fact that
the score rarely changes drastically between depths â€” the search opens with a
tight window [prev - 50, prev + 50] and widens exponentially only on failure.
Aggressive NMP (R=3 even at shallow depths) enables deep null-move probes that
quickly prune calm branches, leaving more budget for sharp tactical lines.

CLI:
    python -m chesspoint72.aiengines.paul.engine_chaos
"""
from __future__ import annotations

import sys
import time
from typing import Iterable, TextIO

from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.transposition import TranspositionTable
from chesspoint72.engine.core.types import Move
from chesspoint72.engine.evaluators.nnue import NnueEvaluator
from chesspoint72.engine.factory import StandardUciController
from chesspoint72.engine.pruning import ForwardPruningPolicy
from chesspoint72.engine.pruning.config import PruningConfig
from chesspoint72.engine.search.negamax import NegamaxSearch
from chesspoint72.engine.search.negamax.negamax import _SearchAborted

from .._common import WEIGHTS_DIR, PassthroughOrdering

_WEIGHTS = WEIGHTS_DIR / "nnue_tactician_final.pt"
_INF = 10_000_000
_ASPIRATION_DELTA = 50


class ChaosSearch(NegamaxSearch):
    """NegamaxSearch with iterative-deepening aspiration windows.

    Depth 1 is always a full-window search to establish a reliable baseline
    score. For depth 2+, the root opens with a Â±50cp window around the
    previous depth's score. Failed high/low results widen the window
    exponentially (delta doubles each time) and re-search, falling back to
    full-window after 4 expansions.
    """

    _last_score: int = 0

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
        """Full-window root search that stores the final alpha in _last_score."""
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
                # Score fell inside the window.
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


def chaos_pruning_config() -> PruningConfig:
    return PruningConfig(
        nmp_enabled=True,
        futility_enabled=True,
        razoring_enabled=True,
        lmr_enabled=True,
        nmp_r_shallow=3,   # deeper null-move reduction even at low depth
        nmp_r_deep=4,
        futility_margin=300,
        razoring_margins=(350, 450, 550),
        lmr_min_depth=3,
        lmr_min_move_index=3,
    )


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    default_depth: int = 6,
    default_time: float = 5.0,
) -> StandardUciController:
    evaluator = NnueEvaluator(_WEIGHTS)
    board = PyChessBoard()
    cfg = chaos_pruning_config()
    pruning_policy = ForwardPruningPolicy(cfg)
    search = ChaosSearch(
        evaluator,
        TranspositionTable(),
        PassthroughOrdering(),
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
    ctrl.engine_name = "Paul-Chaos"
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
