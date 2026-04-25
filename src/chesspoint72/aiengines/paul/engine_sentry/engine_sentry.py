"""engine_sentry.py â€” Paul's Iron Sentry hybrid engine.

Pairs two NNUE specialists (nnue_tank for the middlegame, nnue_finisher for
the endgame) inside a uniformly conservative search shell. The Sentry's
personality is "never blunder": it sacrifices speed and tactical sharpness
for guaranteed depth coverage and risk-averse pruning.

Distinguishing features (each unique to this engine in Paul's suite):

  - **DualPhaseEvaluator**: routes evaluate_position() to nnue_tank when
    >12 non-king pieces remain on the board, otherwise nnue_finisher. Unlike
    Chameleon, this is the *only* phase-aware element here â€” the search
    behaviour stays uniform across phases.
  - **SafetyMoveOrdering**: TT-best > promotions > captures > quiet. The
    promotion-first ordering matters most when the finisher is steering us
    through K+P endings.
  - **sentry_pruning_config**: the most conservative profile in the suite.
    Razoring is fully disabled, NMP reductions are the minimum allowed
    (R=1 / R=2), futility margin is 75cp, LMR is delayed until move index 8.
  - **SentrySearch**: enforces a **minimum-depth guarantee** â€” the first 4
    plies are searched with the time budget temporarily set to infinity, so
    even under severe time pressure we never return a single-ply blunder.
    Above the floor, root-level PVS runs with the real budget.

CLI:
    python -m chesspoint72.aiengines.paul.engine_sentry
"""
from __future__ import annotations

import sys
import time
from typing import Iterable, TextIO

import chess

from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.evaluator import Evaluator
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

_TANK_WEIGHTS = WEIGHTS_DIR / "real_nnue_epoch_4.pt"
_FINISHER_WEIGHTS = WEIGHTS_DIR / "real_nnue_epoch_4.pt"
_INF = 10_000_000
_MIN_GUARANTEED_DEPTH = 4
# Below this many non-king pieces the finisher takes over.
_ENDGAME_THRESHOLD = 12


def _non_king_piece_count(board) -> int:
    py = getattr(board, "py_board", None)
    if not isinstance(py, chess.Board):
        py = chess.Board(board.get_current_fen())
    return chess.popcount(py.occupied) - 2


class DualPhaseEvaluator(Evaluator):
    """Routes evaluation to the tank in the middlegame, finisher in the endgame.

    The phase decision is made per-call so the search can transition smoothly
    as captures cross the threshold.
    """

    def __init__(self, tank_path, finisher_path) -> None:
        self._tank = NnueEvaluator(tank_path)
        self._finisher = NnueEvaluator(finisher_path)

    def evaluate_position(self, board) -> int:
        if _non_king_piece_count(board) <= _ENDGAME_THRESHOLD:
            return self._finisher.evaluate_position(board)
        return self._tank.evaluate_position(board)


class SafetyMoveOrdering(MoveOrderingPolicy):
    """TT-best > promotions > captures > quiet."""

    def order_moves(
        self,
        moves: list[Move],
        board,
        tt_best_move: Move | None = None,
    ) -> list[Move]:
        promotions: list[Move] = []
        captures: list[Move] = []
        rest: list[Move] = []
        tt_first: Move | None = None

        for move in moves:
            if (
                tt_best_move is not None
                and move.from_square == tt_best_move.from_square
                and move.to_square == tt_best_move.to_square
                and move.promotion_piece == tt_best_move.promotion_piece
            ):
                tt_first = move
            elif move.promotion_piece is not None:
                promotions.append(move)
            elif move.is_capture:
                captures.append(move)
            else:
                rest.append(move)

        ordered: list[Move] = []
        if tt_first is not None:
            ordered.append(tt_first)
        ordered.extend(promotions)
        ordered.extend(captures)
        ordered.extend(rest)
        return ordered


def sentry_pruning_config() -> PruningConfig:
    return PruningConfig(
        nmp_enabled=True,
        futility_enabled=True,
        razoring_enabled=False,
        lmr_enabled=True,
        nmp_r_shallow=1,
        nmp_r_deep=2,
        futility_margin=75,
        razoring_margins=(0, 0, 0),
        lmr_min_depth=4,
        lmr_min_move_index=8,
    )


class SentrySearch(NegamaxSearch):
    """Root-PVS search with a hard minimum-depth floor.

    Phase 1 (depths 1.._MIN_GUARANTEED_DEPTH): allotted_time set to +inf so
    in-search time checks never abort. Guarantees we always return a move
    with at least 4-ply consideration.

    Phase 2 (depths above the floor): real budget restored, behaves like a
    standard iterative-deepening search with PVS at the root.
    """

    def find_best_move(
        self,
        board,
        max_depth: int,
        allotted_time: float,
    ) -> Move:
        self._board = board
        self._start_time = time.monotonic()
        self.nodes_evaluated = 0
        self._ply = 0
        self.killer_table.clear()
        self.history_table.clear()

        best_move: Move | None = None
        floor = min(_MIN_GUARANTEED_DEPTH, max_depth)

        # Phase 1 â€” guaranteed completion.
        self._allotted_time = float("inf")
        for depth in range(1, floor + 1):
            candidate = self._root_search(depth)
            if candidate is not None:
                best_move = candidate

        # Phase 2 â€” real time budget.
        self._allotted_time = allotted_time
        for depth in range(floor + 1, max_depth + 1):
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

    def _root_search(self, depth: int) -> Move | None:
        """Root-level Principal Variation Search."""
        alpha = -_INF
        beta = _INF
        best_move: Move | None = None
        board = self._board

        zobrist = board.calculate_zobrist_hash()
        tt_entry = self.transposition_table_reference.retrieve_position(zobrist)
        tt_best_move = tt_entry.best_move if tt_entry is not None else None

        moves = board.generate_legal_moves()
        moves = self.move_ordering_policy.order_moves(moves, board, tt_best_move)

        for i, move in enumerate(moves):
            board.make_move(move)
            self._ply += 1
            if i == 0:
                score = -self.search_node(-beta, -alpha, depth - 1)
            else:
                score = -self.search_node(-alpha - 1, -alpha, depth - 1)
                if alpha < score < beta:
                    score = -self.search_node(-beta, -alpha, depth - 1)
            self._ply -= 1
            board.unmake_move()

            if score > alpha:
                alpha = score
                best_move = move

        return best_move


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    default_depth: int = 9,
    default_time: float = 12.0,
) -> StandardUciController:
    evaluator = DualPhaseEvaluator(_TANK_WEIGHTS, _FINISHER_WEIGHTS)
    board = PyChessBoard()
    cfg = sentry_pruning_config()
    pruning_policy = ForwardPruningPolicy(cfg)
    search = SentrySearch(
        evaluator,
        TranspositionTable(),
        SafetyMoveOrdering(),
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
    ctrl.engine_name = "Paul-Sentry"
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
