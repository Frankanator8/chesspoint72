"""engine_grinder.py â€” Paul's Grinder engine.

Pairs the nnue_tank (512x64, positional specialist) with Principal Variation
Search (PVS) at the root and conservative pruning margins. The idea: the
large tank network produces highly accurate positional scores, and PVS
narrows the search window for all moves after the first, cutting re-searches
while preserving the evaluator's accuracy. Conservative pruning (tight margins,
late LMR start) means we prune only when we're very confident, trading speed
for correctness. Default depth 8 because the tank can afford the depth.

CLI:
    python -m chesspoint72.aiengines.paul.engine_grinder
"""
from __future__ import annotations

import sys
from typing import Iterable, TextIO

from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.transposition import TranspositionTable
from chesspoint72.engine.core.types import Move, NodeType
from chesspoint72.engine.evaluators.nnue import NnueEvaluator
from chesspoint72.engine.factory import StandardUciController
from chesspoint72.engine.pruning import ForwardPruningPolicy
from chesspoint72.engine.pruning.config import PruningConfig
from chesspoint72.engine.search.negamax import NegamaxSearch

from .._common import WEIGHTS_DIR, PassthroughOrdering

_WEIGHTS = WEIGHTS_DIR / "real_nnue_epoch_4.pt"
_INF = 10_000_000


class GrinderSearch(NegamaxSearch):
    """NegamaxSearch with PVS at the root node.

    After the first (presumably best-ordered) move is searched with a full
    window, every subsequent root move uses a null window (-alpha-1, -alpha).
    A re-search with the full window is only triggered on a score improvement,
    saving significant work when the first move is indeed the best.
    """

    def _root_search(self, depth: int) -> Move | None:
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
                # Null-window probe; re-search only if it beats alpha.
                score = -self.search_node(-alpha - 1, -alpha, depth - 1)
                if alpha < score < beta:
                    score = -self.search_node(-beta, -alpha, depth - 1)
            self._ply -= 1
            board.unmake_move()

            if score > alpha:
                alpha = score
                best_move = move

        return best_move


def grinder_pruning_config() -> PruningConfig:
    return PruningConfig(
        nmp_enabled=True,
        futility_enabled=True,
        razoring_enabled=True,
        lmr_enabled=True,
        nmp_r_shallow=2,
        nmp_r_deep=3,
        futility_margin=150,
        razoring_margins=(200, 300, 400),
        lmr_min_depth=3,
        lmr_min_move_index=5,  # start LMR later than default (3)
    )


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    default_depth: int = 8,
    default_time: float = 10.0,
) -> StandardUciController:
    evaluator = NnueEvaluator(_WEIGHTS)
    board = PyChessBoard()
    cfg = grinder_pruning_config()
    pruning_policy = ForwardPruningPolicy(cfg)
    search = GrinderSearch(
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
    ctrl.engine_name = "Paul-Grinder"
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
