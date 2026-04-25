"""Frank v2 — strongest-engine configuration for chesspoint72.

Component choices
-----------------
* Evaluator     : NnueEvaluator — neural network beats hand-crafted heuristics
                  at any fixed node budget; the weights are already shipped.
* Move ordering : FrankV2OrderingPolicy — SEE-based capture bucketing (better
                  than MVV-LVA) combined with killer moves and history quiets.
* Search        : FrankV2Search (NegamaxSearch subclass) — shares its killer
                  and history tables with the ordering policy so beta-cutoff
                  feedback flows back into move ordering automatically.
* Pruning       : ForwardPruningPolicy with default config — all four techniques
                  enabled (NMP, Razoring, Futility, LMR).
* TT            : 256 MB — larger cache improves hit rate at tournament depths.

Usage
-----
    from chesspoint72.aiengines.frank.v2 import build_frank_v2
    ctrl = build_frank_v2()
    ctrl.start_listening_loop()
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from chesspoint72.engine.core.policies import MoveOrderingPolicy
from chesspoint72.engine.core.transposition import TranspositionTable
from chesspoint72.engine.core.types import PieceType
from chesspoint72.engine.evaluators.nnue import NnueEvaluator
from chesspoint72.engine.ordering.heuristics import HistoryTable, KillerMoveTable
from chesspoint72.engine.ordering.see import SEE_VALUES, see_ge
from chesspoint72.engine.pruning.config import default_pruning_config
from chesspoint72.engine.pruning.policy import ForwardPruningPolicy
from chesspoint72.engine.search.negamax.negamax import NegamaxSearch

if TYPE_CHECKING:
    from chesspoint72.engine.core.board import Board
    from chesspoint72.engine.core.types import Move

# Score buckets for move ordering tiers
_TT_SCORE = 1_000_000
_GOOD_CAPTURE_BASE = 800_000
_KILLER0_SCORE = 500_000
_KILLER1_SCORE = 499_999
_BAD_CAPTURE_BASE = -200_000


class FrankV2OrderingPolicy(MoveOrderingPolicy):
    """SEE-based move ordering with killer moves and history quiets.

    Capture tiers:
      good captures (SEE >= 0) : 800_000 + SEE_VALUES[victim]
      bad  captures (SEE <  0) : -200_000 + SEE_VALUES[victim]

    Quiet tiers (between good and bad captures):
      killer 0 : 500_000
      killer 1 : 499_999
      history  : accumulated depth**2 bonus from beta-cutoffs
    """

    def __init__(self) -> None:
        # Tables are replaced by FrankV2Search after construction so that both
        # share the exact same instances; the initial dummies are never used.
        self.killer_table: KillerMoveTable = KillerMoveTable()
        self.history_table: HistoryTable = HistoryTable()
        # Depth hint set by FrankV2Search before each order_moves call so killers
        # can be looked up at the right depth.
        self._current_depth: int = 0

    def set_depth(self, depth: int) -> None:
        self._current_depth = depth

    def order_moves(
        self,
        moves: list[Move],
        board: Board,
        tt_best_move: Move | None = None,
    ) -> list[Move]:
        color = board.side_to_move
        depth = self._current_depth
        killers = self.killer_table.get(depth)
        history_get = self.history_table.get

        scored: list[tuple[int, Move]] = []
        for move in moves:
            if tt_best_move is not None and move == tt_best_move:
                score = _TT_SCORE
            elif move.is_capture:
                piece_info = board.get_piece_at(move.to_square)
                victim_val = SEE_VALUES[piece_info[0].value] if piece_info else 0
                if see_ge(board, move, 0):
                    score = _GOOD_CAPTURE_BASE + victim_val
                else:
                    score = _BAD_CAPTURE_BASE + victim_val
            elif move == killers[0]:
                score = _KILLER0_SCORE
            elif move == killers[1]:
                score = _KILLER1_SCORE
            else:
                score = history_get(color, move.from_square, move.to_square)
            scored.append((score, move))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]


class FrankV2Search(NegamaxSearch):
    """NegamaxSearch with shared tables wired into FrankV2OrderingPolicy.

    The parent creates ``killer_table`` and ``history_table`` in its
    ``__init__``; we replace the policy's placeholder instances with those
    exact objects so every beta-cutoff update immediately affects move ordering.
    """

    def __init__(
        self,
        evaluator: NnueEvaluator,
        transposition_table: TranspositionTable,
        ordering_policy: FrankV2OrderingPolicy,
        pruning_policy: ForwardPruningPolicy,
        pruning_config,
    ) -> None:
        super().__init__(
            evaluator,
            transposition_table,
            ordering_policy,
            pruning_policy,
            pruning_config,
        )
        # Share the search's own tables with the ordering policy.
        ordering_policy.killer_table = self.killer_table
        ordering_policy.history_table = self.history_table

    def search_node(self, alpha: int, beta: int, depth: int) -> int:
        # Keep the ordering policy in sync with the current search depth so
        # killer lookups target the right depth.
        self.move_ordering_policy.set_depth(depth)  # type: ignore[attr-defined]
        return super().search_node(alpha, beta, depth)


def build_frank_v2(
    default_depth: int = 8,
    default_time: float = 5.0,
) -> object:
    """Construct a fully-wired Frank v2 UCI controller.

    Returns a ``StandardUciController`` ready to call
    ``start_listening_loop()`` or to drive programmatically.
    """
    # Import here to avoid circular imports at module level.
    from chesspoint72.engine.boards.pychess import PyChessBoard
    from chesspoint72.engine.factory import StandardUciController

    evaluator = NnueEvaluator()
    tt = TranspositionTable(max_memory_size=256)
    ordering_policy = FrankV2OrderingPolicy()
    pruning_cfg = default_pruning_config()
    pruning_policy = ForwardPruningPolicy(pruning_cfg)
    search = FrankV2Search(evaluator, tt, ordering_policy, pruning_policy, pruning_cfg)
    board = PyChessBoard()

    return StandardUciController(
        board=board,
        search=search,
        evaluator=evaluator,
        default_depth=default_depth,
        default_time=default_time,
    )
