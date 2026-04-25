"""engine_chameleon.py â€” Paul's Chameleon hybrid engine.

The flagship of Paul's suite: a fully phase-adaptive engine. Three things
swap together at phase transitions, giving the engine two distinct
"personalities" for middlegame and endgame play:

  1. **Evaluator** â€” PhaseSwitchingEvaluator routes to nnue_tank when
     non-king piece count > 12, otherwise nnue_finisher. The swap is
     evaluated per node so the transition is smooth.

  2. **Move ordering** â€” PhaseAwareOrdering consults the same threshold and
     applies a different priority list for each phase:
        - Middlegame: TT-best > captures > quiet
        - Endgame:    TT-best > pawn pushes > captures > king moves > quiet
     The pawn-push priority is critical for endgame conversion (rook + pawn
     endings, K + P races) and would slow middlegame search if applied there.

  3. **Pruning policy** â€” two ForwardPruningPolicy instances, both attached
     to the search at construction time, swapped at the start of each
     find_best_move call:
        - Middlegame: balanced (default-style) margins
        - Endgame:    NMP DISABLED (zugzwang risk in K+P endings is real),
                      tighter futility/razoring margins, LMR delayed

Compared to the Sentry â€” which also pairs tank+finisher â€” the Chameleon is
the *aggressive* version: middlegame play uses balanced (not ultra-conservative)
pruning, and the engine actually changes its move-ordering personality
between phases instead of holding a single safety-first profile.

CLI:
    python -m chesspoint72.aiengines.paul.engine_chameleon
"""
from __future__ import annotations

import sys
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

from .._common import WEIGHTS_DIR

_TANK_WEIGHTS = WEIGHTS_DIR / "nnue_tank_final.pt"
_FINISHER_WEIGHTS = WEIGHTS_DIR / "nnue_finisher_final.pt"

# Below this many non-king pieces we are in endgame territory.
_ENDGAME_THRESHOLD = 12


def _non_king_piece_count(board) -> int:
    py = getattr(board, "py_board", None)
    if not isinstance(py, chess.Board):
        py = chess.Board(board.get_current_fen())
    return chess.popcount(py.occupied) - 2


def _is_endgame(board) -> bool:
    return _non_king_piece_count(board) <= _ENDGAME_THRESHOLD


# ---------------------------------------------------------------------- #
# Evaluator: per-node phase-switched NNUE
# ---------------------------------------------------------------------- #


class PhaseSwitchingEvaluator(Evaluator):
    """Routes to nnue_tank in middlegame, nnue_finisher in endgame."""

    def __init__(self, tank_path, finisher_path) -> None:
        self._tank = NnueEvaluator(tank_path)
        self._finisher = NnueEvaluator(finisher_path)

    def evaluate_position(self, board) -> int:
        if _is_endgame(board):
            return self._finisher.evaluate_position(board)
        return self._tank.evaluate_position(board)


# ---------------------------------------------------------------------- #
# Move ordering: phase-aware priority lists
# ---------------------------------------------------------------------- #


class PhaseAwareOrdering(MoveOrderingPolicy):
    """Different priority list per phase.

    Middlegame: TT-best > captures > quiet.
    Endgame:    TT-best > pawn pushes > captures > king moves > quiet.
    """

    def order_moves(
        self,
        moves: list[Move],
        board,
        tt_best_move: Move | None = None,
    ) -> list[Move]:
        py = getattr(board, "py_board", None)
        endgame = _is_endgame(board)

        tt_first: Move | None = None
        captures: list[Move] = []
        pawn_pushes: list[Move] = []
        king_moves: list[Move] = []
        quiet: list[Move] = []

        for move in moves:
            if (
                tt_best_move is not None
                and move.from_square == tt_best_move.from_square
                and move.to_square == tt_best_move.to_square
                and move.promotion_piece == tt_best_move.promotion_piece
            ):
                tt_first = move
                continue
            if move.is_capture or move.promotion_piece is not None:
                captures.append(move)
                continue
            if endgame and isinstance(py, chess.Board):
                piece = py.piece_at(move.from_square)
                if piece is not None:
                    if piece.piece_type == chess.PAWN:
                        pawn_pushes.append(move)
                        continue
                    if piece.piece_type == chess.KING:
                        king_moves.append(move)
                        continue
            quiet.append(move)

        ordered: list[Move] = []
        if tt_first is not None:
            ordered.append(tt_first)
        if endgame:
            ordered.extend(pawn_pushes)
            ordered.extend(captures)
            ordered.extend(king_moves)
            ordered.extend(quiet)
        else:
            ordered.extend(captures)
            ordered.extend(quiet)
        return ordered


# ---------------------------------------------------------------------- #
# Pruning: two configs swapped at the root per move
# ---------------------------------------------------------------------- #


def chameleon_middlegame_config() -> PruningConfig:
    return PruningConfig(
        nmp_enabled=True,
        futility_enabled=True,
        razoring_enabled=True,
        lmr_enabled=True,
        nmp_r_shallow=2,
        nmp_r_deep=3,
        futility_margin=300,
        razoring_margins=(350, 450, 550),
        lmr_min_depth=3,
        lmr_min_move_index=3,
    )


def chameleon_endgame_config() -> PruningConfig:
    """NMP disabled (zugzwang risk in K+P endings); tighter margins."""
    return PruningConfig(
        nmp_enabled=False,
        futility_enabled=True,
        razoring_enabled=True,
        lmr_enabled=True,
        nmp_r_shallow=2,
        nmp_r_deep=3,
        futility_margin=200,
        razoring_margins=(250, 350, 450),
        lmr_min_depth=4,
        lmr_min_move_index=5,
    )


# ---------------------------------------------------------------------- #
# Search: swaps pruning policy at the root of each find_best_move call
# ---------------------------------------------------------------------- #


class ChameleonSearch(NegamaxSearch):
    """At the start of every find_best_move call, swap pruning_policy and
    pruning_config based on the position's phase. Both policies are attached
    to this search instance up-front so the swap is a pure pointer swap.
    """

    def __init__(
        self,
        evaluator,
        transposition_table,
        move_ordering_policy,
        middlegame_policy: ForwardPruningPolicy,
        middlegame_config: PruningConfig,
        endgame_policy: ForwardPruningPolicy,
        endgame_config: PruningConfig,
    ) -> None:
        # Bootstrap with middlegame policy; find_best_move overwrites per call.
        super().__init__(
            evaluator,
            transposition_table,
            move_ordering_policy,
            middlegame_policy,
            middlegame_config,
        )
        self._mg_policy = middlegame_policy
        self._mg_config = middlegame_config
        self._eg_policy = endgame_policy
        self._eg_config = endgame_config
        # Pre-attach BOTH policies â€” the parent only attached the bootstrap.
        if hasattr(self._eg_policy, "attach_search"):
            self._eg_policy.attach_search(self)

    def find_best_move(
        self,
        board,
        max_depth: int,
        allotted_time: float,
    ) -> Move:
        if _is_endgame(board):
            self.pruning_policy = self._eg_policy
            self.pruning_config = self._eg_config
        else:
            self.pruning_policy = self._mg_policy
            self.pruning_config = self._mg_config
        return super().find_best_move(board, max_depth, allotted_time)


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    default_depth: int = 7,
    default_time: float = 8.0,
) -> StandardUciController:
    evaluator = PhaseSwitchingEvaluator(_TANK_WEIGHTS, _FINISHER_WEIGHTS)
    board = PyChessBoard()
    mg_cfg = chameleon_middlegame_config()
    eg_cfg = chameleon_endgame_config()
    mg_policy = ForwardPruningPolicy(mg_cfg)
    eg_policy = ForwardPruningPolicy(eg_cfg)
    search = ChameleonSearch(
        evaluator,
        TranspositionTable(),
        PhaseAwareOrdering(),
        mg_policy, mg_cfg,
        eg_policy, eg_cfg,
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
    ctrl.engine_name = "Paul-Chameleon"
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
