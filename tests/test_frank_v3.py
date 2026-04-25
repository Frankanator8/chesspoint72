from __future__ import annotations

from chesspoint72.aiengines.frank.v3.engine import build_controller, build_frank_v3_evaluator
from chesspoint72.aiengines.frank.v3.ordering import FrankV3MoveOrderingPolicy
from chesspoint72.engine.boards import PyChessBoard


def test_frank_v3_builds_controller() -> None:
    controller = build_controller(prefer_nnue=False, hce_modules="material,pst")
    assert controller is not None


def test_frank_v3_evaluator_hce_fallback_is_usable() -> None:
    evaluator = build_frank_v3_evaluator(prefer_nnue=False, hce_modules="material,pst")
    board = PyChessBoard()
    score = evaluator.evaluate_position(board)
    assert isinstance(score, int)


def test_frank_v3_ordering_prefers_tt_move() -> None:
    board = PyChessBoard()
    moves = board.generate_legal_moves()
    tt_move = next(m for m in moves if m.to_uci_string() == "e2e4")

    policy = FrankV3MoveOrderingPolicy()
    ordered = policy.order_moves(moves, board, tt_move)

    assert ordered[0].to_uci_string() == "e2e4"

