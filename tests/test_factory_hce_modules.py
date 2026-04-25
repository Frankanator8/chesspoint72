from __future__ import annotations

import chess
import pytest

from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.factory import _parse_cli, build_evaluator


def test_parse_cli_reads_hce_modules() -> None:
    evaluator, hce_modules, depth, move_time = _parse_cli(
        ["--evaluator", "hce", "--hce-modules", "classic,ewpm", "--depth", "5", "--time", "0.7"]
    )
    assert evaluator == "hce"
    assert hce_modules == "classic,ewpm"
    assert depth == 5
    assert move_time == pytest.approx(0.7)


def test_build_hce_evaluator_accepts_multiple_modules() -> None:
    evaluator = build_evaluator("hce", "material,pst,mobility")
    board = PyChessBoard()
    score = evaluator.evaluate_position(board)
    assert isinstance(score, int)

    board.set_position_from_fen(chess.STARTING_FEN.replace(" w ", " b "))
    score_black_to_move = evaluator.evaluate_position(board)
    assert isinstance(score_black_to_move, int)


def test_build_hce_evaluator_accepts_module_groups() -> None:
    evaluator = build_evaluator("hce", "classic,advanced")
    board = PyChessBoard()
    score = evaluator.evaluate_position(board)
    assert isinstance(score, int)


def test_build_hce_evaluator_rejects_unknown_module() -> None:
    with pytest.raises(ValueError, match="unknown hce module"):
        build_evaluator("hce", "material,not_a_module")

