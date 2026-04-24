import chess

from chesspoint72.app.controller import GameConfig, GameController


def test_build_move_auto_promotes_to_queen() -> None:
    board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    move = GameController.build_move(chess.A7, chess.A8, board)
    assert move.promotion == chess.QUEEN


def test_human_turn_without_engine() -> None:
    controller = GameController(GameConfig(engine_path=None))
    assert controller._is_human_turn()


def test_human_turn_with_black_engine() -> None:
    controller = GameController(GameConfig(engine_path="/tmp/engine", engine_color=chess.BLACK))
    assert controller._is_human_turn()
    controller.game_state.push_uci("e2e4")
    assert not controller._is_human_turn()

