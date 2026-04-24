import chess

from chesspoint72.models import GameState


def test_push_uci_legal_and_illegal() -> None:
    state = GameState()
    assert state.push_uci("e2e4")
    assert state.board.piece_at(chess.E4)
    assert not state.push_uci("e2e5")


def test_result_for_active_game() -> None:
    state = GameState()
    assert state.result() == "*"

