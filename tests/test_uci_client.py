from types import SimpleNamespace

import chess
import chess.engine

from chesspoint72.engine.uci.client import UciEngineClient


class DummyEngine:
    def __init__(self) -> None:
        self.configured = {}
        self.stopped = False

    def configure(self, options: dict[str, object]) -> None:
        self.configured = options

    def play(self, board: chess.Board, limit: chess.engine.Limit) -> SimpleNamespace:
        assert isinstance(board, chess.Board)
        assert limit.time == 0.5
        return SimpleNamespace(move=chess.Move.from_uci("e2e4"))

    def quit(self) -> None:
        self.stopped = True


def test_request_best_move_with_mocked_engine(monkeypatch) -> None:
    dummy_engine = DummyEngine()

    def fake_popen_uci(path: str) -> DummyEngine:
        assert path == "/tmp/fake-engine"
        return dummy_engine

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", fake_popen_uci)

    client = UciEngineClient("/tmp/fake-engine", think_time=0.5, options={"Skill Level": 5})
    client.start()
    move = client.request_best_move(chess.Board())
    client.stop()

    assert move == chess.Move.from_uci("e2e4")
    assert dummy_engine.configured == {"Skill Level": 5}
    assert dummy_engine.stopped

