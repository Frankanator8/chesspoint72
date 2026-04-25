from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import chess
import chess.engine


@dataclass
class UciEngineClient:
    """Thin UCI client over python-chess engine support."""

    engine_path: str
    think_time: float = 0.2
    options: dict[str, Any] = field(default_factory=dict)
    _engine: chess.engine.SimpleEngine | None = field(init=False, default=None)

    def start(self) -> None:
        if self._engine is not None:
            return
        self._engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        if self.options:
            self._engine.configure(self.options)

    def stop(self) -> None:
        if self._engine is None:
            return
        self._engine.quit()
        self._engine = None

    def request_best_move(self, board: chess.Board) -> chess.Move:
        if self._engine is None:
            raise RuntimeError("UCI engine is not started")
        result = self._engine.play(board, chess.engine.Limit(time=self.think_time))
        return result.move

    def __enter__(self) -> "UciEngineClient":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

