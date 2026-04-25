from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import chess
import chess.engine
import subprocess


@dataclass
class UciEngineClient:
    """Thin UCI client over python-chess engine support."""

    command: str | list[str]
    think_time: float = 0.2
    options: dict[str, Any] = field(default_factory=dict)
    _engine: chess.engine.SimpleEngine | None = field(init=False, default=None)

    def start(self) -> None:
        if self._engine is not None:
            return
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.command)
        except chess.engine.EngineTerminatedError as e:
            # python-chess doesn't surface stderr for startup failures; do a quick
            # best-effort probe so callers get a useful error message.
            probe = None
            try:
                probe = subprocess.run(
                    self.command if isinstance(self.command, list) else [self.command],
                    input="uci\nquit\n",
                    text=True,
                    capture_output=True,
                    timeout=2.0,
                )
            except Exception:
                probe = None

            detail = str(e)
            if probe and probe.stderr.strip():
                stderr_head = "\n".join(probe.stderr.strip().splitlines()[:8])
                detail = f"{detail}\n--- stderr ---\n{stderr_head}"
            raise RuntimeError(detail) from e
        if self.options:
            self._engine.configure(self.options)

    def stop(self) -> None:
        if self._engine is None:
            return
        try:
            self._engine.quit()
        except TimeoutError:
            # Some engines/wrappers may not respond to "quit" reliably.
            self._engine.close()
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

