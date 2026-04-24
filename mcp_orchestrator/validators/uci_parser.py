"""UCI engine subprocess wrapper.

Handles spawning an engine binary, performing the UCI handshake, and
exchanging line-delimited UCI messages. All engine I/O flows through this
module so higher-level tools (perft, SPRT, tactics) stay protocol-agnostic.
"""
from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional


class UCIError(RuntimeError):
    """Raised when the engine misbehaves or times out."""


@dataclass
class UCIEngine:
    """A running UCI engine process with line-buffered stdio."""

    path: str
    process: subprocess.Popen
    _stdout_q: "queue.Queue[str]"
    _reader_thread: threading.Thread

    @classmethod
    def spawn(cls, path: str, cwd: Optional[str] = None) -> "UCIEngine":
        if not os.path.isfile(path):
            raise UCIError(f"engine binary not found: {path}")

        proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd or os.path.dirname(path) or None,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        q: "queue.Queue[str]" = queue.Queue()

        def _pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                q.put(line.rstrip("\r\n"))
            q.put("")  # sentinel on EOF

        t = threading.Thread(target=_pump, daemon=True)
        t.start()
        return cls(path=path, process=proc, _stdout_q=q, _reader_thread=t)

    def send(self, command: str) -> None:
        if self.process.stdin is None or self.process.poll() is not None:
            raise UCIError("engine stdin unavailable (process exited?)")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def read_lines(
        self,
        predicate: Callable[[str], bool],
        timeout: float,
    ) -> list[str]:
        """Read until predicate(line) is True or timeout elapses.

        Returns every line collected (including the terminator).
        """
        deadline = time.monotonic() + timeout
        collected: list[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise UCIError(
                    f"timed out after {timeout:.1f}s waiting for engine; "
                    f"last {len(collected)} line(s) received"
                )
            try:
                line = self._stdout_q.get(timeout=remaining)
            except queue.Empty:
                continue
            if line == "" and self.process.poll() is not None:
                raise UCIError("engine exited before expected terminator")
            collected.append(line)
            if predicate(line):
                return collected

    def handshake(self, timeout: float = 5.0) -> None:
        self.send("uci")
        self.read_lines(lambda ln: ln.strip() == "uciok", timeout=timeout)
        self.send("isready")
        self.read_lines(lambda ln: ln.strip() == "readyok", timeout=timeout)

    def set_position(self, fen: str, moves: Iterable[str] = ()) -> None:
        move_suffix = ""
        move_list = list(moves)
        if move_list:
            move_suffix = " moves " + " ".join(move_list)
        if fen.strip().lower() == "startpos":
            self.send(f"position startpos{move_suffix}")
        else:
            self.send(f"position fen {fen}{move_suffix}")

    def quit(self, grace: float = 1.0) -> None:
        try:
            if self.process.poll() is None:
                self.send("quit")
                try:
                    self.process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=grace)
        except Exception:
            if self.process.poll() is None:
                self.process.kill()

    def __enter__(self) -> "UCIEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.quit()
