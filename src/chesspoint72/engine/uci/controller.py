# @capability: uci
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, Mapping, TextIO

if TYPE_CHECKING:
    from chesspoint72.engine.core.board import Board
    from chesspoint72.engine.core.search import Search


class UciController(ABC):
    """Translates UCI protocol traffic into engine actions.

    ``start_listening_loop`` provides a default dispatch over standard UCI
    commands; subclasses must implement the position/go handlers that bind
    those commands to a concrete Board and Search.
    """

    engine_name: str = "Chesspoint72 Engine"
    engine_author: str = "Chesspoint72"

    search_engine_reference: Search
    current_board_reference: Board
    is_engine_running: bool

    def __init__(
        self,
        board: Board,
        search: Search,
        input_stream: Iterable[str] | None = None,
        output_stream: TextIO | None = None,
    ) -> None:
        self.current_board_reference = board
        self.search_engine_reference = search
        self.is_engine_running = False
        self._input: Iterable[str] = input_stream if input_stream is not None else sys.stdin
        self._output: TextIO = output_stream if output_stream is not None else sys.stdout

    def start_listening_loop(self) -> None:
        self.is_engine_running = True
        for raw_line in self._input:
            if not self.is_engine_running:
                break
            line = raw_line.strip()
            if not line:
                continue
            self._dispatch(line)

    def _dispatch(self, line: str) -> None:
        command, _, rest = line.partition(" ")
        if command == "uci":
            self._send_handshake()
        elif command == "isready":
            self._writeln("readyok")
        elif command == "ucinewgame":
            self.handle_new_game()
        elif command == "position":
            self.handle_position_command(rest)
        elif command == "go":
            self.handle_go_command(rest)
        elif command == "stop":
            self.handle_stop_command()
        elif command == "quit":
            self.is_engine_running = False

    def handle_new_game(self) -> None:
        return None

    def handle_stop_command(self) -> None:
        return None

    @abstractmethod
    def handle_position_command(self, input_string: str) -> None: ...

    @abstractmethod
    def handle_go_command(self, input_string: str) -> None: ...

    def send_info_string(self, info_data: Mapping[str, object]) -> None:
        parts = ["info"]
        for key, value in info_data.items():
            parts.append(f"{key} {value}")
        self._writeln(" ".join(parts))

    def _send_handshake(self) -> None:
        self._writeln(f"id name {self.engine_name}")
        self._writeln(f"id author {self.engine_author}")
        self._writeln("uciok")

    def _writeln(self, text: str) -> None:
        print(text, file=self._output, flush=True)
