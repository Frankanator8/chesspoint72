from __future__ import annotations

import chess

from chesspoint72.engine.factory import build_controller


class BuiltinEngineClient:
    """In-process engine client — same duck-type interface as UciEngineClient."""

    def __init__(
        self,
        evaluator: str | None = None,
        depth: int = 4,
        think_time: float = 0.2,
    ) -> None:
        self._evaluator = evaluator
        self._depth = depth
        self._think_time = think_time
        self._controller = None

    def start(self) -> None:
        self._controller = build_controller(
            evaluator_name=self._evaluator,
            default_depth=self._depth,
            default_time=self._think_time,
        )

    def stop(self) -> None:
        self._controller = None

    def request_best_move(self, board: chess.Board) -> chess.Move:
        assert self._controller is not None, "call start() first"
        self._controller.handle_position_command(f"fen {board.fen()}")
        legal = self._controller.current_board_reference.generate_legal_moves()
        if not legal:
            raise RuntimeError("No legal moves available")
        move = self._controller.search_engine_reference.find_best_move(
            self._controller.current_board_reference,
            self._depth,
            self._think_time,
        )
        if move is None:
            move = legal[0]
        return chess.Move.from_uci(move.to_uci_string())
