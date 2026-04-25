from __future__ import annotations

from dataclasses import dataclass, field

import chess

from chesspoint72.engine.factory import build_controller


@dataclass
class MoveInfo:
    """Rich result from the built-in engine: move + search metadata."""
    move: chess.Move
    score_cp: int           # centipawns, side-to-move perspective
    depth: int
    nodes: int
    pv_uci: list[str] = field(default_factory=list)


class BuiltinEngineClient:
    """In-process engine client — same duck-type interface as UciEngineClient."""

    def __init__(
        self,
        evaluator: str | None = None,
        hce_modules: str | None = None,
        depth: int = 4,
        think_time: float = 0.2,
    ) -> None:
        self._evaluator = evaluator
        self._hce_modules = hce_modules
        self._depth = depth
        self._think_time = think_time
        self._controller = None

    def start(self) -> None:
        self._controller = build_controller(
            evaluator_name=self._evaluator,
            hce_modules=self._hce_modules,
            default_depth=self._depth,
            default_time=self._think_time,
        )

    def stop(self) -> None:
        self._controller = None

    def request_best_move(self, board: chess.Board) -> chess.Move:
        return self.request_move_info(board).move

    def request_move_info(self, board: chess.Board) -> MoveInfo:
        """Return the best move plus search metadata (score, depth, PV)."""
        assert self._controller is not None, "call start() first"
        self._controller.handle_position_command(f"fen {board.fen()}")
        legal = self._controller.current_board_reference.generate_legal_moves()
        if not legal:
            raise RuntimeError("No legal moves available")
        search = self._controller.search_engine_reference
        move = search.find_best_move(
            self._controller.current_board_reference,
            self._depth,
            self._think_time,
        )
        if move is None:
            move = legal[0]
        chess_move = chess.Move.from_uci(move.to_uci_string())
        score = getattr(search, "_last_root_score", 0)
        depth = getattr(search, "depth_reached", 0)
        nodes = getattr(search, "nodes_evaluated", 0)
        pv_uci = search.extract_pv_uci() if hasattr(search, "extract_pv_uci") else []
        return MoveInfo(move=chess_move, score_cp=score, depth=depth, nodes=nodes, pv_uci=pv_uci)
