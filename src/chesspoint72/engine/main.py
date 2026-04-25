"""Chesspoint72 engine executable entrypoint.

Speaks UCI over stdio so the orchestrator (and any UCI-capable GUI or
tournament runner) can drive it. Today this wires the team's
`UciController` base class to a minimal shim that uses python-chess for
position parsing and legal-move generation and picks a random legal move.

That gives us three things:

1. `uci` → `uciok` handshake works, so the MCP orchestrator can reach us.
2. The engine plays legal chess end-to-end, so SPRT runs are unblocked
   (against Stockfish Level 0 or another baseline) starting right now.
3. A drop-in seam for the real `Board` + `Search`: once the team has a
   concrete alpha-beta search and bitboard move-gen, `ShimUciController`
   is the only class that needs to change.

Run:
    python -m chesspoint72.engine
"""
from __future__ import annotations

import os
import random
import sys
from typing import Iterable, TextIO

import chess

from chesspoint72.engine.board import Board
from chesspoint72.engine.evaluator import Evaluator
from chesspoint72.engine.search import Search
from chesspoint72.engine.transposition import TranspositionTable
from chesspoint72.engine.types import Move
from chesspoint72.engine.uci_controller import UciController


# --------------------------------------------------------------------------- #
# Stub concretions to satisfy the abstract interfaces. They are never invoked
# by ShimUciController — the shim owns its own python-chess board and random
# move policy. Replace these with real implementations as they land.
# --------------------------------------------------------------------------- #


class _StubBoard(Board):
    def set_position_from_fen(self, fen_string: str) -> None: ...
    def get_current_fen(self) -> str:
        return chess.STARTING_FEN
    def generate_legal_moves(self) -> list[Move]:
        return []
    def make_move(self, move: Move) -> None: ...
    def unmake_move(self) -> None: ...
    def is_king_in_check(self) -> bool:
        return False
    def calculate_zobrist_hash(self) -> int:
        return 0


class _PyChessBoardAdapter(Board):
    """Adapter exposing a python-chess board via the engine's Board interface."""

    def __init__(self, py_board: chess.Board) -> None:
        super().__init__()
        self._py_board = py_board

    def set_position_from_fen(self, fen_string: str) -> None:
        self._py_board.set_fen(fen_string)

    def get_current_fen(self) -> str:
        return self._py_board.fen()

    def generate_legal_moves(self) -> list[Move]:
        return []
    def make_move(self, move: Move) -> None: ...
    def unmake_move(self) -> None: ...
    def is_king_in_check(self) -> bool:
        return self._py_board.is_check()
    def calculate_zobrist_hash(self) -> int:
        return 0


class _StubEvaluator(Evaluator):
    def evaluate_position(self, board: Board) -> int:
        return 0


def build_evaluator(name: str | None = None) -> Evaluator:
    """Factory: select an Evaluator implementation by name.

    Recognised names: "stub" (default), "nnue". The selected name is also
    read from the CHESSPOINT72_EVALUATOR env var when `name` is None, so the
    Battle Royale (SPRT) runner can flip evaluators without rebuilding.
    """
    chosen = (name or os.environ.get("CHESSPOINT72_EVALUATOR", "stub")).strip().lower()
    if chosen == "nnue":
        from chesspoint72.engine.nnue_evaluator import NnueEvaluator
        weights = os.environ.get("CHESSPOINT72_NNUE_WEIGHTS")
        return NnueEvaluator(weights) if weights else NnueEvaluator()
    if chosen in ("", "stub", "hce"):
        return _StubEvaluator()
    raise ValueError(f"unknown evaluator: {chosen!r}")


class _StubSearch(Search):
    def find_best_move(self, board: Board, max_depth: int, allotted_time: float) -> Move:
        raise NotImplementedError
    def search_node(self, alpha: int, beta: int, depth: int) -> int:
        return 0
    def quiescence_search(self, alpha: int, beta: int) -> int:
        return 0
    def order_moves(self, moves: list[Move]) -> list[Move]:
        return moves


# --------------------------------------------------------------------------- #
# Concrete controller: parses UCI position/go against a python-chess board
# and replies with a random legal move. This is intentionally the weakest
# possible policy — its job is to be a protocol-correct, fully-playable
# baseline while the real search is being written.
# --------------------------------------------------------------------------- #


class ShimUciController(UciController):
    engine_name = "Chesspoint72 Shim"
    engine_author = "Chesspoint72"

    def __init__(
        self,
        board: Board,
        search: Search,
        input_stream: Iterable[str] | None = None,
        output_stream: TextIO | None = None,
        rng: random.Random | None = None,
        evaluator: Evaluator | None = None,
    ) -> None:
        super().__init__(board, search, input_stream, output_stream)
        self._board = chess.Board()
        self._rng = rng or random.Random()
        self._evaluator = evaluator

    def handle_new_game(self) -> None:
        self._board = chess.Board()

    def handle_position_command(self, input_string: str) -> None:
        tokens = input_string.split()
        if not tokens:
            return
        moves: list[str] = []
        if tokens[0] == "startpos":
            self._board = chess.Board()
            if "moves" in tokens:
                moves = tokens[tokens.index("moves") + 1:]
        elif tokens[0] == "fen":
            # FEN is exactly 6 whitespace-separated fields.
            if len(tokens) < 7:
                return
            fen = " ".join(tokens[1:7])
            try:
                self._board = chess.Board(fen)
            except ValueError:
                return
            if "moves" in tokens:
                moves = tokens[tokens.index("moves") + 1:]
        else:
            return
        for uci in moves:
            try:
                self._board.push_uci(uci)
            except (ValueError, AssertionError):
                break

    def handle_go_command(self, input_string: str) -> None:
        if self._board.is_game_over(claim_draw=True):
            self._writeln("bestmove 0000")
            return
        legal = list(self._board.legal_moves)
        if not legal:
            self._writeln("bestmove 0000")
            return

        if self._evaluator is not None and not isinstance(self._evaluator, _StubEvaluator):
            white_to_move = self._board.turn == chess.WHITE
            best_move = legal[0]
            best_score = -10**9
            for mv in legal:
                self._board.push(mv)
                cp = self._evaluator.evaluate_position(_PyChessBoardAdapter(self._board))
                self._board.pop()
                signed = cp if white_to_move else -cp
                if signed > best_score:
                    best_score = signed
                    best_move = mv
            move = best_move
            score_cp = best_score if white_to_move else -best_score
            self.send_info_string({"depth": 1, "score cp": int(score_cp), "nodes": len(legal), "pv": move.uci()})
        else:
            move = self._rng.choice(legal)
            self.send_info_string({"depth": 1, "score cp": 0, "nodes": len(legal), "pv": move.uci()})
        self._writeln(f"bestmove {move.uci()}")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    evaluator_name: str | None = None,
) -> ShimUciController:
    evaluator = build_evaluator(evaluator_name)
    return ShimUciController(
        board=_StubBoard(),
        search=_StubSearch(evaluator, TranspositionTable()),
        input_stream=input_stream,
        output_stream=output_stream,
        evaluator=evaluator,
    )


def main() -> int:
    # CLI form: `python -m chesspoint72.engine --evaluator nnue`
    evaluator_name: str | None = None
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--evaluator" and i + 1 < len(argv):
            evaluator_name = argv[i + 1]
        elif arg.startswith("--evaluator="):
            evaluator_name = arg.split("=", 1)[1]
    controller = build_controller(evaluator_name=evaluator_name)
    try:
        controller.start_listening_loop()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
