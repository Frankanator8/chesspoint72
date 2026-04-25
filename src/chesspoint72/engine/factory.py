"""Engine factory + UCI entrypoint.

Provides a single dispatch table per concern (Evaluator, Search, …) and a
``build_controller`` that wires everything together. Adding a new
implementation = registering it here; no file owned by another teammate is
touched.

Run:
    python -m chesspoint72.engine                       # default stub engine
    python -m chesspoint72.engine --evaluator nnue       # NNUE evaluator
"""
from __future__ import annotations

import os
import random
import sys
from typing import Callable, Iterable, TextIO

import chess

from chesspoint72.engine.core.board import Board
from chesspoint72.engine.core.evaluator import Evaluator
from chesspoint72.engine.core.policies import MoveOrderingPolicy, PruningPolicy
from chesspoint72.engine.core.search import Search
from chesspoint72.engine.core.transposition import TranspositionTable
from chesspoint72.engine.core.types import Move
from chesspoint72.engine.uci.controller import UciController


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


class _StubMoveOrderingPolicy(MoveOrderingPolicy):
    def order_moves(
        self,
        moves: list[Move],
        board: Board,
        tt_best_move: Move | None = None,
    ) -> list[Move]:
        return moves


class _StubPruningPolicy(PruningPolicy):
    def try_prune(
        self,
        board: Board,
        depth: int,
        alpha: int,
        beta: int,
        static_eval: int,
    ) -> int | None:
        return None


class _StubSearch(Search):
    def find_best_move(self, board: Board, max_depth: int, allotted_time: float) -> Move:
        raise NotImplementedError
    def search_node(self, alpha: int, beta: int, depth: int) -> int:
        return 0
    def quiescence_search(self, alpha: int, beta: int) -> int:
        return 0


# --------------------------------------------------------------------------- #
# Factories — one dispatch table per concern. Add a new row to register an
# implementation; do not modify other rows.
# --------------------------------------------------------------------------- #


def _build_nnue() -> Evaluator:
    from chesspoint72.engine.evaluators.nnue import NnueEvaluator
    weights = os.environ.get("CHESSPOINT72_NNUE_WEIGHTS")
    return NnueEvaluator(weights) if weights else NnueEvaluator()


_EVALUATOR_REGISTRY: dict[str, Callable[[], Evaluator]] = {
    "stub": lambda: _StubEvaluator(),
    "hce":  lambda: _StubEvaluator(),  # placeholder until HCE lands
    "nnue": _build_nnue,
}


def build_evaluator(name: str | None = None) -> Evaluator:
    """Select an Evaluator implementation by name.

    Falls back to ``CHESSPOINT72_EVALUATOR`` when *name* is None, then to
    ``"stub"``. The Battle Royale (SPRT) runner uses the env var to flip
    evaluators without rebuilding.
    """
    chosen = (name or os.environ.get("CHESSPOINT72_EVALUATOR", "stub")).strip().lower() or "stub"
    if chosen not in _EVALUATOR_REGISTRY:
        raise ValueError(f"unknown evaluator: {chosen!r}")
    return _EVALUATOR_REGISTRY[chosen]()


# --------------------------------------------------------------------------- #
# Concrete controller: parses UCI position/go against a python-chess board,
# and (when a real evaluator is wired) replies with a 1-ply greedy best move.
# Otherwise picks a random legal move — sufficient for SPRT plumbing while
# the real search lands.
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
        search=_StubSearch(
            evaluator,
            TranspositionTable(),
            _StubMoveOrderingPolicy(),
            _StubPruningPolicy(),
        ),
        input_stream=input_stream,
        output_stream=output_stream,
        evaluator=evaluator,
    )


def main() -> int:
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
