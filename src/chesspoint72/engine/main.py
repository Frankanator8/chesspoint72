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

import random
import sys
from typing import Iterable, TextIO

import chess

from chesspoint72.engine.board import Board
from chesspoint72.engine.evaluator import Evaluator
from chesspoint72.engine.policies import MoveOrderingPolicy, PruningPolicy
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
    ) -> None:
        super().__init__(board, search, input_stream, output_stream)
        self._board = chess.Board()
        self._rng = rng or random.Random()

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
        move = self._rng.choice(legal)
        self.send_info_string({"depth": 1, "score cp": 0, "nodes": len(legal), "pv": move.uci()})
        self._writeln(f"bestmove {move.uci()}")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
) -> ShimUciController:
    return ShimUciController(
        board=_StubBoard(),
        search=_StubSearch(
            _StubEvaluator(),
            TranspositionTable(),
            _StubMoveOrderingPolicy(),
            _StubPruningPolicy(),
        ),
        input_stream=input_stream,
        output_stream=output_stream,
    )


def main() -> int:
    controller = build_controller()
    try:
        controller.start_listening_loop()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
