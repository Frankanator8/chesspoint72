"""Engine factory + UCI entrypoint.

Provides a single dispatch table per concern (Evaluator, Search, …) and a
``build_controller`` that wires everything together. Adding a new
implementation = registering it here; no file owned by another teammate is
touched.

Run:
    python -m chesspoint72.engine                                # stub eval
    python -m chesspoint72.engine --evaluator nnue                # NNUE
    python -m chesspoint72.engine --evaluator material --depth 5  # ID to depth 5
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, Iterable, TextIO

import chess

from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.board import Board
from chesspoint72.engine.core.evaluator import Evaluator
from chesspoint72.engine.core.policies import MoveOrderingPolicy, PruningPolicy
from chesspoint72.engine.core.search import Search
from chesspoint72.engine.core.transposition import TranspositionTable
from chesspoint72.engine.core.types import Move
from chesspoint72.engine.pruning import ForwardPruningPolicy, default_pruning_config
from chesspoint72.engine.search.negamax import NegamaxSearch
from chesspoint72.engine.uci.controller import UciController


# --------------------------------------------------------------------------- #
# Minimal stubs — only those still wired into a live code path.
#
# ``_StubEvaluator`` serves as the fallback in the evaluator registry for the
# "stub" / "hce" keys. The two stub policies are passed to NegamaxSearch until
# the move-ordering and forward-pruning teammates ship real implementations.
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Evaluator registry — one row per implementation.
# --------------------------------------------------------------------------- #


def _build_nnue() -> Evaluator:
    from chesspoint72.engine.evaluators.nnue import NnueEvaluator
    weights = os.environ.get("CHESSPOINT72_NNUE_WEIGHTS")
    return NnueEvaluator(weights) if weights else NnueEvaluator()


class _MaterialEvaluator(Evaluator):
    """Adapter exposing chesspoint72.hce.material as an Evaluator.

    Returns centipawns from White's POV.
    """

    def evaluate_position(self, board: Board) -> int:
        from chesspoint72.hce.material import material_score
        if isinstance(board, chess.Board):
            return int(material_score(board))
        # PyChessBoard exposes the underlying chess.Board via .py_board for
        # zero-copy access; everything else falls through to FEN.
        py_board = getattr(board, "py_board", None)
        if isinstance(py_board, chess.Board):
            return int(material_score(py_board))
        get_fen = getattr(board, "get_current_fen", None)
        fen = get_fen() if callable(get_fen) else board.fen()
        return int(material_score(chess.Board(fen)))


_EVALUATOR_REGISTRY: dict[str, Callable[[], Evaluator]] = {
    "stub":     lambda: _StubEvaluator(),
    "hce":      lambda: _StubEvaluator(),  # placeholder until full HCE lands
    "material": lambda: _MaterialEvaluator(),
    "nnue":     _build_nnue,
}


def build_evaluator(name: str | None = None) -> Evaluator:
    """Select an Evaluator implementation by name.

    Falls back to ``CHESSPOINT72_EVALUATOR`` when *name* is None, then to
    ``"stub"``. The Battle Royale runner uses the env var to flip evaluators
    without rebuilding.
    """
    chosen = (name or os.environ.get("CHESSPOINT72_EVALUATOR", "stub")).strip().lower() or "stub"
    if chosen not in _EVALUATOR_REGISTRY:
        raise ValueError(f"unknown evaluator: {chosen!r}")
    return _EVALUATOR_REGISTRY[chosen]()


# --------------------------------------------------------------------------- #
# StandardUciController — drives a real Search over a real PyChessBoard.
#
# The per-depth ``info`` lines are produced by the *controller*, which calls
# ``search.find_best_move(board, depth=d, allotted_time=remaining)`` for d in
# 1..max_depth. NegamaxSearch's own internal IID still runs inside each call
# (that's how it computes a stable depth-d result), but the controller's outer
# loop is what surfaces per-depth output without needing to modify the Search
# implementation. The wasted work is bounded — sum_{d=1..N} d = N(N+1)/2 — and
# is acceptable for a tournament wrapper. A future Search can expose a hook to
# eliminate the duplication.
# --------------------------------------------------------------------------- #


class StandardUciController(UciController):
    """Real UCI controller backed by ``PyChessBoard`` + a ``Search`` impl."""

    engine_name = "Chesspoint72"
    engine_author = "Chesspoint72"

    def __init__(
        self,
        board: PyChessBoard,
        search: Search,
        input_stream: Iterable[str] | None = None,
        output_stream: TextIO | None = None,
        evaluator: Evaluator | None = None,
        default_depth: int = 4,
        default_time: float = 5.0,
    ) -> None:
        super().__init__(board, search, input_stream, output_stream)
        self._board: PyChessBoard = board
        self._evaluator = evaluator
        self._default_depth = default_depth
        self._default_time = default_time

    # ------------------------------------------------------------------ #
    # UCI command handlers
    # ------------------------------------------------------------------ #

    def handle_new_game(self) -> None:
        self._board.set_position_from_fen(chess.STARTING_FEN)

    def handle_position_command(self, input_string: str) -> None:
        tokens = input_string.split()
        if not tokens:
            return
        moves: list[str] = []
        if tokens[0] == "startpos":
            self._board.set_position_from_fen(chess.STARTING_FEN)
            if "moves" in tokens:
                moves = tokens[tokens.index("moves") + 1:]
        elif tokens[0] == "fen":
            if len(tokens) < 7:
                return
            fen = " ".join(tokens[1:7])
            try:
                self._board.set_position_from_fen(fen)
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
        if self._board.is_game_over():
            self._writeln("bestmove 0000")
            return

        max_depth, allotted = self._parse_go(input_string)
        legal = self._board.generate_legal_moves()
        if not legal:
            self._writeln("bestmove 0000")
            return

        # Per-depth iterative deepening driven from the controller, so each
        # completed depth produces its own UCI ``info`` line.
        deadline = time.monotonic() + allotted
        best_move = legal[0]
        for depth in range(1, max_depth + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            t0 = time.monotonic()
            try:
                move = self.search_engine_reference.find_best_move(
                    self._board, depth, remaining,
                )
            except Exception:
                break
            elapsed = time.monotonic() - t0
            nodes = getattr(self.search_engine_reference, "nodes_evaluated", 0)
            time_ms = max(int(elapsed * 1000), 1)
            nps = int(nodes / max(elapsed, 1e-6))
            if move is not None:
                best_move = move
            self.send_info_string({
                "depth": depth,
                "nodes": nodes,
                "nps": nps,
                "time": time_ms,
                "pv": best_move.to_uci_string(),
            })

        self._writeln(f"bestmove {best_move.to_uci_string()}")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _parse_go(self, input_string: str) -> tuple[int, float]:
        """Parse depth/movetime/clock fields from a UCI ``go`` line."""
        tokens = input_string.split()
        max_depth = self._default_depth
        allotted = self._default_time
        wtime = btime = winc = binc = movetime = -1.0
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == "depth" and i + 1 < len(tokens):
                try:
                    max_depth = max(int(tokens[i + 1]), 1)
                except ValueError:
                    pass
                i += 2
            elif t == "movetime" and i + 1 < len(tokens):
                try:
                    movetime = float(tokens[i + 1]) / 1000.0
                except ValueError:
                    pass
                i += 2
            elif t in ("wtime", "btime", "winc", "binc") and i + 1 < len(tokens):
                try:
                    val = float(tokens[i + 1]) / 1000.0
                except ValueError:
                    val = 0.0
                if   t == "wtime": wtime = val
                elif t == "btime": btime = val
                elif t == "winc":  winc = val
                else:              binc = val
                i += 2
            else:
                i += 1
        if movetime > 0:
            allotted = movetime
        elif wtime >= 0 and btime >= 0:
            our_time = wtime if self._board.side_to_move.value == 0 else btime
            our_inc  = winc  if self._board.side_to_move.value == 0 else binc
            # Conservative: spend ~1/30 of remaining clock plus the increment.
            allotted = max(our_time / 30.0 + our_inc, 0.05)
        return max_depth, allotted


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def build_controller(
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    evaluator_name: str | None = None,
    default_depth: int = 4,
    default_time: float = 5.0,
) -> StandardUciController:
    evaluator = build_evaluator(evaluator_name)
    board = PyChessBoard()
    pruning_config = default_pruning_config()
    pruning_policy = ForwardPruningPolicy(pruning_config)
    search = NegamaxSearch(
        evaluator,
        TranspositionTable(),
        _StubMoveOrderingPolicy(),
        pruning_policy,
        pruning_config,
    )
    return StandardUciController(
        board=board,
        search=search,
        input_stream=input_stream,
        output_stream=output_stream,
        evaluator=evaluator,
        default_depth=default_depth,
        default_time=default_time,
    )


def _parse_cli(argv: list[str]) -> tuple[str | None, int, float]:
    """Parse engine CLI flags. Returns (evaluator_name, default_depth, default_time)."""
    evaluator_name: str | None = None
    default_depth = 4
    default_time = 5.0
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--evaluator" and i + 1 < len(argv):
            evaluator_name = argv[i + 1]; i += 2
        elif a.startswith("--evaluator="):
            evaluator_name = a.split("=", 1)[1]; i += 1
        elif a == "--depth" and i + 1 < len(argv):
            default_depth = max(int(argv[i + 1]), 1); i += 2
        elif a.startswith("--depth="):
            default_depth = max(int(a.split("=", 1)[1]), 1); i += 1
        elif a == "--time" and i + 1 < len(argv):
            default_time = max(float(argv[i + 1]), 0.05); i += 2
        elif a.startswith("--time="):
            default_time = max(float(a.split("=", 1)[1]), 0.05); i += 1
        else:
            i += 1
    return evaluator_name, default_depth, default_time


def main() -> int:
    evaluator_name, default_depth, default_time = _parse_cli(sys.argv[1:])
    controller = build_controller(
        evaluator_name=evaluator_name,
        default_depth=default_depth,
        default_time=default_time,
    )
    try:
        controller.start_listening_loop()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
