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
from pathlib import Path
from typing import Callable, Iterable, TextIO

import chess

from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.board import Board
from chesspoint72.engine.core.evaluator import Evaluator
from chesspoint72.engine.core.policies import MoveOrderingPolicy, PruningPolicy
from chesspoint72.engine.core.search import Search
from chesspoint72.engine.core.transposition import TranspositionTable
from chesspoint72.engine.core.types import Move
from chesspoint72.engine.ordering import HistoryTable, KillerMoveTable, MoveSorter
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


class MoveSorterPolicy(MoveOrderingPolicy):
    """Wraps MoveSorter (TT + MVV-LVA + killers + history) as a MoveOrderingPolicy.

    Exposes ``set_depth(depth)`` so NegamaxSearch can pass the current search
    depth before each ``order_moves`` call; killer lookups then use the correct
    depth slot without changing the ABC signature.

    Tables are injected at construction time so that ``build_controller`` can
    share the exact same ``KillerMoveTable`` and ``HistoryTable`` instances
    between this policy and ``NegamaxSearch``.  The search updates the tables
    on beta-cutoffs; this policy reads them for ordering.
    """

    def __init__(
        self,
        killer_table: KillerMoveTable,
        history_table: HistoryTable,
    ) -> None:
        self._sorter = MoveSorter(
            killer_table=killer_table,
            history_table=history_table,
        )
        self._current_depth: int = 0

    def set_depth(self, depth: int) -> None:
        self._current_depth = depth

    def order_moves(
        self,
        moves: list[Move],
        board: Board,
        tt_best_move: Move | None = None,
    ) -> list[Move]:
        return list(self._sorter.iter_moves(board, moves, tt_best_move, self._current_depth))


# --------------------------------------------------------------------------- #
# Evaluator registry — one row per implementation.
# --------------------------------------------------------------------------- #


_NNUE_WEIGHTS_DIR = Path(__file__).resolve().parent / "evaluators" / "nnue" / "weights"


def _build_nnue() -> Evaluator:
    from chesspoint72.engine.evaluators.nnue import NnueEvaluator
    weights = os.environ.get("CHESSPOINT72_NNUE_WEIGHTS")
    return NnueEvaluator(weights) if weights else NnueEvaluator()


def _build_named_nnue(filename: str) -> Callable[[], Evaluator]:
    def _builder() -> Evaluator:
        from chesspoint72.engine.evaluators.nnue import NnueEvaluator
        return NnueEvaluator(_NNUE_WEIGHTS_DIR / filename)
    return _builder


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


def _as_chess_board(board: Board | chess.Board) -> chess.Board:
    if isinstance(board, chess.Board):
        return board
    py_board = getattr(board, "py_board", None)
    if isinstance(py_board, chess.Board):
        return py_board
    get_fen = getattr(board, "get_current_fen", None)
    fen = get_fen() if callable(get_fen) else board.fen()
    return chess.Board(fen)


HCE_MODULE_GROUPS: dict[str, tuple[str, ...]] = {
    "classic": (
        "material", "pst", "pawns", "king_safety", "mobility", "rooks", "bishops",
    ),
    "advanced": ("ewpm", "srcm", "idam", "otvm", "lmdm", "lscm", "clcm", "desm"),
}
HCE_MODULE_GROUPS["all"] = HCE_MODULE_GROUPS["classic"] + HCE_MODULE_GROUPS["advanced"]


def _normalize_hce_modules(raw: str | None, available: set[str]) -> list[str]:
    if raw is None or not raw.strip():
        return list(HCE_MODULE_GROUPS["all"])

    selected: list[str] = []
    seen: set[str] = set()
    for token in (t.strip().lower() for t in raw.split(",")):
        if not token:
            continue
        expanded = HCE_MODULE_GROUPS.get(token, (token,))
        for name in expanded:
            if name not in available:
                valid = ", ".join(sorted(available | set(HCE_MODULE_GROUPS)))
                raise ValueError(f"unknown hce module: {name!r}; valid values: {valid}")
            if name in seen:
                continue
            seen.add(name)
            selected.append(name)

    if not selected:
        raise ValueError("no hce modules were selected")
    return selected


class _HceEvaluator(Evaluator):
    """Configurable adapter exposing chesspoint72.hce modules as an Evaluator.

    Returns centipawns from the side-to-move perspective.
    """

    def __init__(self, modules: str | None = None) -> None:
        from chesspoint72.hce import hce as hce_impl

        self._feature_fns: dict[str, Callable[[chess.Board], tuple[int, int]]] = {
            "material": hce_impl.material_balance,
            "pst": hce_impl.pst_score,
            "pawns": hce_impl.pawn_structure,
            "king_safety": hce_impl.king_safety,
            "mobility": hce_impl.mobility_score,
            "rooks": hce_impl.rook_bonuses,
            "bishops": hce_impl.bishop_pair,
            "ewpm": hce_impl.ewpm,
            "srcm": hce_impl.srcm,
            "idam": hce_impl.idam,
            "otvm": hce_impl.otvm,
            "lmdm": hce_impl.lmdm,
            "lscm": hce_impl.lscm,
            "clcm": hce_impl.clcm,
            "desm": hce_impl.desm,
        }
        self._selected_modules = _normalize_hce_modules(modules, set(self._feature_fns))
        self._selected_fns = [self._feature_fns[name] for name in self._selected_modules]
        self._get_game_phase = hce_impl.get_game_phase
        self._taper = hce_impl.taper
        self._idam = hce_impl.idam
        self._mate_limit = 31_999

    def evaluate_position(self, board: Board) -> int:
        py_board = _as_chess_board(board)
        phase = self._get_game_phase(py_board)
        total_mg = total_eg = 0
        for fn in self._selected_fns:
            mg, eg = fn(py_board)
            total_mg += mg
            total_eg += eg

        score = self._taper(total_mg, total_eg, phase)
        score = max(-self._mate_limit, min(self._mate_limit, score))

        # IDAM tracks a short score-history trajectory for future calls.
        if "idam" in self._selected_modules:
            self._idam.record(float(score))

        if py_board.turn == chess.BLACK:
            score = -score
        return int(score)


def _build_hce(modules: str | None) -> Evaluator:
    chosen_modules = modules if modules is not None else os.environ.get("CHESSPOINT72_HCE_MODULES")
    return _HceEvaluator(chosen_modules)


_EVALUATOR_REGISTRY: dict[str, Callable[[], Evaluator]] = {
    "stub":           lambda: _StubEvaluator(),
    "material":       lambda: _MaterialEvaluator(),
    "nnue":           _build_nnue,
    "nnue_baseline":  _build_named_nnue("nnue_weights.pt"),
    "nnue_tank":      _build_named_nnue("nnue_tank_final.pt"),
    "nnue_tactician": _build_named_nnue("nnue_tactician_final.pt"),
    "nnue_speedster": _build_named_nnue("nnue_speedster_final.pt"),
    "nnue_finisher":  _build_named_nnue("nnue_finisher_final.pt"),
}


def build_evaluator(name: str | None = None, hce_modules: str | None = None) -> Evaluator:
    """Select an Evaluator implementation by name.

    Falls back to ``CHESSPOINT72_EVALUATOR`` when *name* is None, then to
    ``"stub"``. The Battle Royale runner uses the env var to flip evaluators
    without rebuilding.
    """
    chosen = (name or os.environ.get("CHESSPOINT72_EVALUATOR", "stub")).strip().lower() or "stub"
    if chosen == "hce":
        return _build_hce(hce_modules)
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
    hce_modules: str | None = None,
    default_depth: int = 4,
    default_time: float = 5.0,
) -> StandardUciController:
    evaluator = build_evaluator(evaluator_name, hce_modules)
    board = PyChessBoard()
    pruning_config = default_pruning_config()
    pruning_policy = ForwardPruningPolicy(pruning_config)

    # Shared tables: MoveSorterPolicy reads them for ordering; NegamaxSearch
    # writes them on beta-cutoffs.  Both must point at the same objects.
    killer_table = KillerMoveTable()
    history_table = HistoryTable()
    ordering_policy = MoveSorterPolicy(killer_table, history_table)

    search = NegamaxSearch(
        evaluator,
        TranspositionTable(),
        ordering_policy,
        pruning_policy,
        pruning_config,
    )
    # Replace the tables NegamaxSearch created internally with the shared ones.
    search.killer_table = killer_table
    search.history_table = history_table

    return StandardUciController(
        board=board,
        search=search,
        input_stream=input_stream,
        output_stream=output_stream,
        evaluator=evaluator,
        default_depth=default_depth,
        default_time=default_time,
    )


def _parse_cli(argv: list[str]) -> tuple[str | None, str | None, int, float]:
    """Parse engine CLI flags.

    Returns (evaluator_name, hce_modules, default_depth, default_time).
    """
    evaluator_name: str | None = None
    hce_modules: str | None = None
    default_depth = 4
    default_time = 5.0
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--evaluator" and i + 1 < len(argv):
            evaluator_name = argv[i + 1]; i += 2
        elif a.startswith("--evaluator="):
            evaluator_name = a.split("=", 1)[1]; i += 1
        elif a == "--hce-modules" and i + 1 < len(argv):
            hce_modules = argv[i + 1]; i += 2
        elif a.startswith("--hce-modules="):
            hce_modules = a.split("=", 1)[1]; i += 1
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
    return evaluator_name, hce_modules, default_depth, default_time


def main() -> int:
    evaluator_name, hce_modules, default_depth, default_time = _parse_cli(sys.argv[1:])
    controller = build_controller(
        evaluator_name=evaluator_name,
        hce_modules=hce_modules,
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
