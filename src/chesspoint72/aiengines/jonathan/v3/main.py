"""Calix engine entrypoint and UCI controller.

Run as a UCI engine:
    python -m chesspoint72.aiengines.jonathan.v3.main --agent-mode aware

CLI flags:
    --agent-mode {blind|aware|autonomous}   Selects the AgentContext preset.
    --depth N                                Default search depth.
    --time SECONDS                           Default search time per move.

The Calix UCI controller wraps the upstream ``UciController`` so the agent
re-runs at every ``go`` command in non-blind modes; this lets the
configuration adapt to position and clock as the game progresses.
"""
from __future__ import annotations

import sys
import time
from typing import Iterable, TextIO

import chess

from chesspoint72.aiengines.jonathan.v3.agent import (
    AgentContext,
    EngineConfig,
    build_context,
    select_modules,
    with_runtime_position,
)
from chesspoint72.aiengines.jonathan.v3.policies import (
    CapturesFirstOrderingPolicy,
    StubMoveOrderingPolicy,
    StubPruningPolicy,
)
from chesspoint72.engine.boards import PyChessBoard
from chesspoint72.engine.core.evaluator import Evaluator
from chesspoint72.engine.core.policies import MoveOrderingPolicy, PruningPolicy
from chesspoint72.engine.core.transposition import TranspositionTable
from chesspoint72.engine.factory import build_evaluator
from chesspoint72.engine.pruning import ForwardPruningPolicy
from chesspoint72.engine.pruning.config import PruningConfig
from chesspoint72.engine.search.negamax import NegamaxSearch
from chesspoint72.engine.uci.controller import UciController


# --------------------------------------------------------------------------- #
# Config -> live components
# --------------------------------------------------------------------------- #


def _make_evaluator(cfg: EngineConfig) -> Evaluator:
    """Resolve the named evaluator. Falls back to the upstream factory."""
    return build_evaluator(cfg.evaluator_name, cfg.hce_modules)


def _make_move_ordering(cfg: EngineConfig) -> MoveOrderingPolicy:
    if cfg.move_ordering == "captures_first":
        return CapturesFirstOrderingPolicy()
    return StubMoveOrderingPolicy()


def _make_pruning_policy(cfg: EngineConfig) -> PruningPolicy:
    """Build a pruning policy from the resolved config.

    When ``pruning_enabled`` is False we hand back a no-op policy so the
    upstream NegamaxSearch never tries to call into the forward-pruning
    algorithms — this is what guarantees minimal mode plays raw α-β.
    """
    if not cfg.pruning_enabled:
        return StubPruningPolicy()
    pruning_config = PruningConfig(
        nmp_enabled=cfg.nmp_enabled,
        futility_enabled=cfg.futility_enabled,
        razoring_enabled=cfg.razoring_enabled,
        lmr_enabled=cfg.lmr_enabled,
        nmp_r_shallow=2,
        nmp_r_deep=3,
        futility_margin=300,
        razoring_margins=cfg.razoring_margins,
        lmr_min_depth=3,
        lmr_min_move_index=3,
    )
    return ForwardPruningPolicy(pruning_config)


def build_search(cfg: EngineConfig) -> NegamaxSearch:
    """Wire an EngineConfig into a fully-constructed NegamaxSearch instance."""
    evaluator = _make_evaluator(cfg)
    move_ordering = _make_move_ordering(cfg)
    pruning_policy = _make_pruning_policy(cfg)
    pruning_config = (
        pruning_policy.config
        if isinstance(pruning_policy, ForwardPruningPolicy)
        else None
    )
    return NegamaxSearch(
        evaluator=evaluator,
        transposition_table=TranspositionTable(),
        move_ordering_policy=move_ordering,
        pruning_policy=pruning_policy,
        pruning_config=pruning_config,
    )


def log_activated(cfg: EngineConfig, stream: TextIO) -> None:
    """Print one ``info string`` line per activated module to *stream*.

    Goes to stderr by default (see ``main``) so the activation audit log
    never collides with the UCI protocol on stdout.
    """
    for name, reason in cfg.activated_modules:
        print(f"info string calix activated {name}: {reason}", file=stream, flush=True)


# --------------------------------------------------------------------------- #
# UCI controller
# --------------------------------------------------------------------------- #


class CalixController(UciController):
    """UCI controller that re-resolves the agent at every ``go`` command.

    For aware/autonomous modes the agent re-inspects the board and clock so
    its decisions track the live game; for blind mode it never sees either.
    """

    engine_name = "Calix v3 (Local rule-table selector)"
    engine_author = "Jonathan / Calix Project"

    def __init__(
        self,
        base_context: AgentContext,
        log_stream: TextIO,
        input_stream: Iterable[str] | None = None,
        output_stream: TextIO | None = None,
        default_depth: int = 4,
        default_time: float = 5.0,
    ) -> None:
        self._base_context = base_context
        self._log_stream = log_stream
        self._board = PyChessBoard()
        # The search is rebuilt at each go, so a placeholder satisfies the
        # base class contract until the first go command arrives.
        placeholder_search = build_search(
            select_modules(_blind_context_with_registry(base_context))
        )
        super().__init__(self._board, placeholder_search, input_stream, output_stream)
        self._default_depth = default_depth
        self._default_time = default_time

    # --- UCI handlers ------------------------------------------------------

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
        legal = self._board.generate_legal_moves()
        if not legal:
            self._writeln("bestmove 0000")
            return

        max_depth, allotted = self._parse_go(input_string)

        # Re-run the agent with the current position + clock visible.
        ctx = with_runtime_position(
            self._base_context,
            position_fen=self._board.get_current_fen(),
            time_remaining_ms=int(allotted * 1000),
        )
        cfg = select_modules(ctx)
        cfg.default_depth = max_depth
        cfg.default_time = allotted
        log_activated(cfg, self._log_stream)
        self.search_engine_reference = build_search(cfg)

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

    # --- helpers -----------------------------------------------------------

    def _parse_go(self, input_string: str) -> tuple[int, float]:
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
            our_inc = winc if self._board.side_to_move.value == 0 else binc
            allotted = max(our_time / 30.0 + our_inc, 0.05)
        return max_depth, allotted


def _blind_context_with_registry(ctx: AgentContext) -> AgentContext:
    """Helper used to build the placeholder search at construction time.

    Returns a no-position context that always resolves to a minimal config,
    regardless of what mode *ctx* declares — only used before the first
    ``go`` command, so we don't waste effort building a richer config that
    will be replaced moments later.
    """
    return AgentContext(
        mode="minimal",
        position_fen=None,
        time_remaining_ms=None,
        can_add_modules=False,
        available_modules=ctx.available_modules,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_cli(argv: list[str]) -> tuple[str, int, float]:
    agent_mode = "aware"
    default_depth = 4
    default_time = 5.0
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--agent-mode" and i + 1 < len(argv):
            agent_mode = argv[i + 1]; i += 2
        elif a.startswith("--agent-mode="):
            agent_mode = a.split("=", 1)[1]; i += 1
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
    return agent_mode, default_depth, default_time


def build_controller(
    agent_mode: str = "aware",
    *,
    input_stream: Iterable[str] | None = None,
    output_stream: TextIO | None = None,
    log_stream: TextIO | None = None,
    default_depth: int = 4,
    default_time: float = 5.0,
) -> CalixController:
    """Public factory used by tests and ``main``."""
    base_ctx = build_context(agent_mode)
    return CalixController(
        base_context=base_ctx,
        log_stream=log_stream if log_stream is not None else sys.stderr,
        input_stream=input_stream,
        output_stream=output_stream,
        default_depth=default_depth,
        default_time=default_time,
    )


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    agent_mode, default_depth, default_time = _parse_cli(argv)
    try:
        controller = build_controller(
            agent_mode=agent_mode,
            default_depth=default_depth,
            default_time=default_time,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        controller.start_listening_loop()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
