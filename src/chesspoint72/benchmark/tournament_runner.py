"""Round-robin tournament runner for all available chess engines.

Usage:
    python -m chesspoint72.benchmark.tournament_runner
    python -m chesspoint72.benchmark.tournament_runner --smoke          # quick (4 games/pairing)
    python -m chesspoint72.benchmark.tournament_runner --games 20       # full
    python -m chesspoint72.benchmark.tournament_runner --time 2.0       # 2s per move
    python -m chesspoint72.benchmark.tournament_runner --output out.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

import chess

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── Result types ────────────────────────────────────────────────────────────── #

@dataclass
class GameResult:
    white: str
    black: str
    score: float          # white's perspective: 1.0 / 0.5 / 0.0

@dataclass
class PairingResult:
    a: str
    b: str
    wins_a: int = 0
    draws:  int = 0
    wins_b: int = 0

    @property
    def total(self) -> int:
        return self.wins_a + self.draws + self.wins_b

    @property
    def score_a(self) -> float:
        return (self.wins_a + 0.5 * self.draws) / max(self.total, 1)

@dataclass
class TournamentResult:
    engines:   list[str]
    pairings:  list[PairingResult]
    games:     list[GameResult]
    elo:       dict[str, float]
    elapsed_s: float


# ── Engine adapters ─────────────────────────────────────────────────────────── #

class _Adapter:
    """Common interface: start / stop / get_best_move(fen, time_s) -> chess.Move."""
    name: str

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def get_best_move(self, fen: str, time_s: float) -> chess.Move | None:
        raise NotImplementedError


class ControllerAdapter(_Adapter):
    """Wraps any StandardUciController-based engine."""

    def __init__(self, name: str, build_fn: Callable) -> None:
        self.name = name
        self._build_fn = build_fn
        self._ctrl = None

    def start(self) -> None:
        self._ctrl = self._build_fn()

    def stop(self) -> None:
        self._ctrl = None

    def get_best_move(self, fen: str, time_s: float) -> chess.Move | None:
        ctrl = self._ctrl
        ctrl.handle_position_command(f"fen {fen}")
        board = ctrl.current_board_reference
        move = ctrl.search_engine_reference.find_best_move(board, 30, time_s)
        if move is None:
            return None
        return chess.Move.from_uci(move.to_uci_string())


class InstanceAdapter(_Adapter):
    """Wraps an EngineInstance from engine_config.py."""

    def __init__(self, name: str, build_fn: Callable) -> None:
        self.name = name
        self._build_fn = build_fn
        self._inst = None

    def start(self) -> None:
        self._inst = self._build_fn()

    def stop(self) -> None:
        self._inst = None

    def get_best_move(self, fen: str, time_s: float) -> chess.Move | None:
        inst = self._inst
        inst.board.set_position_from_fen(fen)
        move = inst.search.find_best_move(inst.board, 30, time_s)
        if move is None:
            return None
        return chess.Move.from_uci(move.to_uci_string())


class GMAdapter(_Adapter):
    """Wraps GMEngineInstance (GMSearch + NNUE speedster)."""

    def __init__(self, name: str = "gm_engine") -> None:
        self.name = name
        self._engine = None

    def start(self) -> None:
        from chesspoint72.eval_pipeline.gm_engine import build_gm_engine
        self._engine = build_gm_engine(time_per_move_s=30.0, tt_mb=256)

    def stop(self) -> None:
        self._engine = None

    def get_best_move(self, fen: str, time_s: float) -> chess.Move | None:
        eng = self._engine
        eng.board.set_position_from_fen(fen)
        move = eng.search.find_best_move(eng.board, 30, time_s)
        if move is None:
            return None
        return chess.Move.from_uci(move.to_uci_string())


# ── Engine registry ─────────────────────────────────────────────────────────── #

def _build_registry() -> dict[str, _Adapter]:
    """Build all available engines. Skips any that fail to import."""
    registry: dict[str, _Adapter] = {}

    def _try(name: str, adapter_fn: Callable) -> None:
        try:
            adapter_fn()          # test-import only
            registry[name] = adapter_fn()
        except Exception as exc:
            print(f"  [skip] {name}: {exc}", file=sys.stderr)

    # ── Main engine variants (from engine_config.py) ──────────────────────── #
    def _hce_baseline():
        from chesspoint72.eval_pipeline.engine_config import build_engine_for_test, BASELINE_B
        return InstanceAdapter("hce_baseline", lambda: build_engine_for_test(BASELINE_B))

    def _hce_full():
        from chesspoint72.eval_pipeline.engine_config import build_engine_for_test, HCE_FULL
        return InstanceAdapter("hce_full", lambda: build_engine_for_test(HCE_FULL))

    def _nnue_aspiration():
        from chesspoint72.eval_pipeline.engine_config import build_engine_for_test, TOURNAMENT_CONFIGS
        return InstanceAdapter("nnue_aspiration", lambda: build_engine_for_test(TOURNAMENT_CONFIGS["F_nnue_variant"]))

    # ── GM engine ─────────────────────────────────────────────────────────── #
    def _gm():
        return GMAdapter("gm_engine")

    # ── Frank engines ─────────────────────────────────────────────────────── #
    def _frank_v1():
        from chesspoint72.aiengines.frank.v1.engine import build_frank_controller
        return ControllerAdapter("frank_v1", build_frank_controller)

    def _frank_v3():
        from chesspoint72.aiengines.frank.v3.engine import build_controller
        return ControllerAdapter("frank_v3", build_controller)

    # ── Paul engines ──────────────────────────────────────────────────────── #
    def _paul_bullet():
        from chesspoint72.aiengines.paul.engine_bullet.engine_bullet import build_controller
        return ControllerAdapter("paul_bullet", build_controller)

    def _paul_cannon():
        from chesspoint72.aiengines.paul.engine_cannon.engine_cannon import build_controller
        return ControllerAdapter("paul_cannon", build_controller)

    def _paul_grinder():
        from chesspoint72.aiengines.paul.engine_grinder.engine_grinder import build_controller
        return ControllerAdapter("paul_grinder", build_controller)

    def _paul_chameleon():
        from chesspoint72.aiengines.paul.engine_chameleon.engine_chameleon import build_controller
        return ControllerAdapter("paul_chameleon", build_controller)

    def _paul_sentry():
        from chesspoint72.aiengines.paul.engine_sentry.engine_sentry import build_controller
        return ControllerAdapter("paul_sentry", build_controller)

    def _paul_classic():
        from chesspoint72.aiengines.paul.engine_classic.engine_classic import build_controller
        return ControllerAdapter("paul_classic", build_controller)

    # ── Victor ELO-ladder engines ──────────────────────────────────────────── #
    def _victor_v1():
        from chesspoint72.aiengines.victor.v1.engine import build_controller
        return ControllerAdapter("victor_v1_random", build_controller)

    def _victor_v2():
        from chesspoint72.aiengines.victor.v2.engine import build_controller
        return ControllerAdapter("victor_v2_oneply", build_controller)

    def _victor_v3():
        from chesspoint72.aiengines.victor.v3.engine import build_controller
        return ControllerAdapter("victor_v3_shallow", build_controller)

    def _victor_v4():
        from chesspoint72.aiengines.victor.v4.engine import build_controller
        return ControllerAdapter("victor_v4_positional", build_controller)

    def _victor_v5():
        from chesspoint72.aiengines.victor.v5.engine import build_controller
        return ControllerAdapter("victor_v5_tactical", build_controller)

    def _victor_v6():
        from chesspoint72.aiengines.victor.v6.engine import build_controller
        return ControllerAdapter("victor_v6_strategic", build_controller)

    def _victor_v7():
        from chesspoint72.aiengines.victor.v7.engine import build_controller
        return ControllerAdapter("victor_v7_neural", build_controller)

    for builder in [
        _hce_baseline, _hce_full, _nnue_aspiration, _gm,
        _frank_v1, _frank_v3,
        _paul_bullet, _paul_cannon, _paul_grinder,
        _paul_chameleon, _paul_sentry, _paul_classic,
        _victor_v1, _victor_v2, _victor_v3, _victor_v4,
        _victor_v5, _victor_v6, _victor_v7,
    ]:
        name = builder.__name__.lstrip("_")
        try:
            adapter = builder()
            registry[adapter.name] = adapter
        except Exception as exc:
            print(f"  [skip] {name}: {exc}", file=sys.stderr)

    return registry


# ── Game runner ─────────────────────────────────────────────────────────────── #

_OPENING_FENS = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",   # 1.e4
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",    # 1.d4
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # 1.e4 c5
    "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",  # 1.e4 Nf6
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # 1.e4 d5
    "rnbqkbnr/pppppppp/8/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 1",    # 1.d4 d5 → Catalan-ish
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", # 1.e4 e5 2.Nf3 Nc6
]


def play_game(
    white: _Adapter,
    black: _Adapter,
    time_s: float,
    opening_fen: str = chess.STARTING_FEN,
    max_plies: int = 300,
) -> float:
    """Play one game. Returns white's score (1.0 / 0.5 / 0.0)."""
    board = chess.Board(opening_fen)
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        engine = white if board.turn == chess.WHITE else black
        try:
            move = engine.get_best_move(board.fen(), time_s)
        except Exception:
            return 0.0 if board.turn == chess.WHITE else 1.0
        if move is None or move not in board.legal_moves:
            return 0.0 if board.turn == chess.WHITE else 1.0
        board.push(move)
    result = board.result(claim_draw=True)
    return {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5)


# ── ELO calculator ──────────────────────────────────────────────────────────── #

def calculate_elo(
    engines: list[str],
    pairings: list[PairingResult],
    iterations: int = 400,
) -> dict[str, float]:
    elo = {e: 0.0 for e in engines}
    for _ in range(iterations):
        for p in pairings:
            if p.total == 0:
                continue
            ea = 1 / (1 + 10 ** ((elo[p.b] - elo[p.a]) / 400))
            elo[p.a] += 32 * (p.score_a - ea)
            elo[p.b] += 32 * ((1 - p.score_a) - (1 - ea))
    mean = sum(elo.values()) / len(elo)
    return {e: round(v - mean, 1) for e, v in elo.items()}


# ── Main tournament ─────────────────────────────────────────────────────────── #

def run_tournament(
    engines: list[_Adapter],
    games_per_pairing: int = 10,
    time_s: float = 1.0,
) -> TournamentResult:
    names = [e.name for e in engines]
    n = len(engines)
    pairings_map: dict[tuple[str, str], PairingResult] = {}
    all_games: list[GameResult] = []

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    total_games = len(pairs) * games_per_pairing

    print(f"\n  Engines : {n}   Pairings: {len(pairs)}   Games: {total_games}   Time/move: {time_s}s")
    print()

    t0 = time.monotonic()
    game_num = 0

    for pair_idx, (i, j) in enumerate(pairs):
        ea, eb = engines[i], engines[j]
        key = (ea.name, eb.name)
        pr = PairingResult(a=ea.name, b=eb.name)
        pairings_map[key] = pr

        label = f"  [{pair_idx+1:2d}/{len(pairs)}]  {ea.name:22s} vs {eb.name:22s}  "
        sys.stdout.write(label)
        sys.stdout.flush()

        for g in range(games_per_pairing):
            opening = _OPENING_FENS[game_num % len(_OPENING_FENS)]
            # Alternate colours each game
            if g % 2 == 0:
                white, black = ea, eb
            else:
                white, black = eb, ea

            score = play_game(white, black, time_s, opening)

            # Accumulate from ea's perspective
            score_a = score if white is ea else (1.0 - score)
            if score_a == 1.0:
                pr.wins_a += 1
                sym = "W"
            elif score_a == 0.0:
                pr.wins_b += 1
                sym = "L"
            else:
                pr.draws += 1
                sym = "."

            all_games.append(GameResult(white=white.name, black=black.name, score=score))
            game_num += 1
            sys.stdout.write(sym)
            sys.stdout.flush()

        print(f"   {pr.wins_a}W {pr.draws}D {pr.wins_b}L")

    elapsed = time.monotonic() - t0
    elo = calculate_elo(names, list(pairings_map.values()))
    return TournamentResult(
        engines=names,
        pairings=list(pairings_map.values()),
        games=all_games,
        elo=elo,
        elapsed_s=elapsed,
    )


# ── Display ─────────────────────────────────────────────────────────────────── #

def print_leaderboard(result: TournamentResult) -> None:
    names = result.engines

    # Compute totals per engine
    stats: dict[str, dict] = {n: {"W": 0, "D": 0, "L": 0, "pts": 0.0, "gp": 0} for n in names}
    for g in result.games:
        stats[g.white]["gp"] += 1
        stats[g.black]["gp"] += 1
        if g.score == 1.0:
            stats[g.white]["W"] += 1;  stats[g.black]["L"] += 1
            stats[g.white]["pts"] += 1.0
        elif g.score == 0.0:
            stats[g.black]["W"] += 1;  stats[g.white]["L"] += 1
            stats[g.black]["pts"] += 1.0
        else:
            stats[g.white]["D"] += 1;  stats[g.black]["D"] += 1
            stats[g.white]["pts"] += 0.5;  stats[g.black]["pts"] += 0.5

    ranked = sorted(names, key=lambda n: result.elo[n], reverse=True)

    W = 75
    print()
    print("=" * W)
    print("  CHESSPOINT72 ENGINE TOURNAMENT -- FINAL LEADERBOARD")
    print("=" * W)
    hdr = f"  {'#':>3}  {'Engine':<24}  {'ELO':>6}  {'W':>5}  {'D':>5}  {'L':>5}  {'Score%':>7}"
    print(hdr)
    print("  " + "-" * (W - 2))
    for rank, name in enumerate(ranked, 1):
        s = stats[name]
        pct = 100 * s["pts"] / max(s["gp"], 1)
        elo = result.elo[name]
        sign = "+" if elo >= 0 else ""
        row = (
            f"  {rank:>3}  {name:<24}  {sign}{elo:>5.0f}  "
            f"{s['W']:>5}  {s['D']:>5}  {s['L']:>5}  {pct:>6.1f}%"
        )
        print(row)
    print("=" * W)

    mins = result.elapsed_s / 60
    print(f"\n  {len(result.games)} games played in {mins:.1f} min  ({result.elapsed_s/len(result.games):.1f}s avg/game)\n")


# ── CLI ─────────────────────────────────────────────────────────────────────── #

def main() -> int:
    ap = argparse.ArgumentParser(description="Chesspoint72 engine tournament")
    ap.add_argument("--games",  type=int,   default=10,  help="Games per pairing (default 10)")
    ap.add_argument("--time",   type=float, default=1.0, help="Seconds per move (default 1.0)")
    ap.add_argument("--smoke",  action="store_true",     help="Quick run: 4 games/pairing, 0.5s/move")
    ap.add_argument("--output", type=str,   default=None, help="Save JSON results to file")
    ap.add_argument("--engines", type=str,  default=None, help="Comma-separated engine names to include")
    args = ap.parse_args()

    if args.smoke:
        args.games = 4
        args.time  = 0.5

    print("\n  CHESSPOINT72 ENGINE TOURNAMENT")
    print("  ==============================")
    print("  Loading engines...")

    registry = _build_registry()

    if args.engines:
        selected = [n.strip() for n in args.engines.split(",")]
        registry = {k: v for k, v in registry.items() if k in selected}

    if len(registry) < 2:
        print("  Error: need at least 2 engines.", file=sys.stderr)
        return 1

    engines = list(registry.values())
    print(f"  Loaded {len(engines)} engines: {', '.join(e.name for e in engines)}")

    # Start all engines
    for e in engines:
        try:
            e.start()
        except Exception as exc:
            print(f"  [error] failed to start {e.name}: {exc}", file=sys.stderr)
            return 1

    try:
        result = run_tournament(engines, games_per_pairing=args.games, time_s=args.time)
    finally:
        for e in engines:
            try:
                e.stop()
            except Exception:
                pass

    print_leaderboard(result)

    if args.output:
        data = {
            "engines": result.engines,
            "elo": result.elo,
            "elapsed_s": result.elapsed_s,
            "pairings": [
                {"a": p.a, "b": p.b, "wins_a": p.wins_a, "draws": p.draws, "wins_b": p.wins_b}
                for p in result.pairings
            ],
        }
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
