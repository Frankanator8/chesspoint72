"""Stage 7 — Regime stress tests.

Tests engine performance across position families where engines most commonly
degrade. High average Elo with high variance = unreliable module.

Phase regimes: opening, middlegame, endgame, transition
Tactical regimes: quiet, forcing, zugzwang, mating nets
Structural regimes: open_center, closed_center, isolated_queen_pawn, rook_endgame

EPD suites (WAC, ERET, Bratko-Kopec) measure tactical solving accuracy.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import chess

from chesspoint72.eval_pipeline.ab_test import ABTestResult, run_ab_test
from chesspoint72.eval_pipeline.engine_config import BASELINE_B, EngineConfig, build_engine_for_test

# --------------------------------------------------------------------------- #
# Regime position sets
# --------------------------------------------------------------------------- #

REGIME_POSITIONS: dict[str, list[str]] = {
    "opening": [
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbqkb1r/pppp1ppp/4pn2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 6 5",
    ],
    "middlegame": [
        "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQR1K1 w - - 0 8",
        "r3r1k1/pp3ppp/3p1n2/q3p3/2P1P3/1PN2P2/P5PP/R2QKB1R w KQ - 0 14",
        "2rq1rk1/pb1nbppp/1p2pn2/3p4/2PP4/1PN1PN2/PBQ1BPPP/R4RK1 w - - 0 12",
        "r2q1rk1/1p1nbppp/p2pbn2/4p3/4P3/1NN1BP2/PPPQ2PP/R3KB1R w KQ - 0 11",
    ],
    "endgame": [
        "6k1/pp4pp/2p2n2/3r4/3P1B2/5PP1/PP5P/3R2K1 w - - 0 25",
        "8/5pk1/6p1/7p/7P/6P1/5PK1/8 w - - 0 50",
        "4k3/8/3p4/3P4/8/8/8/4K3 w - - 0 1",
        "8/8/4k3/4P3/4K3/8/8/8 w - - 0 1",
    ],
    "transition": [
        "r1bq1rk1/pp3ppp/2n1pn2/3p4/3P4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 8",
        "2r1r1k1/pp3ppp/2n1bn2/q1pp4/8/1PN1P1P1/PBP1QP1P/R4RK1 w - - 0 15",
    ],
    "quiet": [
        "r4rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2QR1K1 w - - 8 10",
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 0 9",
    ],
    "forcing": [
        "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
        "r3k2r/ppp2ppp/2n1bn2/1B1pp3/4P1q1/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 8",
    ],
    "zugzwang": [
        "8/8/p1p5/1p5p/1P5p/8/PPP2K1p/4R1rk w - - 0 1",
        "1q1k4/2Rr4/8/2Q3K1/8/8/8/8 w - - 0 1",
    ],
    "mating_nets": [
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4",
        "r1bqk2r/1ppp1ppp/p1n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6",
    ],
    "rook_endgame": [
        "8/8/3k4/8/3K4/8/8/R7 w - - 0 1",
        "8/r7/3k4/8/3K4/8/8/4R3 w - - 0 1",
    ],
    "open_center": [
        "r1bqkb1r/pp3ppp/2n1pn2/2pp4/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    ],
    "isolated_queen_pawn": [
        "r1b1qrk1/pp3ppp/2n2n2/3p4/1b1P4/2NBP3/PP2NPPP/R1BQ1RK1 w - - 0 10",
    ],
}


# --------------------------------------------------------------------------- #
# EPD test suites
# --------------------------------------------------------------------------- #

EPD_SUITES: dict[str, str] = {
    "WAC":          "eval_pipeline/epd/win_at_chess.epd",
    "ERET":         "eval_pipeline/epd/eigenmann_rapid_engine.epd",
    "bratko_kopec": "eval_pipeline/epd/bratko_kopec.epd",
}


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass
class RegimeResult:
    regime: str
    n_games: int
    elo: float | None
    los: float

@dataclass
class EPDResult:
    suite: str
    total: int
    passed: int
    accuracy: float

@dataclass
class Stage7Result:
    regime_results: list[RegimeResult] = field(default_factory=list)
    epd_results: list[EPDResult] = field(default_factory=list)
    robustness_score: float = 0.0

    def print_report(self) -> None:
        print("\n=== Stage 7 — Regime Stress Tests ===")
        print("\n  Regime Elo vs Baseline B:")
        for r in self.regime_results:
            elo_str = f"{r.elo:+.1f}" if r.elo is not None else "N/A"
            print(f"    {r.regime:<25} Elo={elo_str:>8}  LOS={r.los:.3f}  (n={r.n_games})")
        print(f"\n  Robustness score: {self.robustness_score:.3f}")
        if self.epd_results:
            print("\n  EPD tactical suites:")
            for e in self.epd_results:
                print(f"    {e.suite:<20} {e.passed}/{e.total}  ({e.accuracy:.1f}%)")
        print()


# --------------------------------------------------------------------------- #
# Robustness score
# --------------------------------------------------------------------------- #

def _robustness_score(regime_elos: list[float]) -> float:
    """Lower variance across regimes = higher robustness. Range: [0, 1]."""
    if not regime_elos:
        return 0.0
    m = sum(regime_elos) / len(regime_elos)
    variance = sum((e - m) ** 2 for e in regime_elos) / len(regime_elos)
    return round(1.0 / (1.0 + math.sqrt(variance)), 4)


# --------------------------------------------------------------------------- #
# EPD runner (requires an EPD file on disk)
# --------------------------------------------------------------------------- #

def run_epd_suite(
    config: EngineConfig,
    suite_path: str,
    time_per_position_s: float = 1.0,
) -> EPDResult | None:
    path = Path(suite_path)
    if not path.exists():
        return None

    engine = build_engine_for_test(config)
    passed = total = 0

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 4)
        if len(parts) < 4:
            continue
        board_fen = " ".join(parts[:4]) + " 0 1"
        ops_blob = parts[4] if len(parts) == 5 else ""
        ops = {}
        for chunk in ops_blob.split(";"):
            chunk = chunk.strip()
            if chunk:
                head, _, tail = chunk.partition(" ")
                ops[head] = tail.strip().strip('"')
        bm = ops.get("bm")
        if not bm:
            continue

        board = chess.Board(board_fen)
        best_sans = bm.split()
        expected_ucis = set()
        for san in best_sans:
            try:
                expected_ucis.add(board.parse_san(san).uci())
            except Exception:
                pass

        engine.board.set_position_from_fen(board_fen)
        try:
            move = engine.search.find_best_move(
                engine.board, max_depth=20, allotted_time=time_per_position_s,
            )
        except Exception:
            move = None

        total += 1
        if move and move.to_uci_string() in expected_ucis:
            passed += 1

    return EPDResult(
        suite=path.stem,
        total=total,
        passed=passed,
        accuracy=round(100.0 * passed / max(total, 1), 2),
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def run_stage7(
    config: EngineConfig | None = None,
    n_games_per_regime: int = 50,
    run_epd: bool = True,
    verbose: bool = True,
) -> Stage7Result:
    """Run all regime stress tests against BASELINE_B."""
    cfg = config or BASELINE_B
    result = Stage7Result()

    if verbose:
        print(f"\nStage 7: regime stress tests for {cfg.name}...")

    regime_elos: list[float] = []
    for regime, positions in REGIME_POSITIONS.items():
        ab = run_ab_test(
            config_a=cfg,
            config_b=BASELINE_B,
            n_games=n_games_per_regime,
            openings=tuple(positions),
            label=f"regime:{regime}",
            verbose=False,
        )
        rr = RegimeResult(regime=regime, n_games=n_games_per_regime, elo=ab.elo, los=ab.los)
        result.regime_results.append(rr)
        if ab.elo is not None:
            regime_elos.append(ab.elo)
        if verbose:
            elo_str = f"{ab.elo:+.1f}" if ab.elo else "N/A"
            print(f"  {regime:<25} Elo={elo_str}  LOS={ab.los:.3f}")

    result.robustness_score = _robustness_score(regime_elos)

    if run_epd:
        for suite_name, path in EPD_SUITES.items():
            epd = run_epd_suite(cfg, path)
            if epd is not None:
                result.epd_results.append(epd)
                if verbose:
                    print(f"  EPD {suite_name:<18} {epd.passed}/{epd.total} ({epd.accuracy:.1f}%)")
            elif verbose:
                print(f"  EPD {suite_name:<18} (file not found: {path})")

    if verbose:
        result.print_report()

    return result
