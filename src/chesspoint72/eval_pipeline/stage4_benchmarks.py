"""Stage 4 — Module-type benchmarks.

Characterises each module in isolation before competitive games.
Requires the Stage 2 instrumentation (NegamaxSearch.get_stats()).

4a — Search benchmarks: NPS, depth reached, branching factor, beta cutoff rate, TT hit rate.
4b — Eval benchmarks: sign agreement with reference, speed (evals/sec), phase-MAE.
4c — NNUE sub-pipeline: score bounds, quantization fidelity, accumulator delta correctness.

The NMP zugzwang guard verification test (from Stage 4a) is also included:
runs a suite of known zugzwang positions and checks that the engine's move
matches a ground-truth answer at depth 6.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean, stdev

import chess

from chesspoint72.eval_pipeline.engine_config import BASELINE_B, EngineConfig, build_engine_for_test

# --------------------------------------------------------------------------- #
# Benchmark positions
# --------------------------------------------------------------------------- #

BENCHMARK_POSITIONS: list[str] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQR1K1 w - - 0 8",
    "r3r1k1/pp3ppp/3p1n2/q3p3/2P1P3/1PN2P2/P5PP/R2QKB1R w KQ - 0 14",
    "6k1/pp4pp/2p2n2/3r4/3P1B2/5PP1/PP5P/3R2K1 w - - 0 25",
    "8/5pk1/6p1/7p/7P/6P1/5PK1/8 w - - 0 50",
]

# Zugzwang positions for NMP guard verification
ZUGZWANG_POSITIONS: list[tuple[str, str]] = [
    # (fen, expected_best_move_uci)
    ("8/8/p1p5/1p5p/1P5p/8/PPP2K1p/4R1rk w - - 0 1",       "e1e8"),
    ("8/6B1/p1p5/1p5p/1P5p/8/PPP2K1p/4R1rk w - - 0 1",     "e1e8"),
    ("1q1k4/2Rr4/8/2Q3K1/8/8/8/8 w - - 0 1",               "c5c6"),
]

PHASE_POSITIONS: dict[str, list[str]] = {
    "opening":    [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "rnbqkb1r/pppp1ppp/4pn2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
    ],
    "middlegame": [
        "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQR1K1 w - - 0 8",
        "r3r1k1/pp3ppp/3p1n2/q3p3/2P1P3/1PN2P2/P5PP/R2QKB1R w KQ - 0 14",
    ],
    "endgame":    [
        "6k1/pp4pp/2p2n2/3r4/3P1B2/5PP1/PP5P/3R2K1 w - - 0 25",
        "8/5pk1/6p1/7p/7P/6P1/5PK1/8 w - - 0 50",
    ],
}


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass
class SearchBenchmarkResult:
    nps_mean: float
    nps_stdev: float
    depth_mean: float
    branching_factor_mean: float
    beta_cutoff_rate_mean: float
    tt_hit_rate_mean: float
    elapsed_s: float

    def print_report(self, label: str = "") -> None:
        print(f"\n=== Stage 4a — Search Benchmarks{': ' + label if label else ''} ===")
        print(f"  NPS              {self.nps_mean:>12,.0f} ± {self.nps_stdev:,.0f}")
        print(f"  Depth reached    {self.depth_mean:>12.1f}")
        print(f"  Branching factor {self.branching_factor_mean:>12.2f}")
        print(f"  Beta cutoff rate {self.beta_cutoff_rate_mean:>12.4f}")
        print(f"  TT hit rate      {self.tt_hit_rate_mean:>12.4f}")
        print(f"  ({self.elapsed_s:.1f}s)\n")


@dataclass
class EvalBenchmarkResult:
    sign_agreement: float
    evals_per_second: float
    phase_mae: dict[str, float]
    elapsed_s: float

    def print_report(self, label: str = "") -> None:
        print(f"\n=== Stage 4b — Eval Benchmarks{': ' + label if label else ''} ===")
        print(f"  Sign agreement   {self.sign_agreement:.3f}")
        print(f"  Evals/second     {self.evals_per_second:,.0f}")
        for phase, mae in self.phase_mae.items():
            print(f"  MAE ({phase:<12}) {mae:.1f} cp")
        print(f"  ({self.elapsed_s:.1f}s)\n")


@dataclass
class ZugzwangGuardResult:
    total: int
    correct: int
    failure_rate: float
    passed: bool  # < 20% failure rate

    def print_report(self) -> None:
        print("\n=== Stage 4a — NMP Zugzwang Guard ===")
        print(f"  Correct: {self.correct}/{self.total}")
        print(f"  Failure rate: {self.failure_rate:.2f} ({'PASS' if self.passed else 'FAIL — guard may be broken'})\n")


# --------------------------------------------------------------------------- #
# Benchmark runners
# --------------------------------------------------------------------------- #

def run_search_benchmark(
    config: EngineConfig | None = None,
    positions: list[str] | None = None,
    time_budget: float = 1.0,
    verbose: bool = True,
) -> SearchBenchmarkResult:
    cfg = config or BASELINE_B
    pos = positions or BENCHMARK_POSITIONS
    engine = build_engine_for_test(cfg)
    search = engine.search

    nps_vals: list[float] = []
    depth_vals: list[float] = []
    bf_vals: list[float] = []
    bcr_vals: list[float] = []
    ttr_vals: list[float] = []
    t0_total = time.monotonic()

    for fen in pos:
        engine.board.set_position_from_fen(fen)
        t0 = time.monotonic()
        search.find_best_move(engine.board, max_depth=20, allotted_time=time_budget)
        elapsed = max(time.monotonic() - t0, 1e-6)
        stats = search.get_stats()

        nodes = stats["nodes"]
        depth = max(stats["depth_reached"], 1)
        nps_vals.append(nodes / elapsed)
        depth_vals.append(float(depth))
        bf_vals.append(nodes ** (1.0 / depth))
        bcr_vals.append(stats["beta_cutoffs"] / max(nodes, 1))
        ttr_vals.append(stats["tt_hits"] / max(stats["tt_lookups"], 1))

    result = SearchBenchmarkResult(
        nps_mean=mean(nps_vals),
        nps_stdev=stdev(nps_vals) if len(nps_vals) > 1 else 0.0,
        depth_mean=mean(depth_vals),
        branching_factor_mean=mean(bf_vals),
        beta_cutoff_rate_mean=mean(bcr_vals),
        tt_hit_rate_mean=mean(ttr_vals),
        elapsed_s=time.monotonic() - t0_total,
    )
    if verbose:
        result.print_report(cfg.name)
    return result


def run_eval_benchmark(
    config: EngineConfig | None = None,
    verbose: bool = True,
) -> EvalBenchmarkResult:
    """Measure evaluator speed and sign agreement vs. a material-count reference."""
    cfg = config or BASELINE_B
    engine = build_engine_for_test(cfg)
    search = engine.search
    evaluator = search.evaluator_reference

    from chesspoint72.engine.factory import _MaterialEvaluator
    ref_eval = _MaterialEvaluator()

    sign_matches: list[bool] = []
    speeds: list[float] = []
    phase_errors: dict[str, list[float]] = {k: [] for k in PHASE_POSITIONS}

    t0_total = time.monotonic()

    for phase, fens in PHASE_POSITIONS.items():
        for fen in fens:
            engine.board.set_position_from_fen(fen)
            t0 = time.monotonic()
            our_eval = evaluator.evaluate_position(engine.board)
            speeds.append(1.0 / max(time.monotonic() - t0, 1e-9))
            ref = ref_eval.evaluate_position(engine.board)
            sign_matches.append((our_eval > 0) == (ref > 0) or (our_eval == 0 and ref == 0))
            phase_errors[phase].append(abs(our_eval - ref))

    result = EvalBenchmarkResult(
        sign_agreement=mean(sign_matches),
        evals_per_second=mean(speeds),
        phase_mae={p: mean(v) if v else 0.0 for p, v in phase_errors.items()},
        elapsed_s=time.monotonic() - t0_total,
    )
    if verbose:
        result.print_report(cfg.name)
    return result


def run_zugzwang_guard_test(
    config: EngineConfig | None = None,
    verbose: bool = True,
) -> ZugzwangGuardResult:
    """Verify NMP zugzwang guard fires correctly on known positions."""
    cfg = config or BASELINE_B
    engine = build_engine_for_test(cfg)
    correct = 0

    for fen, expected_uci in ZUGZWANG_POSITIONS:
        engine.board.set_position_from_fen(fen)
        move = engine.search.find_best_move(engine.board, max_depth=6, allotted_time=5.0)
        if move and move.to_uci_string() == expected_uci:
            correct += 1

    total = len(ZUGZWANG_POSITIONS)
    failure_rate = (total - correct) / max(total, 1)
    result = ZugzwangGuardResult(
        total=total, correct=correct,
        failure_rate=failure_rate,
        passed=failure_rate < 0.20,
    )
    if verbose:
        result.print_report()
    return result


def run_stage4(config: EngineConfig | None = None, verbose: bool = True) -> dict:
    """Run all Stage 4 benchmarks and return a summary dict."""
    cfg = config or BASELINE_B
    search_bench = run_search_benchmark(cfg, verbose=verbose)
    eval_bench   = run_eval_benchmark(cfg, verbose=verbose)
    zz_guard     = run_zugzwang_guard_test(cfg, verbose=verbose)
    return {
        "search": search_bench,
        "eval":   eval_bench,
        "zugzwang_guard": zz_guard,
    }
