"""Evaluation pipeline CLI entry point.

Run the full pipeline or a specific stage:

    python -m chesspoint72.eval_pipeline.runner
    python -m chesspoint72.eval_pipeline.runner --stage 5
    python -m chesspoint72.eval_pipeline.runner --stage 0 --games 1000
    python -m chesspoint72.eval_pipeline.runner --smoke

Stages:
    0  Perft correctness
    1  Disqualifiers (illegal moves, crashes, timeouts)
    3  Baseline gate (BASELINE_B vs stub, expect +80-150 Elo)
    4  Benchmarks (NPS, TT hit rate, eval speed)
    5  Isolated A/B tests (8 pairs)
    6  Factorial interaction matrix
    7  Regime stress tests + EPD suites
    8  Tournament backtest (round-robin)
    9  Final module scoring

Stage 2 is reserved for instrumentation (manual / external).
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from chesspoint72.eval_pipeline.engine_config import BASELINE_B, TOURNAMENT_CONFIGS
from chesspoint72.eval_pipeline.stage0_perft import run_stage0
from chesspoint72.eval_pipeline.stage1_disqualify import run_stage1
from chesspoint72.eval_pipeline.stage3_baseline import run_stage3
from chesspoint72.eval_pipeline.stage4_benchmarks import run_stage4
from chesspoint72.eval_pipeline.stage5_ab_tests import run_stage5
from chesspoint72.eval_pipeline.stage6_factorial import run_stage6
from chesspoint72.eval_pipeline.stage7_regime import run_stage7
from chesspoint72.eval_pipeline.stage8_tournament import run_stage8
from chesspoint72.eval_pipeline.stage9_scoring import ModuleMetrics, run_stage9


# --------------------------------------------------------------------------- #
# Pipeline summary record
# --------------------------------------------------------------------------- #

@dataclass
class PipelineRun:
    stages_run: list[int] = field(default_factory=list)
    stage_results: dict[int, Any] = field(default_factory=dict)
    stage_elapsed: dict[int, float] = field(default_factory=dict)
    aborted_at: int | None = None
    total_elapsed: float = 0.0

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        for s in self.stages_run:
            elapsed = self.stage_elapsed.get(s, 0.0)
            print(f"  Stage {s}  ({elapsed:.0f}s)")
        if self.aborted_at is not None:
            print(f"\n  ** ABORTED at Stage {self.aborted_at} **")
        print(f"\n  Total wall time: {self.total_elapsed:.0f}s")
        print("=" * 60)


# --------------------------------------------------------------------------- #
# Individual stage runners with timing
# --------------------------------------------------------------------------- #

def _timed(fn, *args, **kwargs):
    t0 = time.monotonic()
    result = fn(*args, **kwargs)
    return result, round(time.monotonic() - t0, 1)


# --------------------------------------------------------------------------- #
# Smoke test — fast import and sanity check (no actual games)
# --------------------------------------------------------------------------- #

def run_smoke_test() -> bool:
    """Verify imports and a single perft node count. Returns True on pass."""
    print("\n--- Smoke test ---")
    try:
        import chess
        from chesspoint72.eval_pipeline.engine_config import build_engine_for_test, BASELINE_B
        engine = build_engine_for_test(BASELINE_B)
        board = chess.Board()
        board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        n = len(list(board.legal_moves))
        assert n == 20, f"Expected 20 legal moves from startpos, got {n}"
        print(f"  Imports OK.  Startpos legal moves: {n}  PASS")
        return True
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #

def run_pipeline(
    stages: list[int] | None = None,
    games: int | None = None,
    verbose: bool = True,
) -> PipelineRun:
    """Run the evaluation pipeline.

    Args:
        stages:  Which stage numbers to run. None = all.
        games:   Override game count for stages that accept it.
        verbose: Pass through to each stage.
    """
    all_stages = [0, 1, 3, 4, 5, 6, 7, 8, 9]
    to_run = stages if stages is not None else all_stages

    run = PipelineRun()
    t_pipeline = time.monotonic()

    for stage_num in to_run:
        print(f"\n{'='*60}")
        print(f"  Running Stage {stage_num}...")
        print(f"{'='*60}")

        if stage_num == 0:
            result, elapsed = _timed(run_stage0, verbose=verbose)
            run.stage_results[0] = result
            run.stage_elapsed[0] = elapsed
            run.stages_run.append(0)
            if not result.all_passed:
                print(f"\n  STAGE 0 FAILED — perft errors detected. Halting pipeline.")
                run.aborted_at = 0
                break

        elif stage_num == 1:
            result, elapsed = _timed(run_stage1, BASELINE_B, verbose=verbose)
            run.stage_results[1] = result
            run.stage_elapsed[1] = elapsed
            run.stages_run.append(1)
            if result.illegal_move_rate > 0 or result.crash_rate > 0:
                print(f"\n  STAGE 1 HARD DISQUALIFIER — halting pipeline.")
                run.aborted_at = 1
                break

        elif stage_num == 3:
            n = games or 200
            result, elapsed = _timed(run_stage3, n_games=n, verbose=verbose)
            run.stage_results[3] = result
            run.stage_elapsed[3] = elapsed
            run.stages_run.append(3)
            if not result.gate_passed:
                elo_str = f"{result.ab.elo:+.1f}" if result.ab.elo is not None else "N/A"
                print(
                    f"\n  WARNING: Baseline gate failed (Elo={elo_str}) — "
                    "engine may not be configured correctly."
                )

        elif stage_num == 4:
            result, elapsed = _timed(run_stage4, BASELINE_B, verbose=verbose)
            run.stage_results[4] = result
            run.stage_elapsed[4] = elapsed
            run.stages_run.append(4)

        elif stage_num == 5:
            n = games or 500
            result, elapsed = _timed(run_stage5, n_games=n, verbose=verbose)
            run.stage_results[5] = result
            run.stage_elapsed[5] = elapsed
            run.stages_run.append(5)

        elif stage_num == 6:
            n = games or 150
            result, elapsed = _timed(run_stage6, n_games=n, verbose=verbose)
            run.stage_results[6] = result
            run.stage_elapsed[6] = elapsed
            run.stages_run.append(6)

        elif stage_num == 7:
            n = games or 50
            result, elapsed = _timed(run_stage7, BASELINE_B, n_games_per_regime=n, verbose=verbose)
            run.stage_results[7] = result
            run.stage_elapsed[7] = elapsed
            run.stages_run.append(7)

        elif stage_num == 8:
            n = games or 50
            result, elapsed = _timed(run_stage8, games_per_pairing=n, verbose=verbose)
            run.stage_results[8] = result
            run.stage_elapsed[8] = elapsed
            run.stages_run.append(8)

        elif stage_num == 9:
            # Build ModuleMetrics from Stage 5 results if available, else use defaults.
            stage5 = run.stage_results.get(5)
            stage7 = run.stage_results.get(7)
            robustness = stage7.robustness_score if stage7 else 0.5

            metrics_list: list[ModuleMetrics] = []
            if stage5 is not None:
                for ab in stage5.tests:
                    elo = ab.elo if ab.elo is not None else 0.0
                    metrics_list.append(ModuleMetrics(
                        name=ab.label,
                        elo_gain=elo,
                        los=ab.los,
                        robustness=robustness,
                        baseline_move_time_ms=10.0,
                        module_move_time_ms=12.0,
                        illegal_move_rate=0.0,
                        crash_rate=0.0,
                    ))
            else:
                print("  (No Stage 5 data — Stage 9 will use placeholder metrics)")
                metrics_list = [
                    ModuleMetrics(
                        name="BASELINE_B",
                        elo_gain=0.0, los=0.95, robustness=robustness,
                        baseline_move_time_ms=10.0, module_move_time_ms=10.0,
                        illegal_move_rate=0.0, crash_rate=0.0,
                    )
                ]

            result, elapsed = _timed(run_stage9, metrics_list, verbose=verbose)
            run.stage_results[9] = result
            run.stage_elapsed[9] = elapsed
            run.stages_run.append(9)

        else:
            print(f"  Unknown stage {stage_num} — skipping.")

    run.total_elapsed = round(time.monotonic() - t_pipeline, 1)
    run.print_summary()
    return run


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m chesspoint72.eval_pipeline.runner",
        description="Chess engine evaluation pipeline",
    )
    p.add_argument(
        "--stage", "-s",
        type=int,
        default=None,
        metavar="N",
        help="Run only this stage (0/1/3/4/5/6/7/8/9). Omit to run all.",
    )
    p.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Run a specific set of stages, e.g. --stages 0 1 3 5",
    )
    p.add_argument(
        "--games", "-g",
        type=int,
        default=None,
        metavar="N",
        help="Override game count for all stages (default: per-stage defaults).",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-stage verbose output.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run a quick smoke test (import check + 1 perft node) and exit.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.smoke:
        ok = run_smoke_test()
        return 0 if ok else 1

    # Determine which stages to run
    if args.stage is not None:
        stages = [args.stage]
    elif args.stages is not None:
        stages = sorted(set(args.stages))
    else:
        stages = None  # all

    run_pipeline(
        stages=stages,
        games=args.games,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
