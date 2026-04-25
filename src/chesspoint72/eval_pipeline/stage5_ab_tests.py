"""Stage 5 — Isolated A/B tests (corrected thresholds).

All tests are vs. BASELINE_B (MoveSorterPolicy + HCE classic + always-replace TT).

Standard pairs (5 from the original doc):
  5-1  Move ordering quality       baseline_b vs no_ordering
  5-2  TT caching                  +TT_enabled vs no_tt
  5-3  HCE full vs material-only   hce_full vs hce_material_only
  5-4  NMP pruning                 baseline_b vs no_nmp
  5-5  LMR                         baseline_b vs no_lmr

New tests (3 not in original doc):
  5-A  MovePicker vs MoveSorter    movepicker vs baseline_b
  5-B  Aspiration windows          aspiration vs baseline_b
  5-C  Depth-preferred TT          depth_preferred_tt vs baseline_b

Corrected thresholds:
  KEEP:          LOS >= 0.95 AND Elo > 0
  BORDERLINE:    LOS >= 0.90 AND Elo > 0 (run 1000 games)
  INCONCLUSIVE:  LOS >= 0.75            (run more games)
  REJECT:        LOS <  0.75 OR Elo <= 0
"""
from __future__ import annotations

from dataclasses import dataclass, field

from chesspoint72.eval_pipeline.ab_test import ABTestResult, run_ab_test
from chesspoint72.eval_pipeline.engine_config import (
    ASPIRATION_CONFIG,
    BASELINE_B,
    DEPTH_PREFERRED_TT,
    HCE_FULL,
    HCE_MATERIAL_ONLY,
    MOVEPICKER_CONFIG,
    NO_LMR,
    NO_NMP,
    NO_ORDERING,
    NO_TT,
)

# Minimum games per test (from corrected pipeline doc)
DEFAULT_GAMES = 500


@dataclass
class Stage5Results:
    tests: list[ABTestResult] = field(default_factory=list)

    def print_report(self) -> None:
        print("\n=== Stage 5 — Isolated A/B Tests ===")
        header = f"{'Test':<35} {'Elo':>8} {'95% CI':>18} {'LOS':>7} {'Verdict'}"
        print(f"  {header}")
        print("  " + "-" * 90)
        for r in self.tests:
            elo_str = f"{r.elo:+.1f}" if r.elo is not None else "N/A"
            print(f"  {r.label:<35} {elo_str:>8} {r.elo_ci:>18} {r.los:>7.3f}  {r.verdict}")
        print()


def run_stage5(n_games: int = DEFAULT_GAMES, verbose: bool = True) -> Stage5Results:
    """Run all 8 Stage 5 A/B tests and return the collected results."""
    results = Stage5Results()

    pairs = [
        # Standard pairs
        (BASELINE_B,         NO_ORDERING,        "5-1 Move ordering (B vs stub)"),
        (BASELINE_B,         NO_TT,              "5-2 TT caching (enabled vs disabled)"),
        (HCE_FULL,           HCE_MATERIAL_ONLY,  "5-3 HCE full vs material-only"),
        (BASELINE_B,         NO_NMP,             "5-4 NMP (enabled vs disabled)"),
        (BASELINE_B,         NO_LMR,             "5-5 LMR (enabled vs disabled)"),
        # New tests
        (MOVEPICKER_CONFIG,  BASELINE_B,         "5-A MovePicker vs MoveSorter"),
        (ASPIRATION_CONFIG,  BASELINE_B,         "5-B Aspiration windows vs baseline"),
        (DEPTH_PREFERRED_TT, BASELINE_B,         "5-C Depth-preferred TT vs always-replace"),
    ]

    for cfg_a, cfg_b, label in pairs:
        if verbose:
            print(f"\nRunning: {label} ({n_games} games)...")
        result = run_ab_test(
            config_a=cfg_a,
            config_b=cfg_b,
            n_games=n_games,
            label=label,
            verbose=verbose,
        )
        results.tests.append(result)
        if verbose:
            print(f"  → {result}")

    if verbose:
        results.print_report()

    return results
