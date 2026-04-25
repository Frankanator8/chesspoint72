"""Stage 9 — Final module scoring (corrected formula).

Fixes two bugs in the original formula:
  Bug 1: LOS weight could exceed 1.0  →  now clamped to [0, 1]
  Bug 2: Efficiency ratio unnormalised →  now normalised against best-in-class

Composite score:
    score = 0.55 × elo_gain × los_weight
          + 0.25 × efficiency_norm × 100
          + 0.10 × robustness × 100
          + 0.10 × stability × 100

Hard disqualifications:
    illegal_move_rate > 0  →  score = -9999
    crash_rate > 0         →  score = -9999
    los < 0.90             →  score = 0  (inconclusive)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# Input data containers (fed from earlier stage results)
# --------------------------------------------------------------------------- #

@dataclass
class ModuleMetrics:
    """All metrics needed to score a single module."""
    name: str
    elo_gain: float           # from Stage 5 A/B test
    los: float                # from Stage 5 A/B test
    robustness: float         # from Stage 7 regime variance (0–1)
    baseline_move_time_ms: float  # from Stage 4 benchmark
    module_move_time_ms: float    # from Stage 4 benchmark
    illegal_move_rate: float  # from Stage 1
    crash_rate: float         # from Stage 1


# --------------------------------------------------------------------------- #
# Scoring functions
# --------------------------------------------------------------------------- #

def _los_weight(los: float) -> float:
    """LOS weight clamped to [0, 1]; threshold 0.90."""
    raw = (los - 0.90) / 0.09
    return max(0.0, min(1.0, raw))


def _efficiency_norm(
    elo_gain: float,
    added_ms: float,
    max_efficiency: float,
) -> float:
    """Normalised Elo/ms efficiency ratio."""
    raw = elo_gain / max(added_ms, 0.001)
    return raw / max(max_efficiency, 1e-6)


@dataclass
class ModuleScore:
    name: str
    raw_score: float
    elo_gain: float
    los: float
    los_weight: float
    efficiency_norm: float
    robustness: float
    stability: float
    verdict: str

    def __str__(self) -> str:
        return (
            f"{self.name:<25} "
            f"Elo={self.elo_gain:+.1f}  "
            f"LOS={self.los:.3f}  "
            f"LOS_w={self.los_weight:.3f}  "
            f"Eff={self.efficiency_norm:.3f}  "
            f"Rob={self.robustness:.3f}  "
            f"Score={self.raw_score:.2f}  "
            f"[{self.verdict}]"
        )


@dataclass
class Stage9Result:
    scores: list[ModuleScore] = field(default_factory=list)

    def print_report(self) -> None:
        print("\n=== Stage 9 — Final Module Scoring ===")
        print(
            f"\n  {'Module':<25} {'Elo':>8} {'LOS':>7} {'Eff':>7} "
            f"{'Rob':>7} {'Score':>8}  Verdict"
        )
        print("  " + "-" * 85)
        for s in sorted(self.scores, key=lambda x: -x.raw_score):
            elo_str = f"{s.elo_gain:+.1f}"
            print(
                f"  {s.name:<25} {elo_str:>8} {s.los:>7.3f} "
                f"{s.efficiency_norm:>7.3f} {s.robustness:>7.3f} "
                f"{s.raw_score:>8.2f}  {s.verdict}"
            )
        print()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def compute_module_score(
    metrics: ModuleMetrics,
    all_efficiency_ratios: dict[str, float],
) -> ModuleScore:
    """Compute the composite score for a single module.

    Args:
        metrics:                Per-module data from Stages 1, 4, 5, 7.
        all_efficiency_ratios:  {name: elo/ms} for all modules in this batch.
                                Used to normalise the efficiency component.
    """
    # Hard disqualifiers
    if metrics.illegal_move_rate > 0:
        return ModuleScore(
            name=metrics.name, raw_score=-9999.0,
            elo_gain=metrics.elo_gain, los=metrics.los,
            los_weight=0.0, efficiency_norm=0.0,
            robustness=metrics.robustness, stability=0.0,
            verdict="DISQUALIFIED (illegal moves)",
        )
    if metrics.crash_rate > 0:
        return ModuleScore(
            name=metrics.name, raw_score=-9999.0,
            elo_gain=metrics.elo_gain, los=metrics.los,
            los_weight=0.0, efficiency_norm=0.0,
            robustness=metrics.robustness, stability=0.0,
            verdict="DISQUALIFIED (crashes)",
        )
    if metrics.los < 0.90:
        return ModuleScore(
            name=metrics.name, raw_score=0.0,
            elo_gain=metrics.elo_gain, los=metrics.los,
            los_weight=0.0, efficiency_norm=0.0,
            robustness=metrics.robustness, stability=1.0,
            verdict="INCONCLUSIVE (LOS < 0.90 — run more games)",
        )

    added_ms = max(metrics.module_move_time_ms - metrics.baseline_move_time_ms, 0.001)
    max_eff = max(all_efficiency_ratios.values()) if all_efficiency_ratios else 1.0

    lw = _los_weight(metrics.los)
    eff_raw = metrics.elo_gain / added_ms
    eff_norm = eff_raw / max(max_eff, 1e-6)
    stability = 1.0 if metrics.crash_rate == 0.0 else 0.0

    score = (
        0.55 * metrics.elo_gain * lw
        + 0.25 * eff_norm * 100.0
        + 0.10 * metrics.robustness * 100.0
        + 0.10 * stability * 100.0
    )

    if metrics.los >= 0.95 and metrics.elo_gain > 0:
        verdict = "RETAIN"
    elif metrics.los >= 0.90 and metrics.elo_gain > 0:
        verdict = "BORDERLINE — run more games"
    else:
        verdict = "REJECT"

    return ModuleScore(
        name=metrics.name,
        raw_score=round(score, 2),
        elo_gain=metrics.elo_gain,
        los=metrics.los,
        los_weight=round(lw, 4),
        efficiency_norm=round(eff_norm, 4),
        robustness=round(metrics.robustness, 4),
        stability=stability,
        verdict=verdict,
    )


def run_stage9(
    metrics_list: list[ModuleMetrics],
    verbose: bool = True,
) -> Stage9Result:
    """Score all modules and produce the final leaderboard.

    Args:
        metrics_list:  One ModuleMetrics per module being evaluated.
        verbose:       Print the leaderboard table.
    """
    # Compute efficiency ratios for normalisation
    all_efficiency_ratios: dict[str, float] = {}
    for m in metrics_list:
        added_ms = max(m.module_move_time_ms - m.baseline_move_time_ms, 0.001)
        all_efficiency_ratios[m.name] = m.elo_gain / added_ms

    result = Stage9Result()
    for m in metrics_list:
        result.scores.append(compute_module_score(m, all_efficiency_ratios))

    if verbose:
        result.print_report()

    return result
