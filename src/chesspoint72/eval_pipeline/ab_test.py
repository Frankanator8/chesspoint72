"""Corrected A/B test framework for the eval pipeline.

Fixes two bugs in the original pipeline document:
  1. LOS threshold was 0.75 — now 0.95 (preferred) / 0.90 (minimum)
  2. 300 games is insufficient — minimum 500 games per test

Implements:
  - calculate_los(wins, losses) — Likelihood of Superiority
  - calculate_elo_with_ci(wins, draws, losses) — Elo + 95% confidence interval
  - run_ab_test(config_a, config_b, ...) — full paired A/B match
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import scipy.stats as _stats

from chesspoint72.eval_pipeline.engine_config import EngineConfig, EngineInstance, build_engine_for_test
from chesspoint72.eval_pipeline.game_runner import OPENINGS, play_game


# --------------------------------------------------------------------------- #
# Statistical helpers
# --------------------------------------------------------------------------- #

def calculate_los(wins: int, losses: int) -> float:
    """Likelihood of Superiority — P(engine A is stronger than engine B).

    Uses the normal approximation to the binomial.  Returns 0.5 when there
    are no decisive games (pure draws).
    """
    n = wins + losses
    if n == 0:
        return 0.5
    return float(_stats.norm.cdf((wins - losses) / math.sqrt(n)))


def calculate_elo_with_ci(
    wins: int, draws: int, losses: int, confidence: float = 0.95
) -> tuple[float | None, float | None, float | None]:
    """Elo point estimate + confidence interval using the Wilson score method.

    Returns (elo, lo, hi) where lo/hi are the CI bounds in Elo.
    Returns (None, None, None) when the score is 0 or 1 (undefined Elo).
    """
    n = wins + draws + losses
    if n == 0:
        return None, None, None
    score = (wins + 0.5 * draws) / n
    if score <= 0 or score >= 1:
        return None, None, None

    elo = -400.0 * math.log10(1.0 / score - 1.0)
    z = float(_stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    z2 = z * z

    center = (score + z2 / (2 * n)) / (1 + z2 / n)
    margin = z * math.sqrt(score * (1 - score) / n + z2 / (4 * n * n)) / (1 + z2 / n)

    lo_score = max(center - margin, 1e-6)
    hi_score = min(center + margin, 1 - 1e-6)
    lo = -400.0 * math.log10(1.0 / hi_score - 1.0)
    hi = -400.0 * math.log10(1.0 / lo_score - 1.0)
    return round(elo, 1), round(lo, 1), round(hi, 1)


def _verdict(elo: float | None, los: float) -> str:
    if elo is None or elo <= 0:
        return "REJECT"
    if los >= 0.95:
        return "KEEP"
    if los >= 0.90:
        return "BORDERLINE — run 1000 games before deciding"
    if los >= 0.75:
        return "INCONCLUSIVE — run more games"
    return "REJECT"


# --------------------------------------------------------------------------- #
# ABTestResult
# --------------------------------------------------------------------------- #

@dataclass
class ABTestResult:
    label: str
    wins: int
    draws: int
    losses: int
    elo: float | None
    elo_ci: str
    los: float
    verdict: str
    elapsed_s: float

    def __str__(self) -> str:
        elo_str = f"{self.elo:+.1f}" if self.elo is not None else "N/A"
        return (
            f"[{self.label}] "
            f"W={self.wins} D={self.draws} L={self.losses} | "
            f"Elo {elo_str} {self.elo_ci} | LOS={self.los:.3f} | "
            f"{self.verdict} ({self.elapsed_s:.0f}s)"
        )

    def as_dict(self) -> dict:
        return {
            "label": self.label, "wins": self.wins, "draws": self.draws,
            "losses": self.losses, "elo": self.elo, "elo_ci": self.elo_ci,
            "los": self.los, "verdict": self.verdict, "elapsed_s": self.elapsed_s,
        }


# --------------------------------------------------------------------------- #
# Core A/B runner
# --------------------------------------------------------------------------- #

def run_ab_test(
    config_a: EngineConfig,
    config_b: EngineConfig,
    n_games: int = 500,
    openings: tuple[str, ...] | None = None,
    label: str | None = None,
    verbose: bool = True,
) -> ABTestResult:
    """Run a paired A/B match of *n_games* between two engine configs.

    Colors alternate every game to eliminate first-move advantage.
    Results are tallied from engine A's perspective (W = A wins, L = A loses).

    Args:
        config_a:  The "new" engine being tested.
        config_b:  The baseline engine.
        n_games:   Total games to play. Minimum 500 recommended.
        openings:  FEN strings for opening positions.  Defaults to OPENINGS.
        label:     Short name for this test (used in reporting).
        verbose:   Print progress after each game.

    Returns:
        ABTestResult with Elo, 95% CI, LOS, and verdict.
    """
    book = openings or OPENINGS
    test_label = label or f"{config_a.name} vs {config_b.name}"
    wins = draws = losses = 0
    t0 = time.monotonic()

    engine_a = build_engine_for_test(config_a)
    engine_b = build_engine_for_test(config_b)

    for i in range(n_games):
        opening = book[i % len(book)]
        a_is_white = (i % 2 == 0)

        if a_is_white:
            result = play_game(engine_a, engine_b, opening)
        else:
            result = play_game(engine_b, engine_a, opening)
            # Invert: swap perspective so result is always from A's side
            if result == "1-0":
                result = "0-1"
            elif result == "0-1":
                result = "1-0"

        if result == "1-0":
            wins += 1
        elif result == "0-1":
            losses += 1
        else:
            draws += 1

        if verbose and (i + 1) % 50 == 0:
            elo_est, _, _ = calculate_elo_with_ci(wins, draws, losses)
            los = calculate_los(wins, losses)
            elo_str = f"{elo_est:+.1f}" if elo_est is not None else "N/A"
            print(
                f"  [{test_label}] game {i+1}/{n_games} "
                f"W={wins} D={draws} L={losses} "
                f"Elo={elo_str} LOS={los:.3f}",
                flush=True,
            )

    elo, lo, hi = calculate_elo_with_ci(wins, draws, losses)
    los = calculate_los(wins, losses)
    ci_str = f"[{lo}, {hi}]" if lo is not None else "N/A"

    return ABTestResult(
        label=test_label,
        wins=wins, draws=draws, losses=losses,
        elo=elo, elo_ci=ci_str,
        los=round(los, 3),
        verdict=_verdict(elo, los),
        elapsed_s=round(time.monotonic() - t0, 1),
    )
