"""Stage 8 — Tournament backtest.

Round-robin across the six named engine configs from EVAL_PIPELINE.md.
Every variant plays every other variant; colors alternate within each pairing.
Produces a full Elo ladder from the win/draw/loss matrix.

Variants (cumulative stack, each adds one component to the previous):
    A_baseline_b   = Baseline B (MoveSorterPolicy + HCE classic)
    B_plus_tt      = + depth-preferred TT
    C_plus_pruning = + NMP + LMR
    D_plus_asp     = + aspiration windows
    E_plus_hce     = + HCE full (all modules)
    F_nnue_variant = same stack but NNUE evaluator
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from itertools import combinations

from chesspoint72.eval_pipeline.ab_test import (
    ABTestResult,
    calculate_elo_with_ci,
    calculate_los,
    run_ab_test,
)
from chesspoint72.eval_pipeline.engine_config import TOURNAMENT_CONFIGS, EngineConfig

DEFAULT_GAMES_PER_PAIRING = 50


# --------------------------------------------------------------------------- #
# Elo ladder computation
# --------------------------------------------------------------------------- #

def _build_elo_ladder(
    scores: dict[tuple[str, str], tuple[int, int, int]],
) -> dict[str, float]:
    """Compute relative Elo ratings from W/D/L pairings using the Elo formula.

    All ratings are relative to the first entry (rated at 0).
    Returns {variant_name: elo_rating}.
    """
    names = list({name for pair in scores for name in pair})
    elo = {name: 0.0 for name in names}

    for _ in range(200):  # iterate to convergence
        for (a, b), (w, d, l) in scores.items():
            n = w + d + l
            if n == 0:
                continue
            score_a = (w + 0.5 * d) / n
            expected_a = 1.0 / (1.0 + 10.0 ** ((elo[b] - elo[a]) / 400.0))
            diff = score_a - expected_a
            elo[a] += 8.0 * diff
            elo[b] -= 8.0 * diff

    # Normalise so the lowest-rated variant is 0
    min_elo = min(elo.values())
    return {name: round(r - min_elo, 1) for name, r in elo.items()}


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass
class PairingResult:
    variant_a: str
    variant_b: str
    wins_a: int
    draws: int
    wins_b: int
    elo_ab: float | None

@dataclass
class Stage8Result:
    pairings: list[PairingResult] = field(default_factory=list)
    elo_ladder: dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def print_report(self) -> None:
        print("\n=== Stage 8 — Tournament Backtest ===")

        print("\n  Pairings:")
        for p in self.pairings:
            elo_str = f"{p.elo_ab:+.1f}" if p.elo_ab is not None else "N/A"
            print(
                f"    {p.variant_a:<20} vs {p.variant_b:<20} "
                f"W={p.wins_a} D={p.draws} L={p.wins_b}  Elo={elo_str}"
            )

        print("\n  Elo Ladder:")
        for name, elo in sorted(self.elo_ladder.items(), key=lambda x: -x[1]):
            print(f"    {name:<25} {elo:+.1f}")
        print(f"\n  ({self.elapsed_s:.0f}s)\n")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def run_stage8(
    configs: dict[str, EngineConfig] | None = None,
    games_per_pairing: int = DEFAULT_GAMES_PER_PAIRING,
    verbose: bool = True,
) -> Stage8Result:
    """Run a round-robin tournament across all engine variants.

    Args:
        configs:           Dict of {name: EngineConfig}.  Defaults to TOURNAMENT_CONFIGS.
        games_per_pairing: Games per pair (50 default; increase for better resolution).
        verbose:           Print progress.
    """
    cfgs = configs or TOURNAMENT_CONFIGS
    names = list(cfgs.keys())
    scores: dict[tuple[str, str], tuple[int, int, int]] = {}
    pairings: list[PairingResult] = []
    t0 = time.monotonic()

    if verbose:
        print(f"\nStage 8: round-robin ({len(names)} variants × {games_per_pairing} games/pair)...")

    for a_name, b_name in combinations(names, 2):
        cfg_a = cfgs[a_name]
        cfg_b = cfgs[b_name]
        label = f"{a_name} vs {b_name}"
        if verbose:
            print(f"  Running: {label}")

        ab = run_ab_test(
            config_a=cfg_a,
            config_b=cfg_b,
            n_games=games_per_pairing,
            label=label,
            verbose=False,
        )

        scores[(a_name, b_name)] = (ab.wins, ab.draws, ab.losses)
        pairings.append(PairingResult(
            variant_a=a_name, variant_b=b_name,
            wins_a=ab.wins, draws=ab.draws, wins_b=ab.losses,
            elo_ab=ab.elo,
        ))

        if verbose:
            elo_str = f"{ab.elo:+.1f}" if ab.elo else "N/A"
            print(f"    → W={ab.wins} D={ab.draws} L={ab.losses}  Elo={elo_str}  LOS={ab.los:.3f}")

    elo_ladder = _build_elo_ladder(scores)
    result = Stage8Result(
        pairings=pairings,
        elo_ladder=elo_ladder,
        elapsed_s=round(time.monotonic() - t0, 1),
    )

    if verbose:
        result.print_report()

    return result
