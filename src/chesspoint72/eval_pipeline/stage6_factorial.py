"""Stage 6 — Factorial interaction matrix.

Tests every combination of modules and computes marginal contributions.
Alpha-beta and move ordering are multiplicative (not additive) — a module
that adds +20 Elo alone may add +40 Elo when combined with another because
better ordering makes pruning more effective.

Marginal contribution of module M = Elo(full stack) - Elo(full stack without M).
  Positive  → module earns its place
  Negative  → module is actively hurting when combined (interaction conflict)
  Large synergy → these modules must always be deployed together

Uses 150 games per pairing (sufficient for marginal contribution estimates;
not enough for individual accept/reject decisions).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from chesspoint72.eval_pipeline.ab_test import run_ab_test
from chesspoint72.eval_pipeline.engine_config import BASELINE_B, EngineConfig

# Module names that can be toggled on/off relative to a base config
MODULES = [
    "move_ordering",       # movesorter vs stub
    "transposition_table", # always_replace (enabled) vs NullTT
    "nmp",                 # nmp_enabled True vs False
    "lmr",                 # lmr_enabled True vs False
    "aspiration_windows",  # aspiration True vs False
    "hce_full",            # hce_modules "all" vs "classic"
]

DEFAULT_GAMES_PER_PAIR = 150


def _build_config_from_modules(active: tuple[str, ...]) -> EngineConfig:
    """Build an EngineConfig with exactly the listed modules active."""
    return EngineConfig(
        name=f"combo({'_'.join(active) if active else 'none'})",
        evaluator="hce",
        hce_modules="all" if "hce_full" in active else "classic",
        ordering="movesorter" if "move_ordering" in active else "stub",
        use_tt="transposition_table" in active,
        tt_policy="always_replace",
        nmp_enabled="nmp" in active,
        lmr_enabled="lmr" in active,
        razoring_enabled=False,
        futility_enabled=False,
        aspiration_windows="aspiration_windows" in active,
        depth=4,
        time_limit=1.0,
    )


@dataclass
class FactorialResult:
    combo_elos: dict[tuple[str, ...], float | None] = field(default_factory=dict)
    marginal_contributions: dict[str, float | None] = field(default_factory=dict)

    def print_report(self) -> None:
        print("\n=== Stage 6 — Factorial Interaction Matrix ===")

        print("\n  Module combinations (Elo vs baseline):")
        for combo, elo in sorted(self.combo_elos.items(), key=lambda x: (len(x[0]), x[1] or 0)):
            mods = ", ".join(combo) if combo else "(none — baseline)"
            elo_str = f"{elo:+.1f}" if elo is not None else "N/A"
            print(f"    [{elo_str:>8}]  {mods}")

        print("\n  Marginal contributions (full stack Δ from removing each module):")
        for module, mc in sorted(
            self.marginal_contributions.items(),
            key=lambda x: (x[1] or 0), reverse=True,
        ):
            mc_str = f"{mc:+.1f}" if mc is not None else "N/A"
            flag = "" if mc is None else " ← KEEP" if mc > 0 else " ← CONFLICT (remove)"
            print(f"    {module:<25} {mc_str:>8}{flag}")
        print()


def run_stage6(
    modules: list[str] | None = None,
    n_games: int = DEFAULT_GAMES_PER_PAIR,
    verbose: bool = True,
) -> FactorialResult:
    """Run all module combinations and compute marginal contributions.

    Args:
        modules:  Which modules to include in the matrix.  Defaults to MODULES.
        n_games:  Games per pairing.  150 is sufficient for marginal estimates.
        verbose:  Print progress.
    """
    mods = modules or MODULES
    result = FactorialResult()

    # All combinations from 0 to len(mods) modules active
    all_combos: list[tuple[str, ...]] = [()]
    for r in range(1, len(mods) + 1):
        for combo in combinations(mods, r):
            all_combos.append(combo)

    if verbose:
        print(f"\nStage 6: running {len(all_combos)} combinations × {n_games} games each...")

    for combo in all_combos:
        candidate = _build_config_from_modules(combo)
        ab = run_ab_test(
            config_a=candidate,
            config_b=BASELINE_B,
            n_games=n_games,
            label=f"combo({'+'.join(combo) if combo else 'none'})",
            verbose=False,
        )
        result.combo_elos[combo] = ab.elo
        if verbose:
            elo_str = f"{ab.elo:+.1f}" if ab.elo else "N/A"
            print(f"  [{elo_str:>8}] {', '.join(combo) if combo else '(none)'}")

    # Marginal contributions from the full stack
    full_stack = tuple(mods)
    full_elo = result.combo_elos.get(full_stack)

    for module in mods:
        without = tuple(m for m in mods if m != module)
        without_elo = result.combo_elos.get(without)
        if full_elo is not None and without_elo is not None:
            result.marginal_contributions[module] = round(full_elo - without_elo, 1)
        else:
            result.marginal_contributions[module] = None

    if verbose:
        result.print_report()

    return result
