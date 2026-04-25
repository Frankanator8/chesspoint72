"""Configuration layer for the forward-pruning module.

Defines the PruningConfig struct, a default factory, and copy-on-write
helpers for disabling individual techniques. Contains *no* pruning logic —
the algorithms module reads these values, it does not depend on this
file's behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PruningConfig:
    """Tunable parameters consumed by the forward-pruning algorithms."""

    # Master toggles — one per technique.
    nmp_enabled: bool
    futility_enabled: bool
    razoring_enabled: bool
    lmr_enabled: bool

    # Null-move reduction depths.
    nmp_r_shallow: int  # used when remaining depth < 6
    nmp_r_deep: int     # used when remaining depth >= 6

    # Futility slack in centipawns at depth 1.
    futility_margin: int

    # Razoring per-depth slack. Indexed by ``depth - 2`` (depths 2, 3, 4).
    razoring_margins: tuple[int, int, int]

    # LMR gating thresholds.
    lmr_min_depth: int
    lmr_min_move_index: int


def default_pruning_config() -> PruningConfig:
    """Return a PruningConfig with all techniques enabled and conservative margins."""
    return PruningConfig(
        nmp_enabled=True,
        futility_enabled=True,
        razoring_enabled=True,
        lmr_enabled=True,
        nmp_r_shallow=2,
        nmp_r_deep=3,
        futility_margin=300,
        razoring_margins=(350, 450, 550),
        lmr_min_depth=3,
        lmr_min_move_index=3,
    )


def disable_nmp(config: PruningConfig) -> PruningConfig:
    return replace(config, nmp_enabled=False)


def disable_futility(config: PruningConfig) -> PruningConfig:
    return replace(config, futility_enabled=False)


def disable_razoring(config: PruningConfig) -> PruningConfig:
    return replace(config, razoring_enabled=False)


def disable_lmr(config: PruningConfig) -> PruningConfig:
    return replace(config, lmr_enabled=False)
