"""Configuration layer for the forward-pruning module.

Defines the PruningConfig struct, a default factory, and copy-on-write
helpers for disabling individual techniques. Contains *no* pruning logic —
the search/pruning module reads these values, it does not depend on this
file's behaviour.

Conforms to ``chesspoint72/forward_pruning/INTERFACE_CONTRACT.md``. Field
names and types must match the contract exactly; renaming or retyping
anything here silently breaks ``chesspoint72/forward_pruning/pruning.py``.
"""
from __future__ import annotations

# Section 0 — Imports.
# dataclasses gives us the frozen struct + ``replace`` for the disable helpers.
from dataclasses import dataclass, replace


# Section 1 — PruningConfig struct.
# Frozen so the contract's "must not be mutated at runtime" invariant is
# enforced by the type system rather than by convention. Mutations go
# through the disable_* helpers, which return a new instance.
@dataclass(frozen=True)
class PruningConfig:
    """Tunable parameters consumed by the forward-pruning module.

    See ``INTERFACE_CONTRACT.md`` for field-by-field semantics.
    """

    # Master toggles — one per technique.
    nmp_enabled: bool
    futility_enabled: bool
    razoring_enabled: bool
    lmr_enabled: bool

    # Null-move reduction depths. The current pruning module hardcodes
    # 2/3 with the same depth-<6 split; these fields are exposed so the
    # next iteration of the search can read them instead of literals.
    nmp_r_shallow: int  # used when remaining depth < 6
    nmp_r_deep: int     # used when remaining depth >= 6

    # Futility slack in centipawns at depth 1.
    futility_margin: int

    # Razoring per-depth slack. Indexed by ``depth - 2`` (depths 2, 3, 4).
    # Tuple, not list, to keep the structure immutable end-to-end.
    razoring_margins: tuple[int, int, int]

    # LMR gating thresholds.
    lmr_min_depth: int
    lmr_min_move_index: int


# Section 2 — Default factory.
# These are the starting values specified by the configuration prompt. They
# intentionally diverge from the contract's "typical" examples (the contract
# called out 200 / (300, 500, 900); this layer ships 300 / (350, 450, 550)) —
# the contract spec is field shape, not exact numbers.
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


# Section 3 — Toggle helpers.
# Each returns a new PruningConfig with exactly one technique disabled; the
# original is untouched. Designed for A/B isolation tests in the harness.
def disable_nmp(config: PruningConfig) -> PruningConfig:
    return replace(config, nmp_enabled=False)


def disable_futility(config: PruningConfig) -> PruningConfig:
    return replace(config, futility_enabled=False)


def disable_razoring(config: PruningConfig) -> PruningConfig:
    return replace(config, razoring_enabled=False)


def disable_lmr(config: PruningConfig) -> PruningConfig:
    return replace(config, lmr_enabled=False)
