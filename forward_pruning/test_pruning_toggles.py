"""Standalone toggle test for forward_pruning_config.

Run directly:
    .venv/bin/python forward_pruning/test_pruning_toggles.py

No external dependencies — uses the built-in ``assert`` statement only.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run directly (not via pytest).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forward_pruning.forward_pruning_config import (
    PruningConfig,
    default_pruning_config,
    disable_futility,
    disable_lmr,
    disable_nmp,
    disable_razoring,
)


_TOGGLES = ("nmp_enabled", "futility_enabled", "razoring_enabled", "lmr_enabled")


def _assert_only_disabled(cfg: PruningConfig, expected_off: str) -> None:
    """Assert exactly ``expected_off`` is False; every other toggle stays True."""
    for flag in _TOGGLES:
        actual = getattr(cfg, flag)
        if flag == expected_off:
            assert actual is False, f"{flag} should be False after disable; got {actual!r}"
        else:
            assert actual is True, (
                f"{flag} should remain True when disabling {expected_off}; got {actual!r}"
            )


def main() -> int:
    # 1. Instantiate a default config — sanity-check the starting state.
    base = default_pruning_config()
    for flag in _TOGGLES:
        assert getattr(base, flag) is True, f"default {flag} must be True"

    # 2 + 3. Each disable helper flips exactly one flag, leaves the rest alone,
    # and does not mutate the original (frozen dataclass: replace -> new instance).
    cases = [
        (disable_nmp, "nmp_enabled"),
        (disable_futility, "futility_enabled"),
        (disable_razoring, "razoring_enabled"),
        (disable_lmr, "lmr_enabled"),
    ]
    for helper, expected_off in cases:
        modified = helper(base)
        _assert_only_disabled(modified, expected_off)
        # Confirm the helper is non-mutating: base must still be all-True.
        for flag in _TOGGLES:
            assert getattr(base, flag) is True, (
                f"base config was mutated by {helper.__name__} (flag={flag})"
            )
        # Confirm it returned a new object, not the same reference.
        assert modified is not base, f"{helper.__name__} returned the original object"

    print("All toggle tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
