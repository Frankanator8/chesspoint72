"""Tests for PruningConfig toggle helpers."""
from __future__ import annotations

from chesspoint72.engine.pruning.config import (
    PruningConfig,
    default_pruning_config,
    disable_futility,
    disable_lmr,
    disable_nmp,
    disable_razoring,
)

_TOGGLES = ("nmp_enabled", "futility_enabled", "razoring_enabled", "lmr_enabled")


def _assert_only_disabled(cfg: PruningConfig, expected_off: str) -> None:
    for flag in _TOGGLES:
        actual = getattr(cfg, flag)
        if flag == expected_off:
            assert actual is False, f"{flag} should be False after disable; got {actual!r}"
        else:
            assert actual is True, (
                f"{flag} should remain True when disabling {expected_off}; got {actual!r}"
            )


def test_toggle_helpers() -> None:
    base = default_pruning_config()
    for flag in _TOGGLES:
        assert getattr(base, flag) is True, f"default {flag} must be True"

    cases = [
        (disable_nmp, "nmp_enabled"),
        (disable_futility, "futility_enabled"),
        (disable_razoring, "razoring_enabled"),
        (disable_lmr, "lmr_enabled"),
    ]
    for helper, expected_off in cases:
        modified = helper(base)
        _assert_only_disabled(modified, expected_off)
        for flag in _TOGGLES:
            assert getattr(base, flag) is True, (
                f"base config was mutated by {helper.__name__} (flag={flag})"
            )
        assert modified is not base, f"{helper.__name__} returned the original object"
