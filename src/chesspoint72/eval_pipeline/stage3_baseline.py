"""Stage 3 — True baseline establishment.

200-game sanity check: BASELINE_B (MoveSorterPolicy + HCE classic) vs
STUB_CONFIG (unsorted moves + HCE classic).

Expected Elo gain: +80 to +150.
Minimum acceptable: +40 Elo.

If the gain is below +40, the MoveSorterPolicy wiring is broken — stop and
debug the integration before any further testing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from chesspoint72.eval_pipeline.ab_test import ABTestResult, run_ab_test
from chesspoint72.eval_pipeline.engine_config import BASELINE_B, STUB_CONFIG

_MIN_ELO_GATE = 40.0   # minimum gain to consider the wiring correct
_EXPECTED_LO  = 80.0   # lower end of expected range
_EXPECTED_HI  = 150.0  # upper end of expected range


@dataclass
class Stage3Result:
    ab: ABTestResult
    gate_passed: bool
    message: str

    def print_report(self) -> None:
        print("\n=== Stage 3 — True Baseline Establishment ===")
        print(f"  {self.ab}")
        print(f"\n  Gate (>= +{_MIN_ELO_GATE} Elo): {'PASS' if self.gate_passed else 'FAIL — check wiring'}")
        print(f"  {self.message}\n")


def run_stage3(n_games: int = 200, verbose: bool = True) -> Stage3Result:
    """Run 200-game stub vs Baseline B sanity check.

    Returns Stage3Result. If ``gate_passed`` is False, the
    MoveSorterPolicy wiring in factory.py must be debugged before proceeding.
    """
    ab = run_ab_test(
        config_a=BASELINE_B,
        config_b=STUB_CONFIG,
        n_games=n_games,
        label="Baseline B vs Stub",
        verbose=verbose,
    )

    elo = ab.elo or 0.0
    gate_passed = elo >= _MIN_ELO_GATE

    if not gate_passed:
        message = (
            f"WIRING BROKEN — gain {elo:+.1f} Elo is below the {_MIN_ELO_GATE} Elo gate. "
            "Check that MoveSorterPolicy is the active ordering policy in build_controller."
        )
    elif elo < _EXPECTED_LO:
        message = f"Gain {elo:+.1f} Elo is below expected range [{_EXPECTED_LO}, {_EXPECTED_HI}]. Wiring OK but ordering weaker than expected."
    else:
        message = f"Gain {elo:+.1f} Elo is within expected range. Baseline B is confirmed."

    result = Stage3Result(ab=ab, gate_passed=gate_passed, message=message)
    if verbose:
        result.print_report()
    return result
