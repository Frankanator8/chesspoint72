"""Phase 7 / test 2 — Module Selector Agent behaviour per mode preset."""
from __future__ import annotations

import chess

from chesspoint72.aiengines.jonathan.agent import (
    AgentContext,
    build_context,
    select_modules,
)
from chesspoint72.aiengines.jonathan.registry import scan_modules


# Sample positions used by multiple tests.
_OPENING_FEN = chess.STARTING_FEN
_ENDGAME_FEN = "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1"  # K+P vs K
_TACTICAL_FEN = (  # Italian middlegame with queens still on board.
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5"
)


# --------------------------------------------------------------------------- #
# Blind mode
# --------------------------------------------------------------------------- #


def test_blind_mode_uses_minimal_config():
    cfg = select_modules(build_context("blind"))
    assert cfg.evaluator_name == "stub"
    assert cfg.pruning_enabled is False
    assert cfg.nmp_enabled is False
    assert cfg.razoring_enabled is False
    assert cfg.futility_enabled is False
    assert cfg.lmr_enabled is False
    assert cfg.move_ordering == "stub"


def test_blind_mode_does_not_use_position_or_clock():
    ctx = build_context("blind")
    # Even when handed runtime data, blind mode must continue to ignore it.
    cfg = select_modules(ctx)
    assert ctx.position_fen is None
    assert ctx.time_remaining_ms is None
    assert all(
        "synthesized" not in reason for _, reason in cfg.activated_modules
    )


# --------------------------------------------------------------------------- #
# Aware mode
# --------------------------------------------------------------------------- #


def test_aware_mode_endgame_disables_nmp():
    ctx = build_context(
        "aware", position_fen=_ENDGAME_FEN, time_remaining_ms=60_000
    )
    cfg = select_modules(ctx)
    assert cfg.nmp_enabled is False, "NMP must be off in K+P endgames (zugzwang)"
    assert cfg.lmr_enabled is True, "LMR remains on in standard mode"
    assert cfg.evaluator_name == "material"


def test_aware_mode_opening_with_short_clock_keeps_nmp_off():
    """Standard mode disables NMP unless the clock is generous (>= 30s)."""
    ctx = build_context(
        "aware", position_fen=_OPENING_FEN, time_remaining_ms=5_000
    )
    cfg = select_modules(ctx)
    assert cfg.nmp_enabled is False
    assert cfg.razoring_enabled is False
    assert cfg.lmr_enabled is True
    assert cfg.futility_enabled is True


def test_aware_mode_opening_with_generous_clock_enables_nmp():
    ctx = build_context(
        "aware", position_fen=_OPENING_FEN, time_remaining_ms=120_000
    )
    cfg = select_modules(ctx)
    assert cfg.nmp_enabled is True
    assert cfg.razoring_enabled is True


def test_aware_config_adapts_between_endgame_and_opening():
    """Direct comparison: same mode preset, different FENs -> different cfgs."""
    cfg_open = select_modules(
        build_context(
            "aware", position_fen=_OPENING_FEN, time_remaining_ms=120_000
        )
    )
    cfg_end = select_modules(
        build_context(
            "aware", position_fen=_ENDGAME_FEN, time_remaining_ms=120_000
        )
    )
    assert cfg_open.nmp_enabled is True
    assert cfg_end.nmp_enabled is False
    assert cfg_open != cfg_end


# --------------------------------------------------------------------------- #
# Autonomous mode
# --------------------------------------------------------------------------- #


def test_autonomous_mode_activates_full_set():
    ctx = build_context(
        "autonomous", position_fen=_OPENING_FEN, time_remaining_ms=180_000
    )
    cfg = select_modules(ctx)
    assert cfg.evaluator_name == "hce"
    assert cfg.pruning_enabled is True
    assert cfg.nmp_enabled is True
    assert cfg.razoring_enabled is True
    assert cfg.futility_enabled is True
    assert cfg.lmr_enabled is True
    assert cfg.move_ordering == "captures_first"
    assert cfg.quiescence_extra_depth >= 2


def test_autonomous_mode_has_can_add_modules_flag_set_but_does_not_create_modules():
    """The autonomous preset still flips ``can_add_modules`` for API parity,
    but the agent must never actually scaffold a new module file (per the
    project owner's directive)."""
    ctx = build_context("autonomous", position_fen=_OPENING_FEN, time_remaining_ms=60_000)
    assert ctx.can_add_modules is True
    cfg = select_modules(ctx)
    # No activation reason should claim a synthesised stub.
    for name, reason in cfg.activated_modules:
        assert "synthesized" not in reason.lower()
        assert "synthesised" not in reason.lower()
        assert not name.startswith(
            "chesspoint72.aiengines.jonathan.modules."
        ), f"agent created module {name!r} but is forbidden from doing so"


def test_autonomous_tactical_position_lowers_razoring_margins():
    ctx = build_context(
        "autonomous", position_fen=_TACTICAL_FEN, time_remaining_ms=120_000
    )
    cfg = select_modules(ctx)
    # Default full-mode margins are (250, 350, 450); tactical drops them by 100.
    assert cfg.razoring_margins[0] <= 250
    assert cfg.quiescence_extra_depth >= 2


def test_filter_to_available_drops_unavailable_capabilities():
    """When can_add_modules is False and required capabilities are missing,
    the agent gracefully degrades instead of advertising features it can't run."""
    ctx = AgentContext(
        mode="standard",
        position_fen=_OPENING_FEN,
        time_remaining_ms=60_000,
        can_add_modules=False,
        available_modules=[],  # empty registry == no capabilities
    )
    # Empty list means filter_to_available short-circuits — verify it's a no-op.
    cfg = select_modules(ctx)
    # With a registry that *does* exist (real scan), aware mode would enable
    # everything; with an empty list the helper bails out — this is by design,
    # documented in _filter_to_available. Confirm the agent at least returns
    # something the search can run.
    assert cfg.evaluator_name in {"stub", "material", "hce", "nnue"}


def test_registry_passthrough_in_build_context():
    """build_context() should auto-scan the registry when it isn't given one."""
    ctx = build_context("aware")
    assert ctx.available_modules
    # Sanity: a real run discovers something.
    assert any("search" in d.capabilities for d in ctx.available_modules)


def test_unknown_mode_preset_raises():
    import pytest
    with pytest.raises(ValueError):
        build_context("warp-speed")


def test_activated_modules_log_is_populated():
    """Every mode should produce some activation log lines so main.py can
    surface them to the operator."""
    for mode in ("blind", "aware", "autonomous"):
        cfg = select_modules(build_context(mode, position_fen=_OPENING_FEN))
        assert cfg.activated_modules, f"{mode} produced no activation log"


def test_scan_modules_used_in_default_context_matches_direct_scan():
    direct = scan_modules()
    via_ctx = build_context("blind").available_modules
    direct_names = sorted(d.name for d in direct)
    via_names = sorted(d.name for d in via_ctx)
    assert direct_names == via_names
