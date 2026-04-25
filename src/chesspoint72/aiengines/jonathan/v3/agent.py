"""Module Selector Agent for Calix v3 — local, deterministic rule table.

v1 expresses the "agent" as a single function with if/elif branches. v3
implements the same *rules/idea* as a table of independent rules:

- Start from a neutral config.
- Apply exactly one mode preset rule.
- Apply zero or more position/clock adjustment rules.
- Gate the result against the registry.

This is a different *implementation style* from v1/v2, but intentionally
produces the same decisions for the same inputs. No network / API calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Literal

from chesspoint72.aiengines.jonathan.v3.registry import (
    ModuleDescriptor,
    find_capability,
    scan_modules,
)


Mode = Literal["minimal", "standard", "full"]


# --------------------------------------------------------------------------- #
# Data structures (identical contract to v1 so callers are interchangeable)
# --------------------------------------------------------------------------- #


@dataclass
class AgentContext:
    """All information the agent gets at decision time."""

    mode: Mode
    position_fen: str | None
    time_remaining_ms: int | None
    can_add_modules: bool
    available_modules: list[ModuleDescriptor] = field(default_factory=list)


@dataclass
class EngineConfig:
    """Resolved configuration ready for the search factory."""

    evaluator_name: str
    hce_modules: str | None = None
    pruning_enabled: bool = False
    nmp_enabled: bool = False
    razoring_enabled: bool = False
    futility_enabled: bool = False
    lmr_enabled: bool = False
    razoring_margins: tuple[int, int, int] = (350, 450, 550)
    move_ordering: str = "stub"
    quiescence_extra_depth: int = 0
    default_depth: int = 4
    default_time: float = 5.0
    activated_modules: list[tuple[str, str]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Position inspection helpers (used both by the rule fallback and to enrich
# the selector with cheap structural hints)
# --------------------------------------------------------------------------- #


def _is_endgame(fen: str) -> bool:
    placement = fen.split(" ", 1)[0]
    minors_majors = sum(1 for ch in placement if ch in "QqRrBbNn")
    return minors_majors <= 6


def _is_tactical(fen: str) -> bool:
    placement = fen.split(" ", 1)[0]
    occupied = sum(1 for ch in placement if ch.isalpha())
    has_queen = "Q" in placement or "q" in placement
    return occupied >= 24 and has_queen


# --------------------------------------------------------------------------- #
# Rule table
# --------------------------------------------------------------------------- #


Rule = tuple[str, Callable[[AgentContext, EngineConfig], bool], Callable[[AgentContext, EngineConfig], None]]


def _clock_generous_ms(time_remaining_ms: int | None) -> bool:
    return time_remaining_ms is not None and time_remaining_ms >= 30_000


def _rule_mode_minimal(ctx: AgentContext, cfg: EngineConfig) -> bool:
    return ctx.mode == "minimal"


def _apply_mode_minimal(_ctx: AgentContext, cfg: EngineConfig) -> None:
    cfg.evaluator_name = "stub"
    cfg.hce_modules = None
    cfg.pruning_enabled = False
    cfg.nmp_enabled = False
    cfg.razoring_enabled = False
    cfg.futility_enabled = False
    cfg.lmr_enabled = False
    cfg.move_ordering = "stub"
    cfg.quiescence_extra_depth = 0
    cfg.activated_modules.extend(
        [
            ("stub_evaluator", "minimal mode"),
            ("stub_ordering", "minimal mode"),
            ("stub_pruning", "minimal mode"),
        ]
    )


def _rule_mode_standard(ctx: AgentContext, cfg: EngineConfig) -> bool:
    return ctx.mode == "standard"


def _apply_mode_standard(ctx: AgentContext, cfg: EngineConfig) -> None:
    cfg.evaluator_name = "material"
    cfg.hce_modules = None
    cfg.move_ordering = "captures_first"
    cfg.pruning_enabled = True
    cfg.futility_enabled = True
    cfg.lmr_enabled = True
    cfg.nmp_enabled = _clock_generous_ms(ctx.time_remaining_ms)
    cfg.razoring_enabled = _clock_generous_ms(ctx.time_remaining_ms)
    cfg.activated_modules.extend(
        [
            ("material_evaluator", "standard mode"),
            ("captures_first_ordering", "standard mode"),
            ("futility_pruning", "standard mode"),
            ("lmr", "standard mode"),
        ]
    )
    if cfg.nmp_enabled:
        cfg.activated_modules.append(("null_move_pruning", "clock generous (>= 30s)"))
    if cfg.razoring_enabled:
        cfg.activated_modules.append(("razoring", "clock generous (>= 30s)"))


def _rule_mode_full(ctx: AgentContext, cfg: EngineConfig) -> bool:
    return ctx.mode == "full"


def _apply_mode_full(_ctx: AgentContext, cfg: EngineConfig) -> None:
    cfg.evaluator_name = "hce"
    cfg.hce_modules = "all"
    cfg.move_ordering = "captures_first"
    cfg.pruning_enabled = True
    cfg.nmp_enabled = True
    cfg.razoring_enabled = True
    cfg.futility_enabled = True
    cfg.lmr_enabled = True
    cfg.razoring_margins = (250, 350, 450)
    cfg.quiescence_extra_depth = 2
    cfg.activated_modules.extend(
        [
            ("hce_evaluator_all_modules", "full mode"),
            ("captures_first_ordering", "full mode"),
            ("null_move_pruning", "full mode"),
            ("razoring", "full mode"),
            ("futility_pruning", "full mode"),
            ("lmr", "full mode"),
        ]
    )


def _rule_endgame(ctx: AgentContext, cfg: EngineConfig) -> bool:
    return ctx.position_fen is not None and _is_endgame(ctx.position_fen)


def _apply_endgame(_ctx: AgentContext, cfg: EngineConfig) -> None:
    cfg.nmp_enabled = False
    cfg.activated_modules.append(("null_move_pruning_disabled", "endgame: zugzwang risk"))
    if cfg.evaluator_name == "hce":
        cfg.hce_modules = "classic"
        cfg.activated_modules.append(("hce_classic_modules", "endgame: drop advanced features"))


def _rule_tactical(ctx: AgentContext, cfg: EngineConfig) -> bool:
    return ctx.position_fen is not None and _is_tactical(ctx.position_fen)


def _apply_tactical(_ctx: AgentContext, cfg: EngineConfig) -> None:
    cfg.razoring_margins = (
        max(150, cfg.razoring_margins[0] - 100),
        max(200, cfg.razoring_margins[1] - 100),
        max(250, cfg.razoring_margins[2] - 100),
    )
    cfg.quiescence_extra_depth = max(cfg.quiescence_extra_depth, 2)
    cfg.activated_modules.append(("razoring_margins_lowered", "tactical position"))


def _filter_to_available(cfg: EngineConfig, available: list[ModuleDescriptor]) -> EngineConfig:
    if not available:
        return cfg
    has_pruning = find_capability(available, "pruning") is not None
    has_ordering = (
        find_capability(available, "move_ordering") is not None
        or find_capability(available, "ordering") is not None
    )
    has_hce = find_capability(available, "hce") is not None
    has_nnue = find_capability(available, "nnue") is not None
    if not has_pruning:
        cfg.pruning_enabled = False
        cfg.nmp_enabled = cfg.razoring_enabled = False
        cfg.futility_enabled = cfg.lmr_enabled = False
    if not has_ordering and cfg.move_ordering != "stub":
        cfg.move_ordering = "stub"
    if cfg.evaluator_name == "hce" and not has_hce:
        cfg.evaluator_name = "material"
    if cfg.evaluator_name == "nnue" and not has_nnue:
        cfg.evaluator_name = "material"
    return cfg


def select_modules(ctx: AgentContext) -> EngineConfig:
    """Resolve an EngineConfig deterministically (no API, no memoisation)."""
    cfg = EngineConfig(evaluator_name="stub", default_depth=4, default_time=5.0)
    cfg.activated_modules.append(("calix_v3_selector", "local rule-table selector"))

    rules: list[Rule] = [
        ("mode=minimal", _rule_mode_minimal, _apply_mode_minimal),
        ("mode=standard", _rule_mode_standard, _apply_mode_standard),
        ("mode=full", _rule_mode_full, _apply_mode_full),
        ("endgame", _rule_endgame, _apply_endgame),
        ("tactical", _rule_tactical, _apply_tactical),
    ]

    # Exactly one mode rule must apply.
    mode_applied = False
    for name, pred, apply in rules[:3]:
        if pred(ctx, cfg):
            apply(ctx, cfg)
            mode_applied = True
            break
    if not mode_applied:
        raise ValueError(f"unknown agent mode: {ctx.mode!r}")

    # Apply adjustment rules.
    for _name, pred, apply in rules[3:]:
        if pred(ctx, cfg):
            apply(ctx, cfg)

    # Enforce pruning gating.
    if not cfg.pruning_enabled:
        cfg.nmp_enabled = False
        cfg.razoring_enabled = False
        cfg.futility_enabled = False
        cfg.lmr_enabled = False

    return _filter_to_available(cfg, ctx.available_modules)


def reset_cache() -> None:
    """Backward-compatible no-op (v3 no longer memoises)."""
    return None


def build_context(
    mode_name: str,
    *,
    position_fen: str | None = None,
    time_remaining_ms: int | None = None,
    available_modules: list[ModuleDescriptor] | None = None,
) -> AgentContext:
    if available_modules is None:
        available_modules = scan_modules()
    label = mode_name.strip().lower()
    if label == "blind":
        return AgentContext(
            mode="minimal",
            position_fen=None,
            time_remaining_ms=None,
            can_add_modules=False,
            available_modules=available_modules,
        )
    if label == "aware":
        return AgentContext(
            mode="standard",
            position_fen=position_fen,
            time_remaining_ms=time_remaining_ms,
            can_add_modules=False,
            available_modules=available_modules,
        )
    if label == "autonomous":
        return AgentContext(
            mode="full",
            position_fen=position_fen,
            time_remaining_ms=time_remaining_ms,
            can_add_modules=True,
            available_modules=available_modules,
        )
    raise ValueError(
        f"unknown agent mode preset: {mode_name!r} "
        "(expected one of: blind, aware, autonomous)"
    )


def with_runtime_position(
    ctx: AgentContext,
    *,
    position_fen: str | None,
    time_remaining_ms: int | None,
) -> AgentContext:
    if ctx.mode == "minimal":
        return ctx
    return replace(ctx, position_fen=position_fen, time_remaining_ms=time_remaining_ms)
