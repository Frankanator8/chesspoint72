"""Module Selector Agent for Calix v2 — local, deterministic selector.

v1 implements module selection as a direct rule cascade. v2 keeps the same
*rules/idea* (three modes + clock + position heuristics + registry gating),
but expresses the decision as a tiny "utility scoring" problem:

- Each capability (eval, ordering, pruning toggles) earns score based on
  mode + hints (clock/endgame/tactical).
- The final configuration is derived from those scores and then filtered
  against the registry so we never request unavailable capabilities.

This module performs **no network calls** and never invokes any external
LLM/API.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from chesspoint72.aiengines.jonathan.v2.registry import (
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
# Scoring-based selector (same intent as v1, different implementation style)
# --------------------------------------------------------------------------- #


def _clock_generous_ms(time_remaining_ms: int | None) -> bool:
    return time_remaining_ms is not None and time_remaining_ms >= 30_000


def _pick_mode_defaults(ctx: AgentContext, cfg: EngineConfig) -> None:
    """Apply mode defaults using utility scoring.

    This intentionally resolves to the same choices as v1 for the same
    context, but without a rigid if/elif cascade.
    """
    # Evaluator
    eval_scores: dict[str, int] = {
        "stub": 0,
        "material": 0,
        "hce": 0,
        "nnue": 0,
    }
    if ctx.mode == "minimal":
        eval_scores["stub"] += 10
    elif ctx.mode == "standard":
        eval_scores["material"] += 10
    elif ctx.mode == "full":
        eval_scores["hce"] += 10
    else:
        raise ValueError(f"unknown agent mode: {ctx.mode!r}")
    cfg.evaluator_name = max(eval_scores, key=eval_scores.get)

    # Move ordering
    ordering_scores = {"stub": 0, "captures_first": 0}
    if ctx.mode == "minimal":
        ordering_scores["stub"] += 10
    else:
        ordering_scores["captures_first"] += 10
    cfg.move_ordering = max(ordering_scores, key=ordering_scores.get)

    # Pruning + toggles
    pruning_score = 0
    if ctx.mode in ("standard", "full"):
        pruning_score += 10
    cfg.pruning_enabled = pruning_score > 0
    if not cfg.pruning_enabled:
        return

    # Standard: futility+LMR always; NMP/razoring only with generous clock.
    if ctx.mode == "standard":
        cfg.futility_enabled = True
        cfg.lmr_enabled = True
        if _clock_generous_ms(ctx.time_remaining_ms):
            cfg.nmp_enabled = True
            cfg.razoring_enabled = True
    # Full: everything, aggressive margins and deeper qsearch.
    elif ctx.mode == "full":
        cfg.nmp_enabled = True
        cfg.razoring_enabled = True
        cfg.futility_enabled = True
        cfg.lmr_enabled = True
        cfg.razoring_margins = (250, 350, 450)
        cfg.quiescence_extra_depth = 2


def _apply_position_overrides(ctx: AgentContext, cfg: EngineConfig) -> None:
    if ctx.position_fen is None:
        return
    if _is_endgame(ctx.position_fen):
        cfg.nmp_enabled = False
        if cfg.evaluator_name == "hce":
            cfg.hce_modules = "classic"
    if _is_tactical(ctx.position_fen):
        cfg.razoring_margins = (
            max(150, cfg.razoring_margins[0] - 100),
            max(200, cfg.razoring_margins[1] - 100),
            max(250, cfg.razoring_margins[2] - 100),
        )
        cfg.quiescence_extra_depth = max(cfg.quiescence_extra_depth, 2)


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
    _pick_mode_defaults(ctx, cfg)
    _apply_position_overrides(ctx, cfg)

    # Fill in evaluator auxiliary fields.
    if cfg.evaluator_name == "hce":
        cfg.hce_modules = cfg.hce_modules or ("all" if ctx.mode == "full" else "classic")
    else:
        cfg.hce_modules = None

    # Enforce pruning gating.
    if not cfg.pruning_enabled:
        cfg.nmp_enabled = False
        cfg.razoring_enabled = False
        cfg.futility_enabled = False
        cfg.lmr_enabled = False

    # Activation audit log (kept lightweight; stderr remains UCI-clean).
    cfg.activated_modules.append(("calix_v2_selector", "local scoring-based selector"))
    if ctx.mode == "minimal":
        cfg.activated_modules.append(("stub_evaluator", "minimal mode"))
        cfg.activated_modules.append(("stub_ordering", "minimal mode"))
        cfg.activated_modules.append(("stub_pruning", "minimal mode"))
    elif ctx.mode == "standard":
        cfg.activated_modules.append(("material_evaluator", "standard mode"))
        cfg.activated_modules.append(("captures_first_ordering", "standard mode"))
    else:
        cfg.activated_modules.append(("hce_evaluator", "full mode"))
        cfg.activated_modules.append(("captures_first_ordering", "full mode"))
    if ctx.position_fen and _is_endgame(ctx.position_fen):
        cfg.activated_modules.append(("endgame_hint", "endgame detected"))
    if ctx.position_fen and _is_tactical(ctx.position_fen):
        cfg.activated_modules.append(("tactical_hint", "tactical position detected"))

    return _filter_to_available(cfg, ctx.available_modules)


def reset_cache() -> None:
    """Backward-compatible no-op (v2 no longer memoises)."""
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
