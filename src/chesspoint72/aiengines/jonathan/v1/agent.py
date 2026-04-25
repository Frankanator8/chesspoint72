"""Module Selector Agent — the deciding brain inside Calix.

The agent is rule-based, deterministic, and self-contained: it runs in
process and never calls out to a model. The behaviour required by Phase 3
of the engine spec is captured here as a set of ``select_modules`` decision
branches keyed off ``AgentContext.mode`` and the optional position / clock
hints.

Public surface:
    AgentContext   — input bundle (mode, FEN, clock, permission, registry).
    EngineConfig   — fully resolved configuration handed to the search.
    select_modules — pure function: AgentContext -> EngineConfig.
    build_context  — convenience constructor for the three named modes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from chesspoint72.aiengines.jonathan.v1.registry import (
    ModuleDescriptor,
    find_capability,
    scan_modules,
)


Mode = Literal["minimal", "standard", "full"]


# --------------------------------------------------------------------------- #
# Data structures
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

    # Evaluator selection.
    evaluator_name: str  # "stub" | "material" | "hce" | "nnue"
    hce_modules: str | None = None

    # Pruning toggles — interpreted by the Calix factory wiring.
    pruning_enabled: bool = False
    nmp_enabled: bool = False
    razoring_enabled: bool = False
    futility_enabled: bool = False
    lmr_enabled: bool = False
    razoring_margins: tuple[int, int, int] = (350, 450, 550)

    # Move ordering: "stub" (no-op) or "captures_first" (TT-first + captures).
    move_ordering: str = "stub"

    # Quiescence depth budget. Higher = deeper tactical resolution.
    quiescence_extra_depth: int = 0

    # Search budget defaults — overridden by UCI ``go`` command line.
    default_depth: int = 4
    default_time: float = 5.0

    # Audit log: (module_name, reason) pairs the entrypoint prints to stderr.
    activated_modules: list[tuple[str, str]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Position inspection helpers
# --------------------------------------------------------------------------- #


def _is_endgame(fen: str) -> bool:
    """True when the position has few non-king/pawn pieces.

    Threshold mirrors Stockfish's classic phase boundary: at most 6 minor
    or major pieces total across both sides.
    """
    placement = fen.split(" ", 1)[0]
    minors_majors = sum(1 for ch in placement if ch in "QqRrBbNn")
    return minors_majors <= 6


def _is_tactical(fen: str) -> bool:
    """Cheap heuristic: many opposing pieces in close contact = tactical.

    We treat any position with fewer than 28 empty squares (== 36 occupied)
    AND at least one queen still on the board as tactical. Crude but cheap
    — the agent does not need a real eval here.
    """
    placement = fen.split(" ", 1)[0]
    occupied = sum(1 for ch in placement if ch.isalpha())
    has_queen = "Q" in placement or "q" in placement
    return occupied >= 24 and has_queen


# --------------------------------------------------------------------------- #
# Decision rules
# --------------------------------------------------------------------------- #


def _apply_minimal(_ctx: AgentContext, cfg: EngineConfig) -> EngineConfig:
    """Raw α-β only — no pruning, no ordering, stub eval.

    The minimal mode exists so the engine can play strictly-legal chess
    even when nothing about the position or the clock is known.
    """
    cfg.activated_modules.append(("negamax", "required for legal play"))
    cfg.activated_modules.append(("transposition_table", "required for repetition handling"))
    cfg.activated_modules.append(("stub_evaluator", "minimal mode: no eval"))
    cfg.activated_modules.append(("stub_ordering", "minimal mode: no ordering"))
    cfg.activated_modules.append(("stub_pruning", "minimal mode: no pruning"))
    return cfg


def _apply_standard(ctx: AgentContext, cfg: EngineConfig) -> EngineConfig:
    """Conservative defaults: ordering + material eval + LMR/futility.

    NMP and razoring stay off unless the clock is generous (>= 30s remain),
    matching the Phase 3 spec.
    """
    cfg.evaluator_name = "material"
    cfg.move_ordering = "captures_first"
    cfg.pruning_enabled = True
    cfg.futility_enabled = True
    cfg.lmr_enabled = True
    cfg.activated_modules.append(("material_evaluator", "standard mode: cheap baseline eval"))
    cfg.activated_modules.append(("captures_first_ordering", "standard mode: TT/captures-first"))
    cfg.activated_modules.append(("futility_pruning", "standard mode: frontier-depth filter"))
    cfg.activated_modules.append(("lmr", "standard mode: late-move reductions"))

    if ctx.time_remaining_ms is not None and ctx.time_remaining_ms >= 30_000:
        cfg.nmp_enabled = True
        cfg.razoring_enabled = True
        cfg.activated_modules.append(
            ("null_move_pruning", "standard mode: clock generous (>= 30s)")
        )
        cfg.activated_modules.append(
            ("razoring", "standard mode: clock generous (>= 30s)")
        )
    return cfg


def _apply_full(_ctx: AgentContext, cfg: EngineConfig) -> EngineConfig:
    """Everything on with aggressive margins."""
    cfg.evaluator_name = "hce"
    cfg.hce_modules = "all"
    cfg.move_ordering = "captures_first"
    cfg.pruning_enabled = True
    cfg.nmp_enabled = True
    cfg.razoring_enabled = True
    cfg.futility_enabled = True
    cfg.lmr_enabled = True
    cfg.razoring_margins = (250, 350, 450)  # tighter -> prunes more
    cfg.quiescence_extra_depth = 2
    cfg.activated_modules.extend(
        [
            ("hce_evaluator_all_modules", "full mode: maximum signal"),
            ("captures_first_ordering", "full mode"),
            ("null_move_pruning", "full mode"),
            ("razoring", "full mode: aggressive margins"),
            ("futility_pruning", "full mode"),
            ("lmr", "full mode"),
        ]
    )
    return cfg


def _apply_position_overrides(ctx: AgentContext, cfg: EngineConfig) -> EngineConfig:
    """Adjust *cfg* based on ``ctx.position_fen`` characteristics."""
    if ctx.position_fen is None:
        return cfg

    if _is_endgame(ctx.position_fen):
        cfg.nmp_enabled = False
        cfg.activated_modules.append(
            ("null_move_pruning_disabled", "endgame detected: zugzwang risk")
        )
        # Bump the eval modules toward king-and-pawn aware features when
        # we know we're in an endgame and an HCE eval is in play.
        if cfg.evaluator_name == "hce":
            cfg.hce_modules = "classic"
            cfg.activated_modules.append(
                ("hce_classic_modules", "endgame: drop advanced features")
            )

    if _is_tactical(ctx.position_fen):
        # Lower razoring margins -> more pruning of clearly-lost lines so
        # the saved budget can fund deeper tactical resolution.
        cfg.razoring_margins = (
            max(150, cfg.razoring_margins[0] - 100),
            max(200, cfg.razoring_margins[1] - 100),
            max(250, cfg.razoring_margins[2] - 100),
        )
        cfg.quiescence_extra_depth = max(cfg.quiescence_extra_depth, 2)
        cfg.activated_modules.append(
            ("razoring_margins_lowered", "tactical position: prune lost lines harder")
        )
    return cfg


def _filter_to_available(
    cfg: EngineConfig, available: list[ModuleDescriptor]
) -> EngineConfig:
    """When the agent may not add modules, gate evaluator/pruning by registry.

    The agent's *config keys* still describe the desired shape; this helper
    only down-grades capabilities that the registry cannot provide. It is a
    no-op for an empty registry — minimal mode is always reachable because
    the stubs live inside the Calix package itself.
    """
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


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def select_modules(ctx: AgentContext) -> EngineConfig:
    """Resolve an EngineConfig from an AgentContext.

    The function is deterministic and side-effect-free. The agent never
    creates, scaffolds, or registers new modules — even when
    ``ctx.can_add_modules`` is True, the autonomous mode only widens the
    set of *existing* registry modules the agent is willing to activate.
    """
    cfg = EngineConfig(
        evaluator_name="stub",
        default_depth=4,
        default_time=5.0,
    )

    if ctx.mode == "minimal":
        cfg = _apply_minimal(ctx, cfg)
    elif ctx.mode == "standard":
        cfg = _apply_standard(ctx, cfg)
    elif ctx.mode == "full":
        cfg = _apply_full(ctx, cfg)
    else:
        raise ValueError(f"unknown agent mode: {ctx.mode!r}")

    cfg = _apply_position_overrides(ctx, cfg)

    # The agent never creates new modules. The ``can_add_modules`` flag is
    # carried through for API compatibility, but its True value only widens
    # the set of *existing* modules the agent will activate — it never
    # triggers code generation. All modes therefore filter to the registry.
    cfg = _filter_to_available(cfg, ctx.available_modules)

    return cfg


def build_context(
    mode_name: str,
    *,
    position_fen: str | None = None,
    time_remaining_ms: int | None = None,
    available_modules: list[ModuleDescriptor] | None = None,
) -> AgentContext:
    """Construct an ``AgentContext`` from one of the three named modes.

    *mode_name* is a CLI-facing label: ``blind``, ``aware``, or
    ``autonomous``. It maps onto the internal ``mode`` field plus the
    permission flag. The ``available_modules`` list is auto-scanned when
    not provided.
    """
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
    """Return a copy of *ctx* with fresh runtime hints filled in.

    Blind mode is left untouched — the operator chose to deny the agent
    runtime visibility, so the UCI loop must not leak it back in. For aware
    and autonomous modes the hints overwrite whatever was seen previously,
    so each ``go`` command sees the latest position and clock.
    """
    if ctx.mode == "minimal":
        return ctx
    return replace(
        ctx,
        position_fen=position_fen,
        time_remaining_ms=time_remaining_ms,
    )
