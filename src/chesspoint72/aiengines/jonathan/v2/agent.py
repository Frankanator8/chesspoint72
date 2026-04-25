"""Module Selector Agent for Calix v2 — driven by Claude Opus 4.7.

The agent calls the Anthropic API to choose which engine modules to
activate for the current ``AgentContext``. The model returns a structured
``EngineConfig`` that the rest of the engine consumes exactly as it does
in v1.

Resilience contract:
- If ``ANTHROPIC_API_KEY`` is unset, the ``anthropic`` package is missing,
  the API errors, or the response cannot be parsed, the agent silently
  falls back to v1's deterministic rule cascade so the engine still
  plays.
- ``select_modules`` memoises by ``(mode, can_add_modules,
  available_modules name set)`` so the per-``go`` UCI loop pays one API
  call per game, not one per move.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from chesspoint72.aiengines.jonathan.v2.registry import (
    ModuleDescriptor,
    find_capability,
    scan_modules,
)


# Calix v2 is pinned to Claude Opus 4.7 — the most capable model for
# multi-constraint selection over the module registry.
CALIX_MODEL: str = "claude-opus-4-7"

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
# JSON schema returned by Claude — passed via output_config.format
# --------------------------------------------------------------------------- #


_CONFIG_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "evaluator_name": {
            "type": "string",
            "enum": ["stub", "material", "hce", "nnue"],
        },
        "hce_modules": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": (
                "Comma-separated HCE module names, the preset 'classic'/'advanced'/'all',"
                " or null when evaluator_name != 'hce'."
            ),
        },
        "pruning_enabled": {"type": "boolean"},
        "nmp_enabled": {"type": "boolean"},
        "razoring_enabled": {"type": "boolean"},
        "futility_enabled": {"type": "boolean"},
        "lmr_enabled": {"type": "boolean"},
        "razoring_margins": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 3,
            "maxItems": 3,
        },
        "move_ordering": {
            "type": "string",
            "enum": ["stub", "captures_first"],
        },
        "quiescence_extra_depth": {"type": "integer"},
        "default_depth": {"type": "integer"},
        "default_time": {"type": "number"},
        "activated_modules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["name", "reason"],
                "additionalProperties": False,
            },
        },
        "rationale": {
            "type": "string",
            "description": "One-sentence summary of the choice — recorded in the activation log.",
        },
    },
    "required": [
        "evaluator_name",
        "hce_modules",
        "pruning_enabled",
        "nmp_enabled",
        "razoring_enabled",
        "futility_enabled",
        "lmr_enabled",
        "razoring_margins",
        "move_ordering",
        "quiescence_extra_depth",
        "default_depth",
        "default_time",
        "activated_modules",
        "rationale",
    ],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# Position inspection helpers (used both by the rule fallback and to enrich
# the LLM prompt with cheap structural hints)
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
# Prompt assembly
# --------------------------------------------------------------------------- #


def _format_module_registry(modules: list[ModuleDescriptor]) -> str:
    """Render the registry as a stable, deterministic block of text.

    The text is deterministic so it can sit inside a cached prompt prefix:
    rows are sorted by module name, capability lists by capability name,
    and config-field dicts by key.
    """
    lines: list[str] = []
    for desc in sorted(modules, key=lambda m: m.name):
        caps = ", ".join(sorted(desc.capabilities))
        cfg = ", ".join(f"{k}={desc.config_fields[k]!r}" for k in sorted(desc.config_fields))
        cfg_part = f"   defaults: {cfg}" if cfg else "   defaults: (none)"
        lines.append(f"- {desc.name}\n   capabilities: {caps}\n{cfg_part}")
    return "\n".join(lines)


_SYSTEM_TEMPLATE = """You are the Module Selector Agent for the Calix chess engine.

Calix runs a UCI alpha-beta search whose components are pluggable: an
evaluator, a move-ordering policy, and a forward-pruning policy with
individual toggles for null-move pruning (NMP), razoring, futility, and
LMR. Your job is to read the registry of *available* modules below and
return a fully-resolved EngineConfig that picks the right combination
for the user-supplied AgentContext.

Hard constraints — break any of these and the engine fails:
- evaluator_name MUST be one of: stub, material, hce, nnue.
- hce_modules MUST be null unless evaluator_name == "hce". When it is
  "hce", use the preset "all", "classic", or "advanced", or a
  comma-separated list of HCE feature names from the registry.
- move_ordering MUST be one of: stub, captures_first.
- razoring_margins MUST be a 3-element list of positive integers
  representing the depth-2 / depth-3 / depth-4 margins.
- pruning_enabled gates the four toggles. If pruning_enabled is false,
  set nmp_enabled, razoring_enabled, futility_enabled, lmr_enabled all
  to false.
- activated_modules is the audit trail printed to the operator. Include
  one entry per module/capability you activated, with a short reason.

Mode-driven defaults:
- mode=minimal: stub eval, stub ordering, no pruning. Used when no
  position or clock context is available.
- mode=standard: material eval, captures-first ordering, futility + LMR
  on. Enable NMP and razoring only when the clock is generous.
- mode=full: HCE eval, captures-first ordering, every pruning technique
  on with aggressive margins.

Position-aware adjustments (apply on top of the mode defaults):
- Endgame (few minor/major pieces): disable NMP regardless of mode
  (zugzwang risk). If using HCE, drop to the "classic" preset.
- Tactical position (queens still on the board, board still crowded):
  lower razoring margins and bump quiescence_extra_depth so the engine
  resolves more captures.
- can_add_modules is informational only — you must not invent module
  names that are not present in the registry below.

AVAILABLE MODULES
=================
{registry}
"""


def _build_system_prompt(modules: list[ModuleDescriptor]) -> str:
    return _SYSTEM_TEMPLATE.format(registry=_format_module_registry(modules))


def _build_user_prompt(ctx: AgentContext) -> str:
    fen = ctx.position_fen or "<unknown>"
    clock = "<unknown>" if ctx.time_remaining_ms is None else f"{ctx.time_remaining_ms} ms"
    hints: list[str] = []
    if ctx.position_fen:
        if _is_endgame(ctx.position_fen):
            hints.append("position-hint: endgame (few non-king/pawn pieces)")
        if _is_tactical(ctx.position_fen):
            hints.append("position-hint: tactical (queens on, board crowded)")
    hint_block = "\n".join(hints) if hints else "position-hint: none"
    return (
        "AgentContext for the upcoming search:\n"
        f"  mode             = {ctx.mode}\n"
        f"  position_fen     = {fen}\n"
        f"  time_remaining   = {clock}\n"
        f"  can_add_modules  = {ctx.can_add_modules}\n"
        f"{hint_block}\n\n"
        "Return a single JSON object that matches the schema. No prose."
    )


# --------------------------------------------------------------------------- #
# Anthropic API call + response parsing
# --------------------------------------------------------------------------- #


def _config_from_dict(payload: dict[str, Any]) -> EngineConfig:
    margins = payload["razoring_margins"]
    activated = [
        (entry["name"], entry["reason"])
        for entry in payload.get("activated_modules", [])
    ]
    rationale = payload.get("rationale")
    if rationale:
        activated.append((f"calix_{CALIX_MODEL}", f"rationale: {rationale}"))
    pruning_on = bool(payload["pruning_enabled"])
    return EngineConfig(
        evaluator_name=payload["evaluator_name"],
        hce_modules=payload.get("hce_modules"),
        pruning_enabled=pruning_on,
        nmp_enabled=pruning_on and bool(payload["nmp_enabled"]),
        razoring_enabled=pruning_on and bool(payload["razoring_enabled"]),
        futility_enabled=pruning_on and bool(payload["futility_enabled"]),
        lmr_enabled=pruning_on and bool(payload["lmr_enabled"]),
        razoring_margins=(int(margins[0]), int(margins[1]), int(margins[2])),
        move_ordering=payload["move_ordering"],
        quiescence_extra_depth=max(0, int(payload["quiescence_extra_depth"])),
        default_depth=max(1, int(payload["default_depth"])),
        default_time=max(0.05, float(payload["default_time"])),
        activated_modules=activated,
    )


def _call_claude(ctx: AgentContext) -> EngineConfig | None:
    """Invoke the Anthropic API. Returns None on any failure path."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic  # noqa: PLC0415 — optional, lazy import
    except ImportError:
        return None

    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=CALIX_MODEL,
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": _build_system_prompt(ctx.available_modules),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": _build_user_prompt(ctx)}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": _CONFIG_JSON_SCHEMA,
                },
                "effort": "low",
            },
        )
    except Exception:
        return None

    text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "")
    if not text:
        return None
    try:
        payload = json.loads(text)
        return _config_from_dict(payload)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Rule-based fallback (port of v1 — used whenever the API call fails)
# --------------------------------------------------------------------------- #


def _apply_minimal(_ctx: AgentContext, cfg: EngineConfig) -> EngineConfig:
    cfg.activated_modules.append(("negamax", "required for legal play"))
    cfg.activated_modules.append(("transposition_table", "required for repetition handling"))
    cfg.activated_modules.append(("stub_evaluator", "minimal mode: no eval"))
    cfg.activated_modules.append(("stub_ordering", "minimal mode: no ordering"))
    cfg.activated_modules.append(("stub_pruning", "minimal mode: no pruning"))
    return cfg


def _apply_standard(ctx: AgentContext, cfg: EngineConfig) -> EngineConfig:
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
        cfg.activated_modules.append(("null_move_pruning", "standard mode: clock generous (>= 30s)"))
        cfg.activated_modules.append(("razoring", "standard mode: clock generous (>= 30s)"))
    return cfg


def _apply_full(_ctx: AgentContext, cfg: EngineConfig) -> EngineConfig:
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
    if ctx.position_fen is None:
        return cfg
    if _is_endgame(ctx.position_fen):
        cfg.nmp_enabled = False
        cfg.activated_modules.append(
            ("null_move_pruning_disabled", "endgame detected: zugzwang risk")
        )
        if cfg.evaluator_name == "hce":
            cfg.hce_modules = "classic"
            cfg.activated_modules.append(
                ("hce_classic_modules", "endgame: drop advanced features")
            )
    if _is_tactical(ctx.position_fen):
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


def _select_modules_rule_based(ctx: AgentContext) -> EngineConfig:
    cfg = EngineConfig(evaluator_name="stub", default_depth=4, default_time=5.0)
    if ctx.mode == "minimal":
        cfg = _apply_minimal(ctx, cfg)
    elif ctx.mode == "standard":
        cfg = _apply_standard(ctx, cfg)
    elif ctx.mode == "full":
        cfg = _apply_full(ctx, cfg)
    else:
        raise ValueError(f"unknown agent mode: {ctx.mode!r}")
    cfg = _apply_position_overrides(ctx, cfg)
    cfg.activated_modules.append(
        ("calix_rule_based_fallback", "Anthropic API unavailable — using v1 rule cascade")
    )
    return _filter_to_available(cfg, ctx.available_modules)


# --------------------------------------------------------------------------- #
# Public API — memoised for one API call per game
# --------------------------------------------------------------------------- #


_LAST_KEY: tuple | None = None
_LAST_CFG: EngineConfig | None = None


def _cache_key(ctx: AgentContext) -> tuple:
    names = tuple(sorted(d.name for d in ctx.available_modules))
    return (ctx.mode, ctx.can_add_modules, names)


def select_modules(ctx: AgentContext) -> EngineConfig:
    """Resolve an EngineConfig — calls Claude once, then memoises.

    Memoisation is keyed on the stable parts of the context (mode +
    permission + registry shape) so the per-``go`` UCI loop doesn't pay
    a full API round trip on every move.
    """
    global _LAST_KEY, _LAST_CFG
    key = _cache_key(ctx)
    if key == _LAST_KEY and _LAST_CFG is not None:
        return _LAST_CFG
    cfg = _call_claude(ctx)
    if cfg is None:
        cfg = _select_modules_rule_based(ctx)
    _LAST_KEY = key
    _LAST_CFG = cfg
    return cfg


def reset_cache() -> None:
    """Drop the memoised config — useful in tests."""
    global _LAST_KEY, _LAST_CFG
    _LAST_KEY = None
    _LAST_CFG = None


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
