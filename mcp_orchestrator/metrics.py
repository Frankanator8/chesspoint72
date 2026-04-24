"""Token / cost savings tracker for the MCP orchestrator.

Every MCP tool call is logged as one JSONL record to `metrics/calls.jsonl`.
For each call we estimate the tokens we *did not* spend by routing the work
through this server instead of having an agent drive the engine via raw
chat turns (describe the command, read stdout line-by-line into context,
parse the result, etc.).

The baseline per tool is a conservative estimate of the conversation that
would otherwise have been needed. It's a heuristic, not a benchmark — we
record the assumptions alongside every entry so they can be revised later.

Cost model: Claude Opus 4.7 list pricing (input $15 / MTok, output $75 /
MTok). We assume tokens-saved are 80% input / 20% output.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable

_METRICS_DIR = Path(__file__).resolve().parent / "metrics"
_METRICS_DIR.mkdir(exist_ok=True)
_LOG_PATH = _METRICS_DIR / "calls.jsonl"
_LOCK = threading.Lock()

# $/token. Derived from list pricing: $15/MTok input, $75/MTok output.
_INPUT_COST_PER_TOKEN = 15.0 / 1_000_000
_OUTPUT_COST_PER_TOKEN = 75.0 / 1_000_000
_INPUT_FRACTION = 0.80

# Per-tool baseline of tokens that would have been spent if the agent had
# to drive the engine by hand in chat. Tuned for a single invocation.
# - run_perft: UCI handshake (~200 tok) + position setup (~100) + raw
#   divide output at depth 5+ can exceed 3K tokens just on the move list
#   the agent would have to ingest. 3500 is conservative.
# - play_sprt_match: each game streams dozens of UCI lines × many games.
# - run_tactics: each EPD position round-trips bestmove plus info lines.
_BASELINE_TOKENS: dict[str, int] = {
    "run_perft": 3500,
    "play_sprt_match": 40000,
    "run_tactics": 8000,
}
_DEFAULT_BASELINE = 2000


@dataclass
class CallRecord:
    ts: float
    tool: str
    duration_s: float
    ok: bool
    tokens_saved: int
    cost_saved_usd: float
    args_summary: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def _estimate_tokens_saved(tool: str, result: Any) -> int:
    base = _BASELINE_TOKENS.get(tool, _DEFAULT_BASELINE)
    # Scale perft by depth when we can see it in the echoed result, since
    # the divide output grows roughly 30× per depth level.
    if tool == "run_perft" and isinstance(result, dict):
        depth = result.get("depth")
        if isinstance(depth, int) and depth > 4:
            base = int(base * (1.8 ** (depth - 4)))
    return base


def _cost_saved(tokens: int) -> float:
    input_tok = tokens * _INPUT_FRACTION
    output_tok = tokens * (1 - _INPUT_FRACTION)
    return input_tok * _INPUT_COST_PER_TOKEN + output_tok * _OUTPUT_COST_PER_TOKEN


def _append(record: CallRecord) -> None:
    line = json.dumps(asdict(record), separators=(",", ":"))
    with _LOCK:
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _summarise_args(args: tuple, kwargs: dict) -> dict[str, Any]:
    # Keep the args log small and non-sensitive: record only scalar types
    # and truncate long strings (engine paths can be long).
    out: dict[str, Any] = {}
    for i, v in enumerate(args):
        out[f"arg{i}"] = _scrub(v)
    for k, v in kwargs.items():
        out[k] = _scrub(v)
    return out


def _scrub(v: Any) -> Any:
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    if isinstance(v, str):
        return v if len(v) <= 120 else v[:117] + "..."
    return str(type(v).__name__)


def track(tool_name: str) -> Callable:
    """Decorator: log a call record with estimated savings around a tool."""

    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            ok = True
            result: Any = None
            try:
                result = fn(*args, **kwargs)
                if isinstance(result, dict) and result.get("ok") is False:
                    ok = False
                return result
            except Exception:
                ok = False
                raise
            finally:
                duration = time.monotonic() - start
                tokens = _estimate_tokens_saved(tool_name, result) if ok else 0
                rec = CallRecord(
                    ts=time.time(),
                    tool=tool_name,
                    duration_s=round(duration, 4),
                    ok=ok,
                    tokens_saved=tokens,
                    cost_saved_usd=round(_cost_saved(tokens), 6),
                    args_summary=_summarise_args(args, kwargs),
                )
                try:
                    _append(rec)
                except Exception:
                    # Never let metrics break a tool call.
                    pass

        return wrapper

    return deco


def log_path() -> Path:
    return _LOG_PATH


def load_records() -> list[dict]:
    if not _LOG_PATH.exists():
        return []
    out: list[dict] = []
    with _LOG_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def summary() -> dict[str, Any]:
    recs = load_records()
    total_tokens = sum(r.get("tokens_saved", 0) for r in recs)
    total_cost = sum(r.get("cost_saved_usd", 0.0) for r in recs)
    by_tool: dict[str, dict[str, float]] = {}
    for r in recs:
        t = r.get("tool", "?")
        slot = by_tool.setdefault(t, {"calls": 0, "tokens_saved": 0, "cost_saved_usd": 0.0})
        slot["calls"] += 1
        slot["tokens_saved"] += r.get("tokens_saved", 0)
        slot["cost_saved_usd"] += r.get("cost_saved_usd", 0.0)
    return {
        "calls": len(recs),
        "tokens_saved": total_tokens,
        "cost_saved_usd": round(total_cost, 4),
        "by_tool": by_tool,
        "log_path": str(_LOG_PATH),
    }
