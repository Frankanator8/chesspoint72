"""Render token/cost savings graphs from the MCP call log.

Usage:
    python -m mcp_orchestrator.dashboard               # writes PNGs to metrics/
    python -m mcp_orchestrator.dashboard --show        # also pop up windows
    python -m mcp_orchestrator.dashboard --summary     # print totals only

The call log is the JSONL file written by `mcp_orchestrator.metrics`. Each
record carries the per-call estimate of tokens saved (vs. driving the
engine through chat turns by hand) and the derived USD cost at Opus 4.7
list pricing. This script aggregates those estimates into four charts:

  1. Cumulative tokens saved over time
  2. Cumulative USD saved over time
  3. Tokens saved per tool (bar)
  4. Call count per tool (bar)

Graphs go to `mcp_orchestrator/metrics/*.png` so they can be attached to
demo slides or the hackathon submission.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from mcp_orchestrator import metrics

_OUT_DIR = Path(__file__).resolve().parent / "metrics"


def _print_summary() -> None:
    s = metrics.summary()
    print(f"Log file:      {s['log_path']}")
    print(f"Total calls:   {s['calls']}")
    print(f"Tokens saved:  {s['tokens_saved']:,}")
    print(f"USD saved:     ${s['cost_saved_usd']:.4f}")
    if s["by_tool"]:
        print("\nPer-tool breakdown:")
        header = f"  {'tool':<22} {'calls':>6} {'tokens':>12} {'usd':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for tool, row in sorted(
            s["by_tool"].items(), key=lambda kv: kv[1]["tokens_saved"], reverse=True
        ):
            print(
                f"  {tool:<22} {int(row['calls']):>6} "
                f"{int(row['tokens_saved']):>12,} "
                f"${row['cost_saved_usd']:>9.4f}"
            )


def _render(show: bool) -> None:
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is required for graphs. Install with:\n"
            "    pip install matplotlib",
            file=sys.stderr,
        )
        sys.exit(2)

    records = metrics.load_records()
    if not records:
        print(f"No records in {metrics.log_path()} — run some tools first.")
        return

    records.sort(key=lambda r: r["ts"])
    times = [dt.datetime.fromtimestamp(r["ts"]) for r in records]
    cum_tokens, cum_cost, tok, cost = [], [], 0, 0.0
    for r in records:
        tok += r.get("tokens_saved", 0)
        cost += r.get("cost_saved_usd", 0.0)
        cum_tokens.append(tok)
        cum_cost.append(cost)

    _OUT_DIR.mkdir(exist_ok=True)

    # 1. Cumulative tokens
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, cum_tokens, marker="o", linewidth=2)
    ax.set_title("Cumulative tokens saved via MCP orchestrator")
    ax.set_ylabel("tokens saved")
    ax.set_xlabel("time")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(_OUT_DIR / "cumulative_tokens.png", dpi=120)

    # 2. Cumulative cost
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, cum_cost, marker="o", color="#2a9d8f", linewidth=2)
    ax.set_title("Cumulative USD saved (Opus 4.7 list pricing)")
    ax.set_ylabel("USD")
    ax.set_xlabel("time")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(_OUT_DIR / "cumulative_cost.png", dpi=120)

    # 3. / 4. Per-tool bars
    by_tool: dict[str, dict] = {}
    for r in records:
        slot = by_tool.setdefault(r["tool"], {"calls": 0, "tokens_saved": 0})
        slot["calls"] += 1
        slot["tokens_saved"] += r.get("tokens_saved", 0)
    tools = sorted(by_tool, key=lambda t: by_tool[t]["tokens_saved"], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(tools, [by_tool[t]["tokens_saved"] for t in tools], color="#e76f51")
    ax.set_title("Tokens saved by tool")
    ax.set_ylabel("tokens saved")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(_OUT_DIR / "tokens_by_tool.png", dpi=120)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(tools, [by_tool[t]["calls"] for t in tools], color="#264653")
    ax.set_title("Invocations by tool")
    ax.set_ylabel("calls")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(_OUT_DIR / "calls_by_tool.png", dpi=120)

    print(f"Wrote 4 PNGs to {_OUT_DIR}")
    if show:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="pop up interactive windows")
    ap.add_argument("--summary", action="store_true", help="print totals only, skip graphs")
    args = ap.parse_args()

    _print_summary()
    if args.summary:
        return
    _render(show=args.show)


if __name__ == "__main__":
    main()
