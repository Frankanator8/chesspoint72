"""MCP orchestrator for the Cubist chess hackathon.

Exposes chess-engine testing tools over the Model Context Protocol so that
team agents can compile, validate, and pit engines against each other
without a human in the loop.

Tools exposed:
  - run_perft(engine_executable_path, fen_string, depth)
  - play_sprt_match(engine_a_path, engine_b_path, time_control)   [stub]
  - run_tactics(engine_path, epd_file_path, time_limit)           [stub]

Implemented against the official `mcp` Python SDK's FastMCP helper.
Run with:  python -m mcp_orchestrator.mcp_server
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Allow running as `python mcp_server.py` from inside mcp_orchestrator/
_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT.parent))

from mcp.server.fastmcp import FastMCP  # type: ignore

from mcp_orchestrator.metrics import summary as _metrics_summary, track
from mcp_orchestrator.tournaments.epd_suite import run_tactics as _run_tactics
from mcp_orchestrator.tournaments.sprt_tester import (
    result_to_dict as _sprt_to_dict,
    run_sprt as _run_sprt,
)
from mcp_orchestrator.validators.perft_runner import run_perft as _run_perft
from mcp_orchestrator.validators.uci_parser import UCIError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("mcp_orchestrator")

mcp = FastMCP("chesspoint72-orchestrator")


@mcp.tool()
@track("run_perft")
def run_perft(
    engine_executable_path: str,
    fen_string: str,
    depth: int,
    timeout_seconds: float = 120.0,
) -> dict:
    """Run a perft node-count validation against a UCI engine.

    Args:
        engine_executable_path: Absolute path to the engine binary.
        fen_string: Starting position in FEN, or the literal "startpos".
        depth: Perft depth (plies).
        timeout_seconds: Hard cap on how long to wait for the result.

    Returns:
        {"total": int, "per_move": {uci_move: nodes, ...}, "depth": int}
    """
    log.info("run_perft depth=%d fen=%r engine=%s", depth, fen_string, engine_executable_path)
    try:
        result = _run_perft(
            engine_path=engine_executable_path,
            fen=fen_string,
            depth=int(depth),
            timeout=float(timeout_seconds),
        )
    except (UCIError, ValueError, FileNotFoundError) as e:
        log.warning("run_perft failed: %s", e)
        return {"ok": False, "error": str(e)}

    return {
        "ok": True,
        "depth": int(depth),
        "total": result.total,
        "per_move": result.per_move,
    }


@mcp.tool()
@track("play_sprt_match")
def play_sprt_match(
    engine_a_path: str,
    engine_b_path: str,
    time_control: str,
    elo0: float = 0.0,
    elo1: float = 10.0,
    alpha: float = 0.05,
    beta: float = 0.05,
    max_games: int = 1000,
    openings: list | None = None,
) -> dict:
    """Run a Sequential Probability Ratio Test between two UCI engines.

    Args:
        engine_a_path: Candidate engine (tested against elo1).
        engine_b_path: Reference engine.
        time_control: "base[+inc]" in seconds, e.g. "10+0.1".
        elo0 / elo1: Null and alternative Elo hypotheses (default 0 / 10).
        alpha / beta: Type-I / Type-II error rates (default 0.05 each).
        max_games: Hard cap if SPRT never converges.
        openings: Optional list of starting FENs. If omitted, a small built-in
            balanced-opening book is used to vary games.

    Returns:
        {ok, decision ("H1_accepted"|"H0_accepted"|"inconclusive"), reason,
         W, D, L, games, llr, elo_estimate, bounds, time_control, elapsed_s}
    """
    log.info(
        "play_sprt_match A=%s B=%s tc=%s [%s,%s]",
        engine_a_path, engine_b_path, time_control, elo0, elo1,
    )
    try:
        result = _run_sprt(
            engine_a_path=engine_a_path,
            engine_b_path=engine_b_path,
            time_control=time_control,
            elo0=float(elo0),
            elo1=float(elo1),
            alpha=float(alpha),
            beta=float(beta),
            max_games=int(max_games),
            openings=openings,
        )
    except (ValueError, FileNotFoundError) as e:
        log.warning("play_sprt_match failed: %s", e)
        return {"ok": False, "error": str(e)}
    except Exception as e:
        log.exception("play_sprt_match crashed")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    return {"ok": True, **_sprt_to_dict(result)}


@mcp.tool()
@track("run_tactics")
def run_tactics(
    engine_path: str,
    epd_file_path: str,
    time_limit_ms: int = 1000,
) -> dict:
    """Run an EPD tactical suite against a UCI engine.

    Args:
        engine_path: Absolute path to the engine binary.
        epd_file_path: Path to an EPD file. Each line must carry a `bm`
            operation (SAN), optionally an `id`.
        time_limit_ms: Per-puzzle `go movetime` in milliseconds.

    Returns:
        {ok, total, passed, accuracy_pct, failed: [{fen, id, expected_san,
         expected_uci, engine_move}, ...]}
    """
    log.info(
        "run_tactics engine=%s epd=%s movetime=%dms",
        engine_path, epd_file_path, int(time_limit_ms),
    )
    try:
        result = _run_tactics(
            engine_path=engine_path,
            epd_file_path=epd_file_path,
            time_limit_ms=int(time_limit_ms),
        )
    except (UCIError, ValueError, FileNotFoundError) as e:
        log.warning("run_tactics failed: %s", e)
        return {"ok": False, "error": str(e)}
    except Exception as e:
        log.exception("run_tactics crashed")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    return {"ok": True, **result.as_dict()}


@mcp.tool()
def metrics_summary() -> dict:
    """Return cumulative token/cost savings across all tool invocations."""
    return _metrics_summary()


def main() -> None:
    log.info("starting chesspoint72 MCP orchestrator over stdio")
    mcp.run()


if __name__ == "__main__":
    main()
