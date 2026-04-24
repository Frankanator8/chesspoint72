"""EPD tactical-suite runner.

Loads an EPD file, plays each position against a UCI engine with a fixed
`go movetime`, and checks the engine's `bestmove` against the `bm`
operation. EPD `bm` values are in SAN and may list several acceptable
moves; we convert both sides to UCI via python-chess before comparing.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import chess

from mcp_orchestrator.validators.uci_parser import UCIEngine, UCIError

log = logging.getLogger(__name__)


_BESTMOVE_RE = re.compile(r"^bestmove\s+(\S+)")


@dataclass
class EPDPosition:
    fen: str
    best_moves_san: list[str]
    raw: str
    id: str | None = None


@dataclass
class TacticsResult:
    total: int = 0
    passed: int = 0
    failed_fens: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        accuracy = (self.passed / self.total * 100.0) if self.total else 0.0
        return {
            "total": self.total,
            "passed": self.passed,
            "accuracy_pct": round(accuracy, 2),
            "failed": self.failed_fens,
        }


def parse_epd_file(path: str | Path) -> Iterator[EPDPosition]:
    """Yield EPDPosition objects from an EPD file.

    EPD lines have the form:
        <pieces> <stm> <castling> <ep> <op1> <args1>; <op2> <args2>; ...
    The first four fields are the FEN board (no clocks); remaining
    semicolon-separated entries are operations. We only care about `bm`
    and (optionally) `id`.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Split board from operations. The board is the first 4 tokens.
        tokens = line.split(None, 4)
        if len(tokens) < 4:
            log.warning("epd:%d malformed line, skipping: %r", lineno, raw_line)
            continue
        board_fen = " ".join(tokens[:4])
        # EPD omits halfmove/fullmove — python-chess and most engines want them.
        full_fen = f"{board_fen} 0 1"

        ops_blob = tokens[4] if len(tokens) == 5 else ""
        ops = {}
        for chunk in ops_blob.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            head, _, tail = chunk.partition(" ")
            ops[head] = tail.strip().strip('"')

        bm = ops.get("bm")
        if not bm:
            log.warning("epd:%d no 'bm' operation, skipping: %r", lineno, raw_line)
            continue
        best_moves_san = bm.split()
        yield EPDPosition(
            fen=full_fen,
            best_moves_san=best_moves_san,
            raw=raw_line,
            id=ops.get("id"),
        )


def _san_list_to_uci(fen: str, san_moves: list[str]) -> list[str]:
    board = chess.Board(fen)
    out: list[str] = []
    for san in san_moves:
        try:
            move = board.parse_san(san)
            out.append(move.uci())
        except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
            log.warning("could not parse SAN %r in position %s", san, fen)
    return out


def _ask_bestmove(engine: UCIEngine, fen: str, time_limit_ms: int) -> str | None:
    engine.send("ucinewgame")
    engine.send("isready")
    engine.read_lines(lambda ln: ln.strip() == "readyok", timeout=5.0)
    engine.set_position(fen)
    engine.send(f"go movetime {int(time_limit_ms)}")
    # Generous wall-clock headroom: movetime + 5s for search wind-down.
    timeout = (time_limit_ms / 1000.0) + 5.0
    lines = engine.read_lines(
        lambda ln: ln.startswith("bestmove"),
        timeout=timeout,
    )
    last = lines[-1]
    m = _BESTMOVE_RE.match(last)
    if not m:
        return None
    bestmove = m.group(1)
    if bestmove in ("(none)", "0000"):
        return None
    return bestmove


def run_tactics(
    engine_path: str,
    epd_file_path: str,
    time_limit_ms: int,
    handshake_timeout: float = 5.0,
) -> TacticsResult:
    """Run an EPD suite against a UCI engine.

    A single engine process is reused across positions (with `ucinewgame`
    between each) because spawning per-puzzle dominates runtime for short
    movetimes.
    """
    if not Path(epd_file_path).is_file():
        raise FileNotFoundError(f"EPD file not found: {epd_file_path}")
    if time_limit_ms <= 0:
        raise ValueError("time_limit_ms must be > 0")

    result = TacticsResult()

    with UCIEngine.spawn(engine_path) as engine:
        engine.handshake(timeout=handshake_timeout)

        for pos in parse_epd_file(epd_file_path):
            result.total += 1
            expected_uci = _san_list_to_uci(pos.fen, pos.best_moves_san)
            try:
                bestmove = _ask_bestmove(engine, pos.fen, time_limit_ms)
            except UCIError as e:
                log.warning("engine error on %s: %s", pos.id or pos.fen, e)
                result.failed_fens.append({
                    "fen": pos.fen,
                    "id": pos.id,
                    "expected_san": pos.best_moves_san,
                    "engine_move": None,
                    "error": str(e),
                })
                continue

            if bestmove is not None and bestmove in expected_uci:
                result.passed += 1
            else:
                result.failed_fens.append({
                    "fen": pos.fen,
                    "id": pos.id,
                    "expected_san": pos.best_moves_san,
                    "expected_uci": expected_uci,
                    "engine_move": bestmove,
                })

    return result
