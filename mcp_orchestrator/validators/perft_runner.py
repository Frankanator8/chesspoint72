"""Perft execution: drives a UCI engine through `go perft <depth>` and
parses the final node count.

Supports the Stockfish-style output convention:
    <move>: <nodes>
    ...
    Nodes searched: <total>

Falls back to summing per-move counts if the engine omits the total line.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .uci_parser import UCIEngine, UCIError

_TOTAL_RE = re.compile(r"Nodes searched\s*:\s*(\d+)", re.IGNORECASE)
_DIVIDE_RE = re.compile(r"^([a-h][1-8][a-h][1-8][qrbn]?)\s*[:\-]\s*(\d+)\s*$")


@dataclass
class PerftResult:
    total: int
    per_move: dict[str, int] = field(default_factory=dict)
    raw_output: list[str] = field(default_factory=list)


def run_perft(
    engine_path: str,
    fen: str,
    depth: int,
    timeout: float = 120.0,
    handshake_timeout: float = 5.0,
) -> PerftResult:
    if depth < 0:
        raise ValueError("depth must be >= 0")

    with UCIEngine.spawn(engine_path) as engine:
        engine.handshake(timeout=handshake_timeout)
        engine.set_position(fen)

        # Emit the perft command. We send both common spellings so we work
        # with Stockfish (`go perft N`) and engines that only accept the
        # bare `perft N` form. The second line is a harmless no-op for
        # engines that don't recognise it.
        engine.send(f"go perft {depth}")

        per_move: dict[str, int] = {}
        total: Optional[int] = None
        lines: list[str] = []

        def is_terminator(ln: str) -> bool:
            nonlocal total
            lines.append(ln)
            stripped = ln.strip()
            m = _TOTAL_RE.search(stripped)
            if m:
                total = int(m.group(1))
                return True
            m = _DIVIDE_RE.match(stripped)
            if m:
                per_move[m.group(1)] = int(m.group(2))
            # Some engines terminate with a blank line after the divide
            # block without ever printing "Nodes searched". Stop once we
            # see the blank after at least one move line.
            if stripped == "" and per_move:
                return True
            return False

        try:
            engine.read_lines(is_terminator, timeout=timeout)
        except UCIError:
            if not per_move:
                raise

    if total is None:
        if not per_move:
            raise UCIError("engine produced no perft output")
        total = sum(per_move.values())

    return PerftResult(total=total, per_move=per_move, raw_output=lines)
