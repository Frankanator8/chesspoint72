"""Stage 0 — Perft validation.

Zero-tolerance gate: any mismatch halts the pipeline. Tests PyChessBoard's
legal-move generation against known node counts for four standard positions.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import chess

from chesspoint72.engine.boards.pychess import PyChessBoard
from chesspoint72.engine.core.types import Move

# --------------------------------------------------------------------------- #
# Test suite (from EVAL_PIPELINE.md)
# --------------------------------------------------------------------------- #

PERFT_SUITE: dict[str, dict] = {
    "startpos": {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "results": {1: 20, 2: 400, 3: 8902, 4: 197281, 5: 4865609},
    },
    "kiwipete": {
        "fen": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
        "results": {1: 48, 2: 2039, 3: 97862, 4: 4085603},
    },
    "pos3": {
        "fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -",
        "results": {1: 14, 2: 191, 3: 2812, 4: 43238, 5: 674624},
    },
    "pos4": {
        "fen": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "results": {1: 6, 2: 264, 3: 9467, 4: 422333},
    },
}


# --------------------------------------------------------------------------- #
# Perft implementation (uses python-chess directly — no UCI subprocess)
# --------------------------------------------------------------------------- #

def _perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    nodes = 0
    for move in board.legal_moves:
        board.push(move)
        nodes += _perft(board, depth - 1)
        board.pop()
    return nodes


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass
class PerftCaseResult:
    name: str
    depth: int
    expected: int
    actual: int
    passed: bool
    elapsed_ms: float


@dataclass
class Stage0Result:
    cases: list[PerftCaseResult]
    all_passed: bool
    elapsed_s: float

    def print_report(self) -> None:
        print("\n=== Stage 0 — Perft Validation ===")
        for c in self.cases:
            status = "PASS" if c.passed else "FAIL"
            print(
                f"  {c.name:12s} d{c.depth}  {status}"
                f"  got={c.actual:>10,}  exp={c.expected:>10,}"
                f"  ({c.elapsed_ms:.0f}ms)"
            )
        verdict = "ALL PASS — board is correct" if self.all_passed else "FAILURES DETECTED — halt pipeline"
        print(f"\n  Result: {verdict} ({self.elapsed_s:.1f}s total)\n")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def run_stage0(max_depth: int = 5, verbose: bool = True) -> Stage0Result:
    """Run perft on all positions up to *max_depth*.

    Returns Stage0Result. If ``all_passed`` is False the pipeline must halt —
    move generation is broken and every downstream Elo figure would be invalid.
    """
    cases: list[PerftCaseResult] = []
    t_start = time.monotonic()

    for name, data in PERFT_SUITE.items():
        fen = data["fen"]
        # Normalise FEN: python-chess needs 6 fields; EPD-style has 4
        parts = fen.split()
        if len(parts) == 4:
            fen = fen + " 0 1"

        for depth, expected in data["results"].items():
            if depth > max_depth:
                continue
            board = chess.Board(fen)
            t0 = time.monotonic()
            actual = _perft(board, depth)
            elapsed_ms = (time.monotonic() - t0) * 1000
            passed = (actual == expected)
            cases.append(PerftCaseResult(name, depth, expected, actual, passed, elapsed_ms))

    all_passed = all(c.passed for c in cases)
    result = Stage0Result(cases=cases, all_passed=all_passed, elapsed_s=time.monotonic() - t_start)

    if verbose:
        result.print_report()

    return result
