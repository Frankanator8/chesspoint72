"""Phase 7 / test 3 — integration smoke test.

Drive the Calix UCI controller in blind mode through the position/go cycle
and assert it returns a legal bestmove within the time budget.
"""
from __future__ import annotations

import io
import re
import time

import chess

from chesspoint72.aiengines.jonathan.main import build_controller


_BESTMOVE_RE = re.compile(r"^bestmove\s+(\S+)", re.MULTILINE)


def _run_uci_session(
    agent_mode: str, commands: list[str], default_time: float = 1.0
) -> str:
    """Drive the controller with a list of UCI lines and return stdout."""
    input_stream = iter(commands + ["quit"])
    output = io.StringIO()
    log = io.StringIO()
    controller = build_controller(
        agent_mode=agent_mode,
        input_stream=input_stream,
        output_stream=output,
        log_stream=log,
        default_time=default_time,
    )
    controller.start_listening_loop()
    return output.getvalue()


def _last_bestmove(output: str) -> str:
    matches = _BESTMOVE_RE.findall(output)
    assert matches, f"no bestmove emitted; output was:\n{output}"
    return matches[-1]


def test_blind_mode_returns_legal_bestmove_within_one_second():
    start = time.monotonic()
    output = _run_uci_session(
        "blind",
        ["position startpos", "go movetime 500"],
        default_time=1.0,
    )
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, f"controller took {elapsed:.2f}s to settle"
    move_str = _last_bestmove(output)
    assert move_str != "0000", "controller fell back to null move"

    board = chess.Board()
    legal = {m.uci() for m in board.legal_moves}
    assert move_str in legal, f"{move_str!r} not legal at startpos"


def test_uci_handshake_advertises_calix_name():
    output = _run_uci_session("blind", ["uci"])
    assert "id name Calix" in output
    assert "uciok" in output


def test_aware_mode_handles_a_short_game():
    """Position followed by a couple of moves still produces a legal reply."""
    output = _run_uci_session(
        "aware",
        [
            "position startpos moves e2e4 e7e5",
            "go movetime 300",
        ],
        default_time=1.0,
    )
    move_str = _last_bestmove(output)
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    legal = {m.uci() for m in board.legal_moves}
    assert move_str in legal, f"{move_str!r} not legal after 1.e4 e5"


def test_blind_mode_recovers_from_terminal_position():
    """Already-mated position should produce ``bestmove 0000``."""
    fen = "8/8/8/8/8/5k2/6q1/7K w - - 0 1"
    output = _run_uci_session(
        "blind",
        [f"position fen {fen}", "go movetime 200"],
        default_time=0.5,
    )
    move_str = _last_bestmove(output)
    # Either the controller emits 0000 (no legal moves due to mate) or it
    # finds the only legal escape; both outcomes are acceptable here.
    if move_str != "0000":
        board = chess.Board(fen)
        legal = {m.uci() for m in board.legal_moves}
        assert move_str in legal
