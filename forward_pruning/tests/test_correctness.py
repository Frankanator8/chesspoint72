"""Phase-5 correctness harness for the forward-pruning module.

Run from repo root with the project venv:

    .venv/bin/python -m pytest forward_pruning/tests/test_correctness.py -v

Or directly:

    .venv/bin/python forward_pruning/tests/test_correctness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo root and src/ on path so this file is runnable both via pytest and
# `python forward_pruning/tests/test_correctness.py`.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import chess

from chesspoint72.engine.transposition import TranspositionTable

from forward_pruning import pruning
from forward_pruning._test_support import (
    CapturesFirstOrdering,
    MaterialEvaluator,
    NullPruningPolicy,
    PythonChessBoard,
    TestPruningConfig,
)
from forward_pruning.search_modified import PrunedNegamaxSearch


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _build_search(config: TestPruningConfig) -> PrunedNegamaxSearch:
    return PrunedNegamaxSearch(
        evaluator=MaterialEvaluator(),
        transposition_table=TranspositionTable(),
        move_ordering_policy=CapturesFirstOrdering(),
        pruning_policy=NullPruningPolicy(),
        pruning_config=config,
    )


def _all_off() -> TestPruningConfig:
    return TestPruningConfig(
        nmp_enabled=False, futility_enabled=False,
        razoring_enabled=False, lmr_enabled=False,
    )


def _python_chess_perft(fen: str | None, depth: int) -> int:
    """Reference perft using python-chess."""
    cb = chess.Board(fen) if fen else chess.Board()
    if depth == 0:
        return 1
    nodes = 0
    for m in cb.legal_moves:
        cb.push(m)
        nodes += _python_chess_perft(cb.fen(), depth - 1)
        cb.pop()
    return nodes


def _adapter_perft(fen: str | None, depth: int) -> int:
    """Perft routed through our PythonChessBoard adapter — proves make/unmake roundtrips."""
    b = PythonChessBoard(fen)
    return _adapter_perft_inner(b, depth)


def _adapter_perft_inner(b: PythonChessBoard, depth: int) -> int:
    if depth == 0:
        return 1
    nodes = 0
    for m in b.generate_legal_moves():
        b.make_move(m)
        nodes += _adapter_perft_inner(b, depth - 1)
        b.unmake_move()
    return nodes


# --------------------------------------------------------------------------- #
# (1) Perft / search-equivalence
# --------------------------------------------------------------------------- #

# Standard perft node counts (chessprogramming wiki).
PERFT_STARTING = {
    1: 20,
    2: 400,
    3: 8902,
    4: 197281,
    5: 4865609,
}


def test_perft_baseline_starting_position():
    """Adapter's move generation matches python-chess at depths 1-4."""
    for depth in (1, 2, 3, 4):
        expected = PERFT_STARTING[depth]
        got = _adapter_perft(None, depth)
        assert got == expected, f"perft({depth}) = {got}, expected {expected}"


def test_search_equivalence_pruning_on_vs_off():
    """Pruning must not change the chosen move on a small set of tactical positions
    at moderate depth. If it does, a pruning path is unsound."""
    cases = [
        # Starting position. Many engines pick e2e4, c2c4, d2d4, g1f3 — any of those
        # is fine; we just need the choice to *agree* between ON and OFF.
        chess.STARTING_FEN,
        # Italian opening, rich tactics.
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
        # Quiet middle-game (Ruy-Lopez-ish).
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
    ]
    depth = 4
    for fen in cases:
        off = _build_search(_all_off())
        on = _build_search(TestPruningConfig())
        b1 = PythonChessBoard(fen)
        b2 = PythonChessBoard(fen)
        m_off = off.find_best_move(b1, max_depth=depth, allotted_time=30.0)
        m_on = on.find_best_move(b2, max_depth=depth, allotted_time=30.0)
        assert m_off is not None and m_on is not None, f"no move returned for {fen}"
        # We require best-move agreement; if pruning changes the move at depth 4
        # from a clean material-only baseline, something is unsound.
        assert m_off.to_uci_string() == m_on.to_uci_string(), (
            f"pruning changed best move on {fen}: off={m_off.to_uci_string()} "
            f"vs on={m_on.to_uci_string()}"
        )


# --------------------------------------------------------------------------- #
# (2) Tactical puzzles
# --------------------------------------------------------------------------- #

# Each entry: (FEN, set of acceptable best moves in UCI). The puzzles are
# chosen so a material-only evaluator can solve them at depth 6 — captures and
# forcing sequences only, no positional judgement required.
TACTICS = [
    (
        # Hanging queen along the d-file. Black queen on d4 is undefended;
        # white queen on d2 captures (Qxd4). Kings tucked into corners so
        # neither side starts in check, and Qd8/Qd1 lines are blocked by
        # the queen-vs-queen geometry — Qxd4 is the only meaningful capture.
        "k7/p7/8/8/3q4/8/P2Q4/4K3 w - - 0 1",
        {"d2d4"},
    ),
    (
        # Hanging rook: black rook on a8 is undefended; white plays Rxa8.
        "r3k3/8/8/8/8/8/8/R3K3 w - - 0 1",
        {"a1a8"},
    ),
    (
        # Hanging knight: black knight on d6 attacked by white rook on d3.
        # Rxd6 must be played NOW — if white delays, black just moves the
        # knight away, so the material window is depth-1-visible and not
        # subject to horizon-equivalence drift.
        "4k3/8/3n4/8/8/3R4/8/4K3 w - - 0 1",
        {"d3d6"},
    ),
]


def test_tactics_with_pruning_on():
    config = TestPruningConfig()
    for fen, expected_moves in TACTICS:
        search = _build_search(config)
        board = PythonChessBoard(fen)
        chosen = search.find_best_move(board, max_depth=6, allotted_time=30.0)
        assert chosen is not None, f"no move for {fen}"
        uci = chosen.to_uci_string()
        if uci in expected_moves:
            continue

        # Failure path: disable techniques one-by-one to isolate.
        diagnostics = []
        for flag in ("nmp_enabled", "futility_enabled", "razoring_enabled", "lmr_enabled"):
            cfg = TestPruningConfig(**{flag: False})
            s = _build_search(cfg)
            b = PythonChessBoard(fen)
            m = s.find_best_move(b, max_depth=6, allotted_time=30.0)
            diagnostics.append(f"  with {flag}=False → {m.to_uci_string()}")
        raise AssertionError(
            f"tactic FAILED on {fen}: chose {uci}, expected one of {expected_moves}\n"
            + "\n".join(diagnostics)
        )


# --------------------------------------------------------------------------- #
# (3) Zugzwang test
# --------------------------------------------------------------------------- #

# White-to-move K+P endgame where passing (null move) loses the opposition and
# the win — the classic NMP pitfall.
ZUGZWANG_FEN = "8/8/8/4k3/4p3/4K3/8/8 w - - 0 1"


def test_zugzwang_detector_fires():
    b = PythonChessBoard(ZUGZWANG_FEN)
    assert pruning.is_zugzwang_risk(b), (
        "is_zugzwang_risk should fire: white has only K+P (in fact only K) here"
    )


def test_nmp_skipped_in_zugzwang_risk():
    """try_null_move_pruning must return None when zugzwang risk is detected,
    even when all other gates are open."""
    b = PythonChessBoard(ZUGZWANG_FEN)
    config = TestPruningConfig()
    search = _build_search(config)
    # Drive the search far enough that NMP would otherwise activate (depth >= 3).
    result = pruning.try_null_move_pruning(
        search, b, depth=5, beta=10_000, in_check=False, config=config,
    )
    assert result is None, "NMP must not fire in a zugzwang-prone material config"


def test_search_finds_legal_move_in_zugzwang_position():
    """End-to-end: even with all pruning enabled, the search returns a legal move
    in the zugzwang test position (and does not crash from a bad NMP path)."""
    b = PythonChessBoard(ZUGZWANG_FEN)
    legal_uci = {m.to_uci_string() for m in b.generate_legal_moves()}
    search = _build_search(TestPruningConfig())
    chosen = search.find_best_move(b, max_depth=6, allotted_time=15.0)
    assert chosen is not None
    assert chosen.to_uci_string() in legal_uci


# --------------------------------------------------------------------------- #
# CLI runner — usable without pytest
# --------------------------------------------------------------------------- #


def _run_all() -> int:
    failures = []
    tests = [
        test_perft_baseline_starting_position,
        test_search_equivalence_pruning_on_vs_off,
        test_tactics_with_pruning_on,
        test_zugzwang_detector_fires,
        test_nmp_skipped_in_zugzwang_risk,
        test_search_finds_legal_move_in_zugzwang_position,
    ]
    for t in tests:
        name = t.__name__
        try:
            t()
        except AssertionError as e:
            failures.append((name, str(e)))
            print(f"  FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
        else:
            print(f"  ok    {name}")
    print()
    if failures:
        print(f"{len(failures)} failed / {len(tests)} total")
        return 1
    print(f"all {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())
