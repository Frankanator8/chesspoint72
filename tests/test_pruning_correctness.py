"""Correctness harness for the forward-pruning module."""
from __future__ import annotations

from dataclasses import replace

import chess

from chesspoint72.engine.boards.pychess import PyChessBoard
from chesspoint72.engine.core.board import Board
from chesspoint72.engine.core.evaluator import Evaluator
from chesspoint72.engine.core.policies import MoveOrderingPolicy
from chesspoint72.engine.core.transposition import TranspositionTable
from chesspoint72.engine.core.types import Move
from chesspoint72.engine.pruning import algorithms
from chesspoint72.engine.pruning.config import PruningConfig, default_pruning_config
from chesspoint72.engine.pruning.policy import ForwardPruningPolicy
from chesspoint72.engine.search.negamax import NegamaxSearch


# --------------------------------------------------------------------------- #
# Local test helpers (adapted from forward_pruning/_test_support.py)
# --------------------------------------------------------------------------- #

_PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


class _MaterialEvaluator(Evaluator):
    def evaluate_position(self, board: Board) -> int:
        cb: chess.Board = board._py_board  # type: ignore[attr-defined]
        score = 0
        for piece_type, value in _PIECE_VALUE.items():
            score += value * (
                len(cb.pieces(piece_type, chess.WHITE))
                - len(cb.pieces(piece_type, chess.BLACK))
            )
        return score if cb.turn == chess.WHITE else -score


class _CapturesFirstOrdering(MoveOrderingPolicy):
    def order_moves(
        self,
        moves: list[Move],
        board: Board,
        tt_best_move: Move | None = None,
    ) -> list[Move]:
        captures = [m for m in moves if m.is_capture]
        quiets = [m for m in moves if not m.is_capture]
        ordered = captures + quiets
        if tt_best_move is not None:
            for i, m in enumerate(ordered):
                if (
                    m.from_square == tt_best_move.from_square
                    and m.to_square == tt_best_move.to_square
                    and m.promotion_piece == tt_best_move.promotion_piece
                ):
                    ordered.pop(i)
                    ordered.insert(0, m)
                    break
        return ordered


def _make_board(fen: str | None = None) -> PyChessBoard:
    b = PyChessBoard()
    if fen:
        b.set_position_from_fen(fen)
    return b


def _build_search(config: PruningConfig) -> NegamaxSearch:
    policy = ForwardPruningPolicy(config)
    return NegamaxSearch(
        evaluator=_MaterialEvaluator(),
        transposition_table=TranspositionTable(),
        move_ordering_policy=_CapturesFirstOrdering(),
        pruning_policy=policy,
        pruning_config=config,
    )


def _all_off() -> PruningConfig:
    cfg = default_pruning_config()
    return replace(cfg, nmp_enabled=False, futility_enabled=False,
                   razoring_enabled=False, lmr_enabled=False)


# --------------------------------------------------------------------------- #
# (1) Perft / move-generation correctness
# --------------------------------------------------------------------------- #

PERFT_STARTING = {1: 20, 2: 400, 3: 8902, 4: 197281}


def _adapter_perft(fen: str | None, depth: int) -> int:
    return _adapter_perft_inner(_make_board(fen), depth)


def _adapter_perft_inner(b: PyChessBoard, depth: int) -> int:
    if depth == 0:
        return 1
    nodes = 0
    for m in b.generate_legal_moves():
        b.make_move(m)
        nodes += _adapter_perft_inner(b, depth - 1)
        b.unmake_move()
    return nodes


def test_perft_baseline_starting_position():
    for depth in (1, 2, 3, 4):
        expected = PERFT_STARTING[depth]
        got = _adapter_perft(None, depth)
        assert got == expected, f"perft({depth}) = {got}, expected {expected}"


def test_search_equivalence_pruning_on_vs_off():
    """Pruning must not change the chosen move on tactical positions at depth 4."""
    cases = [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
    ]
    depth = 4
    for fen in cases:
        off = _build_search(_all_off())
        on = _build_search(default_pruning_config())
        m_off = off.find_best_move(_make_board(fen), max_depth=depth, allotted_time=30.0)
        m_on = on.find_best_move(_make_board(fen), max_depth=depth, allotted_time=30.0)
        assert m_off is not None and m_on is not None, f"no move returned for {fen}"
        assert m_off.to_uci_string() == m_on.to_uci_string(), (
            f"pruning changed best move on {fen}: "
            f"off={m_off.to_uci_string()} vs on={m_on.to_uci_string()}"
        )


# --------------------------------------------------------------------------- #
# (2) Tactical puzzles
# --------------------------------------------------------------------------- #

TACTICS = [
    ("k7/p7/8/8/3q4/8/P2Q4/4K3 w - - 0 1", {"d2d4"}),
    ("r3k3/8/8/8/8/8/8/R3K3 w - - 0 1", {"a1a8"}),
    ("4k3/8/3n4/8/8/3R4/8/4K3 w - - 0 1", {"d3d6"}),
]


def test_tactics_with_pruning_on():
    config = default_pruning_config()
    for fen, expected_moves in TACTICS:
        search = _build_search(config)
        chosen = search.find_best_move(_make_board(fen), max_depth=6, allotted_time=30.0)
        assert chosen is not None, f"no move for {fen}"
        uci = chosen.to_uci_string()
        if uci in expected_moves:
            continue

        diagnostics = []
        for flag in ("nmp_enabled", "futility_enabled", "razoring_enabled", "lmr_enabled"):
            cfg = replace(default_pruning_config(), **{flag: False})
            s = _build_search(cfg)
            m = s.find_best_move(_make_board(fen), max_depth=6, allotted_time=30.0)
            diagnostics.append(f"  with {flag}=False → {m.to_uci_string()}")
        raise AssertionError(
            f"tactic FAILED on {fen}: chose {uci}, expected one of {expected_moves}\n"
            + "\n".join(diagnostics)
        )


# --------------------------------------------------------------------------- #
# (3) Zugzwang tests
# --------------------------------------------------------------------------- #

ZUGZWANG_FEN = "8/8/8/4k3/4p3/4K3/8/8 w - - 0 1"


def test_zugzwang_detector_fires():
    b = _make_board(ZUGZWANG_FEN)
    assert algorithms.is_zugzwang_risk(b), (
        "is_zugzwang_risk should fire: white has only K here"
    )


def test_nmp_skipped_in_zugzwang_risk():
    """try_null_move_pruning must return None when zugzwang risk is detected."""
    b = _make_board(ZUGZWANG_FEN)
    config = default_pruning_config()
    search = _build_search(config)
    result = algorithms.try_null_move_pruning(
        search, b, depth=5, beta=10_000, in_check=False, config=config,
    )
    assert result is None, "NMP must not fire in a zugzwang-prone material config"


def test_search_finds_legal_move_in_zugzwang_position():
    """End-to-end: search returns a legal move from the zugzwang test position."""
    b = _make_board(ZUGZWANG_FEN)
    legal_uci = {m.to_uci_string() for m in b.generate_legal_moves()}
    search = _build_search(default_pruning_config())
    chosen = search.find_best_move(b, max_depth=6, allotted_time=15.0)
    assert chosen is not None
    assert chosen.to_uci_string() in legal_uci
