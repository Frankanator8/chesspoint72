"""Tests for Minal's chess engine — all versions."""
from __future__ import annotations

from chesspoint72.aiengines.minal.v1.engine import MinalV1UciController
from chesspoint72.aiengines.minal.v1.engine import build_controller as build_v1
from chesspoint72.aiengines.minal.v1.ordering import MinalV1MoveOrderingPolicy
from chesspoint72.aiengines.minal.v2.engine import MinalV2UciController
from chesspoint72.aiengines.minal.v2.engine import build_controller as build_v2
from chesspoint72.aiengines.minal.v2.ordering import MinalV2MoveOrderingPolicy
from chesspoint72.aiengines.minal.v2.search import MinalV2Search
from chesspoint72.aiengines.minal.v3.engine import MinalV3UciController
from chesspoint72.aiengines.minal.v3.engine import build_controller as build_v3
from chesspoint72.aiengines.minal.v3.evaluator import MinalV3Evaluator
from chesspoint72.aiengines.minal.v3.ordering import MinalV3MoveOrderingPolicy
from chesspoint72.aiengines.minal.v3.search import MinalV3Search
from chesspoint72.engine.boards import PyChessBoard


# ---------------------------------------------------------------------------
# v1 tests
# ---------------------------------------------------------------------------

def test_minal_v1_builds_controller() -> None:
    controller = build_v1(hce_modules="material,pst")
    assert controller is not None
    assert isinstance(controller, MinalV1UciController)


def test_minal_v1_engine_identity() -> None:
    controller = build_v1(hce_modules="material")
    assert controller.engine_name == "Minal v1"
    assert controller.engine_author == "Minal Sabir"


def test_minal_v1_default_depth() -> None:
    controller = build_v1(hce_modules="material")
    assert controller._default_depth == 6


def test_minal_v1_pruning_config_tuned() -> None:
    controller = build_v1(hce_modules="material")
    cfg = controller.search_engine_reference.pruning_config
    assert cfg.futility_margin == 250
    assert cfg.lmr_min_depth == 2
    assert cfg.lmr_min_move_index == 2
    assert cfg.razoring_margins == (300, 400, 500)


def test_minal_v1_ordering_prefers_tt_move() -> None:
    board = PyChessBoard()
    moves = board.generate_legal_moves()
    tt_move = next(m for m in moves if m.to_uci_string() == "e2e4")

    policy = MinalV1MoveOrderingPolicy()
    ordered = policy.order_moves(moves, board, tt_move)

    assert ordered[0].to_uci_string() == "e2e4"


def test_minal_v1_ordering_winning_capture_before_quiet() -> None:
    """A free queen capture should rank above any quiet move."""
    board = PyChessBoard()
    board.set_position_from_fen("4k3/8/8/4p3/3Q4/8/8/4K3 w - - 0 1")
    moves = board.generate_legal_moves()

    policy = MinalV1MoveOrderingPolicy()
    ordered = policy.order_moves(moves, board)

    top = ordered[0].to_uci_string()
    assert top == "d4e5", f"Expected d4e5 first, got {top}"


def test_minal_v1_ordering_no_crash_on_single_move() -> None:
    board = PyChessBoard()
    board.set_position_from_fen("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1")
    moves = board.generate_legal_moves()
    policy = MinalV1MoveOrderingPolicy()
    result = policy.order_moves(moves, board)
    assert len(result) == len(moves)


def test_minal_v1_finds_move_from_start() -> None:
    controller = build_v1(hce_modules="material,pst", default_depth=3, default_time=5.0)
    board = PyChessBoard()
    move = controller.search_engine_reference.find_best_move(board, max_depth=2, allotted_time=5.0)
    assert move is not None
    legal_ucis = {m.to_uci_string() for m in board.generate_legal_moves()}
    assert move.to_uci_string() in legal_ucis


# ---------------------------------------------------------------------------
# v2 tests
# ---------------------------------------------------------------------------

def test_minal_v2_builds_controller() -> None:
    controller = build_v2(hce_modules="material,pst")
    assert controller is not None
    assert isinstance(controller, MinalV2UciController)


def test_minal_v2_engine_identity() -> None:
    controller = build_v2(hce_modules="material")
    assert controller.engine_name == "Minal v2"
    assert controller.engine_author == "Minal Sabir"


def test_minal_v2_uses_custom_search() -> None:
    controller = build_v2(hce_modules="material")
    assert isinstance(controller.search_engine_reference, MinalV2Search)


def test_minal_v2_pruning_config() -> None:
    controller = build_v2(hce_modules="material")
    cfg = controller.search_engine_reference.pruning_config
    assert cfg.futility_margin == 275
    assert cfg.lmr_min_depth == 3
    assert cfg.lmr_min_move_index == 3
    assert cfg.razoring_margins == (325, 425, 525)


def test_minal_v2_ordering_prefers_tt_move() -> None:
    board = PyChessBoard()
    moves = board.generate_legal_moves()
    tt_move = next(m for m in moves if m.to_uci_string() == "e2e4")

    policy = MinalV2MoveOrderingPolicy()
    ordered = policy.order_moves(moves, board, tt_move)

    assert ordered[0].to_uci_string() == "e2e4"


def test_minal_v2_ordering_winning_capture_first() -> None:
    """Free queen-takes-pawn must rank above all quiet moves."""
    board = PyChessBoard()
    board.set_position_from_fen("4k3/8/8/4p3/3Q4/8/8/4K3 w - - 0 1")
    moves = board.generate_legal_moves()

    policy = MinalV2MoveOrderingPolicy()
    ordered = policy.order_moves(moves, board)

    assert ordered[0].to_uci_string() == "d4e5"


def test_minal_v2_ordering_killer_above_quiet() -> None:
    """A killer move must rank above generic quiet moves."""
    from chesspoint72.engine.core.types import Move
    from chesspoint72.engine.ordering.heuristics import KillerMoveTable, HistoryTable

    board = PyChessBoard()
    moves = board.generate_legal_moves()

    # Pick a quiet move (e.g. g1f3) to register as a killer.
    killer = next(m for m in moves if m.to_uci_string() == "g1f3")
    non_killer = next(m for m in moves if m.to_uci_string() == "a2a3")

    class _FakeSearch:
        _ply = 0
        killer_table = KillerMoveTable()
        history_table = HistoryTable()

    fake = _FakeSearch()
    fake.killer_table.update(killer, ply := 0)

    policy = MinalV2MoveOrderingPolicy()
    policy.attach_search(fake)
    ordered = policy.order_moves(moves, board)

    killer_rank = next(i for i, m in enumerate(ordered) if m.to_uci_string() == "g1f3")
    non_killer_rank = next(i for i, m in enumerate(ordered) if m.to_uci_string() == "a2a3")
    assert killer_rank < non_killer_rank, "killer should rank above non-killer quiet move"


def test_minal_v2_finds_legal_move() -> None:
    controller = build_v2(hce_modules="material,pst", default_depth=3, default_time=5.0)
    board = PyChessBoard()
    move = controller.search_engine_reference.find_best_move(board, max_depth=2, allotted_time=5.0)
    assert move is not None
    legal_ucis = {m.to_uci_string() for m in board.generate_legal_moves()}
    assert move.to_uci_string() in legal_ucis


def test_minal_v2_check_extension_does_not_crash() -> None:
    """Engine must handle positions where king is in check without error."""
    board = PyChessBoard()
    # Scholar's mate setup — white is delivering check
    board.set_position_from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    controller = build_v2(hce_modules="material,pst", default_depth=3, default_time=5.0)
    move = controller.search_engine_reference.find_best_move(board, max_depth=3, allotted_time=5.0)
    assert move is not None
    legal_ucis = {m.to_uci_string() for m in board.generate_legal_moves()}
    assert move.to_uci_string() in legal_ucis


# ---------------------------------------------------------------------------
# v3 tests
# ---------------------------------------------------------------------------

def test_minal_v3_builds_controller() -> None:
    controller = build_v3(hce_modules="material,pst")
    assert controller is not None
    assert isinstance(controller, MinalV3UciController)


def test_minal_v3_engine_identity() -> None:
    controller = build_v3(hce_modules="material")
    assert controller.engine_name == "Minal v3"
    assert controller.engine_author == "Minal Sabir"


def test_minal_v3_uses_custom_search_and_evaluator() -> None:
    controller = build_v3(hce_modules="material")
    assert isinstance(controller.search_engine_reference, MinalV3Search)
    assert isinstance(controller.search_engine_reference.evaluator_reference, MinalV3Evaluator)


def test_minal_v3_tempo_bonus() -> None:
    """V3 evaluator must score strictly higher than bare HCE (tempo bonus)."""
    from chesspoint72.engine.factory import build_evaluator
    board = PyChessBoard()
    hce = build_evaluator("hce", "material")
    v3_eval = MinalV3Evaluator(hce)
    assert v3_eval.evaluate_position(board) == hce.evaluate_position(board) + 15


def test_minal_v3_ordering_countermove_above_quiet() -> None:
    """Countermove must rank above generic quiet moves."""
    from chesspoint72.engine.ordering.heuristics import KillerMoveTable, HistoryTable

    board = PyChessBoard()
    moves = board.generate_legal_moves()
    # Use d2d4 as the "opponent's last move" and g1f3 as its countermove.
    prev_move = next(m for m in moves if m.to_uci_string() == "d2d4")
    counter = next(m for m in moves if m.to_uci_string() == "g1f3")
    plain_quiet = next(m for m in moves if m.to_uci_string() == "a2a3")

    cm_table = [[None] * 64 for _ in range(64)]
    cm_table[prev_move.from_square][prev_move.to_square] = counter

    class _FakeSearch:
        _ply = 0
        _prev_move = prev_move
        killer_table = KillerMoveTable()
        history_table = HistoryTable()
        countermove_table = cm_table

    policy = MinalV3MoveOrderingPolicy()
    policy.attach_search(_FakeSearch())
    ordered = policy.order_moves(moves, board)

    counter_rank = next(i for i, m in enumerate(ordered) if m.to_uci_string() == "g1f3")
    quiet_rank = next(i for i, m in enumerate(ordered) if m.to_uci_string() == "a2a3")
    assert counter_rank < quiet_rank, "countermove should rank above plain quiet move"


def test_minal_v3_ordering_winning_capture_first() -> None:
    board = PyChessBoard()
    board.set_position_from_fen("4k3/8/8/4p3/3Q4/8/8/4K3 w - - 0 1")
    moves = board.generate_legal_moves()
    policy = MinalV3MoveOrderingPolicy()
    ordered = policy.order_moves(moves, board)
    assert ordered[0].to_uci_string() == "d4e5"


def test_minal_v3_finds_legal_move_from_start() -> None:
    controller = build_v3(hce_modules="material,pst", default_depth=3, default_time=5.0)
    board = PyChessBoard()
    move = controller.search_engine_reference.find_best_move(board, max_depth=2, allotted_time=5.0)
    assert move is not None
    legal_ucis = {m.to_uci_string() for m in board.generate_legal_moves()}
    assert move.to_uci_string() in legal_ucis


def test_minal_v3_finds_mate_in_one() -> None:
    """V3 must find checkmate in one move (basic tactical sanity)."""
    board = PyChessBoard()
    # Fool's mate position — Qh5# is the only mate in 1
    board.set_position_from_fen("rnbqkbnr/pppp1ppp/8/4p3/6Pp/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 3")
    controller = build_v3(hce_modules="material,pst")
    move = controller.search_engine_reference.find_best_move(board, max_depth=3, allotted_time=10.0)
    assert move is not None
    legal_ucis = {m.to_uci_string() for m in board.generate_legal_moves()}
    assert move.to_uci_string() in legal_ucis


def test_minal_v3_pruning_config() -> None:
    controller = build_v3(hce_modules="material")
    cfg = controller.search_engine_reference.pruning_config
    assert cfg.futility_margin == 275
    assert cfg.lmr_min_depth == 3
    assert cfg.lmr_min_move_index == 4
