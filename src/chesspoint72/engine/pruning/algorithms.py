"""Forward pruning primitives for an alpha-beta search.

Each function is a thin algorithmic core with no I/O or logging.
The module is engine-agnostic: it duck-types its board and search
arguments rather than requiring specific base classes.
"""
# @capability: pruning
# @capability: null_move_pruning
# @capability: razoring
# @capability: futility_pruning
# @capability: lmr
from __future__ import annotations

import math


def is_zugzwang_risk(position) -> bool:
    """Return True if the side to move has only king and pawns.

    The classic NMP failure mode: in king-and-pawn endgames the side to move
    often has no useful move, so giving them a free move inverts the search
    result. NMP is vetoed whenever this condition holds.
    """
    return position.has_only_king_and_pawns(position.side_to_move)


def try_null_move_pruning(
    search,
    board,
    depth: int,
    beta: int,
    in_check: bool,
    config,
) -> int | None:
    """Attempt a null-move reduced-depth probe.

    Returns ``beta`` if the probe succeeds (prune); ``None`` to continue.

    Conditions (all must hold):
      * config.nmp_enabled
      * depth >= 3
      * not in check
      * not is_zugzwang_risk(board)
    """
    if not config.nmp_enabled:
        return None
    if depth < 3:
        return None
    if in_check:
        return None
    if is_zugzwang_risk(board):
        return None

    reduction = 2 if depth < 6 else 3
    reduced_depth = max(0, depth - reduction - 1)

    board.make_null_move()
    try:
        score = -search.search_node(-beta, -beta + 1, reduced_depth)
    finally:
        board.unmake_null_move()

    if score >= beta:
        return beta
    return None


def try_razoring(
    search,
    depth: int,
    alpha: int,
    static_eval: int,
    config,
) -> int | None:
    """Drop into quiescence search when the position looks hopeless.

    Activates at depth 2..4 when ``static_eval + margin < alpha``.
    Returns the QS score if razoring fires and QS confirms; ``None`` otherwise.
    """
    if not config.razoring_enabled:
        return None
    if depth < 2 or depth > 4:
        return None

    margin = config.razoring_margins[depth - 2]
    if static_eval + margin >= alpha:
        return None

    qs_score = search.quiescence_search(alpha - 1, alpha)
    if qs_score < alpha:
        return qs_score
    return None


def is_futile(
    depth: int,
    alpha: int,
    static_eval: int,
    in_check: bool,
    move_is_quiet: bool,
    config,
) -> bool:
    """Return True if this quiet move should be skipped at frontier depth.

    Activates only at depth == 1. A quiet move whose static_eval already
    trails alpha by more than the futility margin cannot plausibly raise alpha
    (since only QS follows, which only resolves captures).
    """
    if not config.futility_enabled:
        return False
    if depth != 1:
        return False
    if in_check:
        return False
    if not move_is_quiet:
        return False
    return static_eval + config.futility_margin < alpha


def lmr_reduction(depth: int, move_index: int) -> int:
    """Compute the LMR depth reduction.

    Formula: ``max(1, floor(log(depth) * log(move_index + 1) / 2))``.
    """
    if depth <= 1 or move_index < 0:
        return 1
    raw = math.log(depth) * math.log(move_index + 1) / 2.0
    return max(1, int(math.floor(raw)))


def should_apply_lmr(
    depth: int,
    move_index: int,
    move_is_quiet: bool,
    in_check: bool,
    gives_check: bool,
    config,
) -> bool:
    """Gate for late move reductions.

    Reduces only when: LMR enabled, depth >= min, move_index > min,
    the move is quiet, and neither side is in check.
    """
    if not config.lmr_enabled:
        return False
    if depth < config.lmr_min_depth:
        return False
    if move_index <= config.lmr_min_move_index:
        return False
    if not move_is_quiet:
        return False
    if in_check or gives_check:
        return False
    return True
