"""Forward pruning primitives for an alpha-beta search.

Each function is a thin algorithmic core:

* It is told whether its config flag is enabled by the caller checking
  ``config.<flag>_enabled`` first, OR it short-circuits internally — both
  styles are supported via the helpers below. Either way, no defaults are
  invented here; the ``PruningConfig`` struct is owned by a separate module.
* No file scaffolding, no I/O, no logging. Pure search-loop helpers.

The module is engine-agnostic. It depends only on:
  - a ``Search`` host that exposes ``search_node(alpha, beta, depth)`` and
    ``quiescence_search(alpha, beta)``;
  - a ``Board`` host with ``side_to_move``, ``make_null_move() / unmake_null_move()``
    (used by NMP), and ``has_only_king_and_pawns(side)`` (used by zugzwang);
  - a ``PruningConfig`` value with the flags and margins documented in
    ``INTERFACE_CONTRACT.md``.

These contracts are expressed as ``typing.Protocol`` classes so any concrete
board / search can plug in without inheritance.
"""
from __future__ import annotations

import math
from typing import Protocol, runtime_checkable


# --------------------------------------------------------------------------- #
# Structural contracts (Protocols, not ABCs — engine code does not inherit).
# --------------------------------------------------------------------------- #


@runtime_checkable
class ZugzwangProbe(Protocol):
    """Minimum surface a board must expose for the zugzwang detector."""

    side_to_move: int  # Color enum value; treated as opaque

    def has_only_king_and_pawns(self, side: int) -> bool: ...


@runtime_checkable
class NullMoveBoard(Protocol):
    """Board surface required by null-move pruning."""

    side_to_move: int

    def make_null_move(self) -> None: ...
    def unmake_null_move(self) -> None: ...
    def has_only_king_and_pawns(self, side: int) -> bool: ...


class SearchHost(Protocol):
    """The negamax search instance the pruning module calls back into."""

    def search_node(self, alpha: int, beta: int, depth: int) -> int: ...
    def quiescence_search(self, alpha: int, beta: int) -> int: ...


# --------------------------------------------------------------------------- #
# Zugzwang detector
# --------------------------------------------------------------------------- #


def is_zugzwang_risk(position: ZugzwangProbe) -> bool:
    """Return True if the side to move is in a zugzwang-prone material config.

    The classic NMP failure mode: in king-and-pawn endgames the side to move
    often has no useful move, so giving them a *free* move (the null move)
    inverts the truth value of the search. We veto NMP whenever the side to
    move has nothing but king and pawns.

    Hot path — called at every NMP candidate node. Keep cheap.
    """
    return position.has_only_king_and_pawns(position.side_to_move)


# --------------------------------------------------------------------------- #
# Null-move pruning
# --------------------------------------------------------------------------- #


def try_null_move_pruning(
    search: SearchHost,
    board: NullMoveBoard,
    depth: int,
    beta: int,
    in_check: bool,
    config,
) -> int | None:
    """Attempt a null-move reduced-depth probe.

    Returns ``beta`` (the standard fail-soft "beta cutoff" sentinel) if the
    probe succeeds; ``None`` if NMP does not fire and the caller should
    continue with the normal search.

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
    reduced_depth = depth - reduction - 1
    if reduced_depth < 0:
        # Depth budget exhausted by reduction — drop into QS via depth=0.
        reduced_depth = 0

    board.make_null_move()
    try:
        # Null-window probe centred on beta. The "side to move" has flipped,
        # so from the new side's POV we want to disprove (-beta+1, -beta).
        score = -search.search_node(-beta, -beta + 1, reduced_depth)
    finally:
        board.unmake_null_move()

    if score >= beta:
        return beta
    return None


# --------------------------------------------------------------------------- #
# Razoring
# --------------------------------------------------------------------------- #


def try_razoring(
    search: SearchHost,
    depth: int,
    alpha: int,
    static_eval: int,
    config,
) -> int | None:
    """Drop directly into quiescence search when the position looks hopeless.

    Activates at depth 2..4 when ``static_eval + margin < alpha``: the static
    score is so far below the lower bound that no quiet move is plausibly
    going to recover it, so we let QS settle the captures and accept the
    verdict if it agrees.

    Returns the QS score if razoring fires *and* QS confirms the position
    fails low (qs_score < alpha). Returns ``None`` otherwise — meaning either
    razoring did not fire, or QS disagreed and the caller must do the full
    search.

    config.razoring_margins is indexed by ``depth - 2`` and must have at
    least 3 entries (one for each of depths 2, 3, 4).
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


# --------------------------------------------------------------------------- #
# Futility pruning (move-level filter)
# --------------------------------------------------------------------------- #


def is_futile(
    depth: int,
    alpha: int,
    static_eval: int,
    in_check: bool,
    move_is_quiet: bool,
    config,
) -> bool:
    """Return True if the caller should skip this quiet move.

    Activates only at frontier nodes (depth == 1). At depth 1 the only
    follow-up is QS, which only resolves captures — so a quiet move whose
    static_eval already trails alpha by more than the futility margin
    cannot plausibly raise alpha.
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


# --------------------------------------------------------------------------- #
# Late move reductions
# --------------------------------------------------------------------------- #


def lmr_reduction(depth: int, move_index: int) -> int:
    """Compute the LMR depth reduction.

    Formula: ``r = max(1, floor(log(depth) * log(move_index + 1) / 2))``.

    The two logs combine independent signals (remaining depth and move
    rank) into a smoothly-growing reduction. The floor of 1 ensures LMR
    always actually reduces when invoked, even at the boundary
    (e.g. depth=3, move_index=4 → product ~0.88 → floor 0 → forced to 1).
    """
    # Defensive: log(0) and log(<=0) blow up; both are gated out by the
    # caller's should_apply_lmr check, but guard anyway so the function is
    # safe in isolation.
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
    """Gate for LMR.

    Only reduce when:
      * LMR is enabled
      * depth >= config.lmr_min_depth (default expectation: 3)
      * move_index > config.lmr_min_move_index (default expectation: 3)
      * the move is quiet (not capture, not promotion)
      * neither the parent is in check nor the move gives check
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
