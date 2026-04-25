# @capability: see
"""
Bitboard-optimized Static Exchange Evaluation (SEE).

Implements the Stockfish 16+ swap loop with:
- Pre-computed leaper attack tables  (O(1) lookup per piece type)
- Loop-based ray tracing for sliders (O(ray length), x-ray safe)
- Least-Valuable Attacker selection  (minimises exchange risk each ply)

Public API
----------
    see_ge(board, move, threshold) -> bool
        Returns True iff the exchange result on ``move`` >= ``threshold``.
    SEE_VALUES: list[int]
        Stockfish 16 piece values in centipawns (index = PieceType.value).
"""
from __future__ import annotations

from array import array
from typing import TYPE_CHECKING

from chesspoint72.engine.core.types import PieceType

if TYPE_CHECKING:
    from chesspoint72.engine.core.board import Board
    from chesspoint72.engine.core.types import Move

# ---------------------------------------------------------------------------
# Piece values — Stockfish 16 exact values (centipawns).
# Indexed by PieceType.value: 1=PAWN..6=KING.  Index 0 is unused (sentinel).
# ---------------------------------------------------------------------------
SEE_VALUES: list[int] = [0, 208, 781, 825, 1276, 2538, 20_000]

# ---------------------------------------------------------------------------
# Bitboard constants
# ---------------------------------------------------------------------------
_MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
_A_FILE: int = 0x0101_0101_0101_0101   # squares a1,a2,...,a8
_H_FILE: int = 0x8080_8080_8080_8080   # squares h1,h2,...,h8
_NOT_A:  int = ~_A_FILE & _MASK64
_NOT_H:  int = ~_H_FILE & _MASK64
_NOT_AB: int = 0xFCFC_FCFC_FCFC_FCFC   # not file A or B  (knight jump guard)
_NOT_GH: int = 0x3F3F_3F3F_3F3F_3F3F   # not file G or H  (knight jump guard)

# ---------------------------------------------------------------------------
# Pre-computed leaper attack tables (64-entry arrays of unsigned 64-bit ints)
# ---------------------------------------------------------------------------
_KNIGHT_ATK: array = array("Q", [0] * 64)
_KING_ATK:   array = array("Q", [0] * 64)
# _PAWN_ATK[color.value][sq] — White(0) attacks NE/NW, Black(1) attacks SE/SW
_PAWN_ATK: list[array] = [array("Q", [0] * 64), array("Q", [0] * 64)]


def _init_leaper_tables() -> None:
    for sq in range(64):
        b = 1 << sq

        # Knight: up to 8 L-shaped jumps; file guards prevent rank wrapping
        kn = (
            ((b << 17) & _NOT_A)
            | ((b << 15) & _NOT_H)
            | ((b << 10) & _NOT_AB)
            | ((b <<  6) & _NOT_GH)
            | ((b >>  6) & _NOT_AB)
            | ((b >> 10) & _NOT_GH)
            | ((b >> 15) & _NOT_A)
            | ((b >> 17) & _NOT_H)
        )
        _KNIGHT_ATK[sq] = kn & _MASK64

        # King: 8 adjacent squares; file guards prevent rank wrapping
        kg = (
            (b << 8)
            | (b >> 8)
            | ((b << 1) & _NOT_A)
            | ((b >> 1) & _NOT_H)
            | ((b << 9) & _NOT_A)
            | ((b >> 9) & _NOT_H)
            | ((b << 7) & _NOT_H)
            | ((b >> 7) & _NOT_A)
        )
        _KING_ATK[sq] = kg & _MASK64

        # Pawn attacks: White pawn at sq attacks NE and NW (toward higher ranks)
        wp = ((b << 9) & _NOT_A) | ((b << 7) & _NOT_H)
        # Black pawn at sq attacks SE and SW (toward lower ranks)
        bp = ((b >> 7) & _NOT_A) | ((b >> 9) & _NOT_H)
        _PAWN_ATK[0][sq] = wp & _MASK64
        _PAWN_ATK[1][sq] = bp & _MASK64


_init_leaper_tables()

# ---------------------------------------------------------------------------
# Slider attack generation — loop-based ray tracer, correct for any occupancy
# ---------------------------------------------------------------------------

def _rook_attacks_occ(sq: int, occ: int) -> int:
    """All squares a rook on *sq* attacks given occupancy *occ*."""
    rank = sq >> 3
    file = sq & 7
    attacks = 0
    # North (+rank)
    for r in range(rank + 1, 8):
        s = r * 8 + file
        b = 1 << s
        attacks |= b
        if occ & b:
            break
    # South (-rank)
    for r in range(rank - 1, -1, -1):
        s = r * 8 + file
        b = 1 << s
        attacks |= b
        if occ & b:
            break
    # East (+file)
    for f in range(file + 1, 8):
        s = rank * 8 + f
        b = 1 << s
        attacks |= b
        if occ & b:
            break
    # West (-file)
    for f in range(file - 1, -1, -1):
        s = rank * 8 + f
        b = 1 << s
        attacks |= b
        if occ & b:
            break
    return attacks


def _bishop_attacks_occ(sq: int, occ: int) -> int:
    """All squares a bishop on *sq* attacks given occupancy *occ*."""
    rank = sq >> 3
    file = sq & 7
    attacks = 0
    for dr, df in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        r, f = rank + dr, file + df
        while 0 <= r < 8 and 0 <= f < 8:
            b = 1 << (r * 8 + f)
            attacks |= b
            if occ & b:
                break
            r += dr
            f += df
    return attacks


# ---------------------------------------------------------------------------
# Bitboard helpers
# ---------------------------------------------------------------------------

def _lsb_bb(bb: int) -> int:
    """Return a bitboard containing only the least-significant set bit."""
    return bb & (-bb)


def _piece_type_val(bbs: list[int], sq: int) -> int:
    """PieceType.value (1-6) of the piece on *sq*, or 0 if empty."""
    bit = 1 << sq
    if bbs[0]  & bit: return 1   # wP
    if bbs[6]  & bit: return 1   # bP
    if bbs[1]  & bit: return 2   # wN
    if bbs[7]  & bit: return 2   # bN
    if bbs[2]  & bit: return 3   # wB
    if bbs[8]  & bit: return 3   # bB
    if bbs[3]  & bit: return 4   # wR
    if bbs[9]  & bit: return 4   # bR
    if bbs[4]  & bit: return 5   # wQ
    if bbs[10] & bit: return 5   # bQ
    if bbs[5]  & bit: return 6   # wK
    if bbs[11] & bit: return 6   # bK
    return 0


# ---------------------------------------------------------------------------
# _attackers_to — all pieces on any side attacking square *sq*
# ---------------------------------------------------------------------------

def _attackers_to(sq: int, occ: int, bbs: list[int]) -> int:
    """Bitboard of every piece attacking *sq* given occupancy *occ*.

    The *occ* parameter is threaded through slider computations so that
    removing a piece from *occ* in the SEE swap loop automatically reveals
    any x-ray attackers sitting behind it.

    Bitboard layout (``board.bitboards``):
        index = color.value * 6 + (piece_type.value - 1)
        [0]=wP [1]=wN [2]=wB [3]=wR [4]=wQ [5]=wK
        [6]=bP [7]=bN [8]=bB [9]=bR [10]=bQ [11]=bK

    Pawn direction:
        White pawns attack from squares a BLACK pawn at *sq* would see.
        Black pawns attack from squares a WHITE pawn at *sq* would see.
    """
    bishop_queen = bbs[2] | bbs[8] | bbs[4] | bbs[10]
    rook_queen   = bbs[3] | bbs[9] | bbs[4] | bbs[10]
    return (
        (_PAWN_ATK[1][sq] & bbs[0])                            # wP attacking sq
        | (_PAWN_ATK[0][sq] & bbs[6])                          # bP attacking sq
        | (_KNIGHT_ATK[sq] & (bbs[1] | bbs[7]))                # all knights
        | (_KING_ATK[sq]   & (bbs[5] | bbs[11]))               # all kings
        | (_bishop_attacks_occ(sq, occ) & bishop_queen)        # diagonal sliders
        | (_rook_attacks_occ(sq, occ)   & rook_queen)          # orthogonal sliders
    )


# ---------------------------------------------------------------------------
# Public SEE entry point
# ---------------------------------------------------------------------------

def see_ge(board: Board, move: Move, threshold: int) -> bool:
    """Return True iff the static exchange on *move* is >= *threshold*.

    Algorithm
    ---------
    Stockfish 16 swap loop with Least-Valuable Attacker selection.

    1. Quick exits before the loop: if the initial gain is already
       sufficient (or definitely insufficient) without any recapture.
    2. Loop: alternate sides, pick LVA each ply, remove it from
       occupancy (revealing x-ray attackers), break when no attacker
       remains or the recapture is clearly sub-optimal.

    Non-normal moves (promotions) and en passant (victim not on to_sq)
    are handled conservatively: returns ``0 >= threshold``.

    Args:
        board:     Position supplying bitboards and side_to_move.
        move:      The capture to evaluate.
        threshold: Minimum required exchange gain (centipawns).
                   Pass 0 for "is this capture winning?",
                   negative to allow modest losses.
    """
    # Promotions and en passant: skip the exchange logic
    if move.promotion_piece is not None:
        return 0 >= threshold

    from_sq = move.from_square
    to_sq   = move.to_square
    bbs     = board.bitboards

    victim_type = _piece_type_val(bbs, to_sq)
    moving_type = _piece_type_val(bbs, from_sq)

    # En passant: captured pawn not on to_sq → victim_type == 0
    if victim_type == 0:
        return 0 >= threshold

    # ---- Quick exit 1: can we even reach threshold by gaining the victim? ----
    swap = SEE_VALUES[victim_type] - threshold
    if swap < 0:
        return False

    # ---- Quick exit 2: even if they recapture our piece for free, we win ----
    swap = SEE_VALUES[moving_type] - swap
    if swap <= 0:
        return True

    # ---- Full swap loop ----
    # Build initial occupancy with BOTH the moving piece and captured piece removed.
    # (The capture has conceptually happened: from_sq is emptied, to_sq's victim
    # is removed.  The moving piece "passes through" to_sq for x-ray purposes.)
    occ = 0
    for bb in bbs:
        occ |= bb
    occ = (occ ^ (1 << from_sq) ^ (1 << to_sq)) & _MASK64

    # Attackers on to_sq with this initial occupancy
    attackers = _attackers_to(to_sq, occ, bbs)

    # stm is the side that just captured (initiator); the loop immediately
    # flips it so the OPPONENT is first to consider recapture.
    stm = board.side_to_move.value  # 0=WHITE, 1=BLACK
    res = 1  # initiator is "winning" by default; opponent has to disprove it

    while True:
        stm ^= 1  # switch sides: opponent recaptures first
        attackers &= occ  # drop pieces already removed from the board

        # Pieces of the side to move that can still attack to_sq
        base = stm * 6
        stm_attackers = attackers & (
            bbs[base] | bbs[base + 1] | bbs[base + 2]
            | bbs[base + 3] | bbs[base + 4] | bbs[base + 5]
        )
        if not stm_attackers:
            break  # no recapture available; current res holds

        res ^= 1  # this side will recapture; flip who is "ahead"

        # Pre-cache slider union bitboards (recalculated only when occ changes)
        bq = bbs[2] | bbs[8] | bbs[4] | bbs[10]  # bishops + queens
        rq = bbs[3] | bbs[9] | bbs[4] | bbs[10]  # rooks   + queens

        # --- Least-Valuable Attacker selection ---

        # PAWN (value = 208)
        bb = stm_attackers & bbs[base]
        if bb:
            swap = 208 - swap
            if swap < res:
                break
            occ ^= _lsb_bb(bb)
            # Removing a pawn may reveal a diagonal slider
            attackers |= _bishop_attacks_occ(to_sq, occ) & bq
            continue

        # KNIGHT (value = 781)
        bb = stm_attackers & bbs[base + 1]
        if bb:
            swap = 781 - swap
            if swap < res:
                break
            occ ^= _lsb_bb(bb)
            # Knights never occlude sliders
            continue

        # BISHOP (value = 825)
        bb = stm_attackers & bbs[base + 2]
        if bb:
            swap = 825 - swap
            if swap < res:
                break
            occ ^= _lsb_bb(bb)
            attackers |= _bishop_attacks_occ(to_sq, occ) & bq
            continue

        # ROOK (value = 1276)
        bb = stm_attackers & bbs[base + 3]
        if bb:
            swap = 1276 - swap
            if swap < res:
                break
            occ ^= _lsb_bb(bb)
            attackers |= _rook_attacks_occ(to_sq, occ) & rq
            continue

        # QUEEN (value = 2538)
        bb = stm_attackers & bbs[base + 4]
        if bb:
            swap = 2538 - swap
            if swap < res:
                break
            occ ^= _lsb_bb(bb)
            # Queens reveal both diagonal and orthogonal x-rays
            attackers |= (
                _bishop_attacks_occ(to_sq, occ) & bq
                | _rook_attacks_occ(to_sq, occ) & rq
            )
            continue

        # KING (value = 20000) — legal only when opponent has no more attackers
        bb = stm_attackers & bbs[base + 5]
        if bb:
            opp_base = (1 - stm) * 6
            opp_still_on_sq = attackers & (
                bbs[opp_base] | bbs[opp_base + 1] | bbs[opp_base + 2]
                | bbs[opp_base + 3] | bbs[opp_base + 4] | bbs[opp_base + 5]
            )
            # King cannot capture into a defended square; flip result if it must
            return bool(res ^ 1 if opp_still_on_sq else res)

        break  # no attacker found for this side (shouldn't normally reach here)

    return bool(res)
