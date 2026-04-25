"""Hand-Crafted Evaluation (HCE) — unified evaluator.

Public API
----------
evaluate(board) -> int
    Centipawns, positive = good for the side to move.
explain(board)  -> dict[str, int]
    Per-feature tapered scores for debugging.
"""
from __future__ import annotations

import chess

# ═══════════════════════════════════════════════════════════════════════════════
# 1 — Constants
# ═══════════════════════════════════════════════════════════════════════════════

MATERIAL_MG: dict[int, int] = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:     0,
}
MATERIAL_EG: dict[int, int] = {
    chess.PAWN:   120,   # promotion threat makes pawns more valuable
    chess.KNIGHT: 310,
    chess.BISHOP: 340,   # open diagonals benefit bishops
    chess.ROOK:   480,
    chess.QUEEN:  890,
    chess.KING:     0,
}

PHASE_WEIGHTS: dict[int, int] = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK:   2,
    chess.QUEEN:  4,
}
MAX_PHASE: int = 24  # 4N×1 + 4B×1 + 4R×2 + 2Q×4

# Pawn structure ── (mg, eg)
DOUBLED_PAWN_PENALTY:  tuple[int, int] = (-10, -20)
ISOLATED_PAWN_PENALTY: tuple[int, int] = (-15, -25)
PASSED_PAWN_BONUS_MG: list[int] = [  0,   5,  10,  20,  35,  60, 100,  0]
PASSED_PAWN_BONUS_EG: list[int] = [  0,  10,  20,  40,  65, 100, 150,  0]

# Rook bonuses ── (mg, eg)
ROOK_OPEN_FILE_BONUS:     tuple[int, int] = (25, 15)
ROOK_SEMIOPEN_FILE_BONUS: tuple[int, int] = (12,  8)
ROOK_SEVENTH_RANK_BONUS:  tuple[int, int] = (20, 30)

# King safety ── eg component is always 0; safety is a midgame concept
KING_OPEN_FILE_PENALTY:     tuple[int, int] = (-45,  0)
KING_SEMIOPEN_FILE_PENALTY: tuple[int, int] = (-20,  0)
PAWN_SHIELD_BONUS:          tuple[int, int] = ( 10,  0)

# Bishop pair ── larger bonus in EG (open positions suit bishops)
BISHOP_PAIR_BONUS: tuple[int, int] = (30, 50)

# Mobility ── per reachable square (pseudo-legal), (mg, eg)
MOBILITY_BONUS: dict[int, tuple[int, int]] = {
    chess.KNIGHT: (4, 3),
    chess.BISHOP: (3, 3),
    chess.ROOK:   (2, 3),
    chess.QUEEN:  (1, 1),
}
TRAPPED_PIECE_PENALTY: tuple[int, int] = (-50, -50)

MATE_SCORE: int = 32000

# ═══════════════════════════════════════════════════════════════════════════════
# 2 — PST builder
# ═══════════════════════════════════════════════════════════════════════════════

# (file, rank) coords of d4, e4, d5, e5
_CENTER_COORDS: tuple[tuple[int, int], ...] = ((3, 3), (4, 3), (3, 4), (4, 4))
_MAX_CENTER_DIST: int = 3  # max Chebyshev distance to nearest centre square


def _center_dist(sq: int) -> int:
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return min(max(abs(f - cf), abs(r - cr)) for cf, cr in _CENTER_COORDS)


def build_pst(
    center_bonus: float,
    advancement_bonus: float,
    edge_penalty: float,
    offsets: dict[int, float] | None = None,
) -> dict[int, int]:
    """Build a PST (a1=0 indexing, White's perspective) from geometric rules.

    center_bonus      — added per step closer to the centre (Chebyshev)
    advancement_bonus — added per rank (rank 0 = back rank, rank 7 = promo rank)
    edge_penalty      — subtracted when the square is on file a/h or rank 1/8
    offsets           — per-square manual adjustments applied after the formula
    """
    pst: dict[int, int] = {}
    for sq in chess.SQUARES:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        on_edge = f in (0, 7) or r in (0, 7)
        val = (
            center_bonus * (_MAX_CENTER_DIST - _center_dist(sq))
            + advancement_bonus * r
            - (edge_penalty if on_edge else 0.0)
        )
        pst[sq] = round(val)
    if offsets:
        for sq, delta in offsets.items():
            pst[sq] += round(delta)
    return pst


def _mirror(sq: int) -> int:
    """Flip rank axis so Black pieces use the same table as White (a1↔a8)."""
    return sq ^ 56


# ── Per-piece PST parameters ─────────────────────────────────────────────────

_KNIGHT_OFFSETS: dict[int, float] = {
    chess.D4: 8, chess.E4: 8, chess.D5: 8, chess.E5: 8,  # ideal outposts
    chess.C3: 4, chess.F3: 4, chess.C6: 4, chess.F6: 4,
}

# King MG: reward castled positions, penalise centre exposure
_KING_MG_OFFSETS: dict[int, float] = {
    chess.G1:  20, chess.C1: 15,   # castled
    chess.A1:   8, chess.H1:  8,
    chess.B1:   5, chess.F1:  3,
    chess.D1: -10, chess.E1: -12,  # uncastled centre files
}

# fmt: off
PST_MG: dict[int, dict[int, int]] = {
    chess.PAWN:   build_pst( 3,  7,  5),
    chess.KNIGHT: build_pst( 9,  0, 20, _KNIGHT_OFFSETS),
    chess.BISHOP: build_pst( 4,  1, 12),
    chess.ROOK:   build_pst( 1,  2,  3),
    chess.QUEEN:  build_pst( 3, -2, 10),  # negative advancement = deter early queen
    chess.KING:   build_pst(-6, -3, -10, _KING_MG_OFFSETS),  # neg params invert geometry
}
PST_EG: dict[int, dict[int, int]] = {
    chess.PAWN:   build_pst( 1, 12,  8),  # advancement dominates in EG
    chess.KNIGHT: build_pst( 9,  0, 20, _KNIGHT_OFFSETS),
    chess.BISHOP: build_pst( 5,  1, 12),
    chess.ROOK:   build_pst( 1,  3,  3),
    chess.QUEEN:  build_pst( 5,  2, 10),
    chess.KING:   build_pst(12,  2,  8),  # king centralises in EG
}
# fmt: on

# ═══════════════════════════════════════════════════════════════════════════════
# 3 — Phase calculator
# ═══════════════════════════════════════════════════════════════════════════════

def get_game_phase(board: chess.Board) -> int:
    """Return phase in [0, MAX_PHASE]. MAX_PHASE = full midgame, 0 = pure endgame."""
    phase = sum(
        (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))) * w
        for pt, w in PHASE_WEIGHTS.items()
    )
    return max(0, min(phase, MAX_PHASE))


# ═══════════════════════════════════════════════════════════════════════════════
# 4 — Feature functions  (all return (mg, eg) from White's perspective)
# ═══════════════════════════════════════════════════════════════════════════════

def material_balance(board: chess.Board) -> tuple[int, int]:
    mg = eg = 0
    for sq, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        mg += sign * MATERIAL_MG[piece.piece_type]
        eg += sign * MATERIAL_EG[piece.piece_type]
    return mg, eg


def pst_score(board: chess.Board) -> tuple[int, int]:
    mg = eg = 0
    for sq, piece in board.piece_map().items():
        idx = sq if piece.color == chess.WHITE else _mirror(sq)
        sign = 1 if piece.color == chess.WHITE else -1
        mg += sign * PST_MG[piece.piece_type][idx]
        eg += sign * PST_EG[piece.piece_type][idx]
    return mg, eg


def pawn_structure(board: chess.Board) -> tuple[int, int]:
    mg = eg = 0

    # Group pawn ranks by file for O(1) file queries
    w_by_file: list[list[int]] = [[] for _ in range(8)]
    b_by_file: list[list[int]] = [[] for _ in range(8)]
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        w_by_file[chess.square_file(sq)].append(chess.square_rank(sq))
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        b_by_file[chess.square_file(sq)].append(chess.square_rank(sq))

    for f in range(8):
        w_ranks = w_by_file[f]
        b_ranks = b_by_file[f]
        adj  = [af for af in (f - 1, f + 1) if 0 <= af <= 7]
        span = [af for af in (f - 1, f,     f + 1) if 0 <= af <= 7]

        # ── Doubled pawns ────────────────────────────────────────────────────
        if len(w_ranks) > 1:
            mg += DOUBLED_PAWN_PENALTY[0] * (len(w_ranks) - 1)
            eg += DOUBLED_PAWN_PENALTY[1] * (len(w_ranks) - 1)
        if len(b_ranks) > 1:
            mg -= DOUBLED_PAWN_PENALTY[0] * (len(b_ranks) - 1)
            eg -= DOUBLED_PAWN_PENALTY[1] * (len(b_ranks) - 1)

        # ── Isolated pawns ───────────────────────────────────────────────────
        if w_ranks and not any(w_by_file[af] for af in adj):
            mg += ISOLATED_PAWN_PENALTY[0] * len(w_ranks)
            eg += ISOLATED_PAWN_PENALTY[1] * len(w_ranks)
        if b_ranks and not any(b_by_file[af] for af in adj):
            mg -= ISOLATED_PAWN_PENALTY[0] * len(b_ranks)
            eg -= ISOLATED_PAWN_PENALTY[1] * len(b_ranks)

        # ── Passed pawns ─────────────────────────────────────────────────────
        # White: no black pawn at rank >= r on same/adjacent files
        for r in w_ranks:
            if not any(br >= r for bf in span for br in b_by_file[bf]):
                mg += PASSED_PAWN_BONUS_MG[r]
                eg += PASSED_PAWN_BONUS_EG[r]

        # Black: no white pawn at rank <= r on same/adjacent files.
        # Normalise rank so index 7 = almost promoting for Black.
        for r in b_ranks:
            if not any(wr <= r for bf in span for wr in w_by_file[bf]):
                black_adv = 7 - r
                mg -= PASSED_PAWN_BONUS_MG[black_adv]
                eg -= PASSED_PAWN_BONUS_EG[black_adv]

    return mg, eg


def king_safety(board: chess.Board) -> tuple[int, int]:
    mg = 0
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        king_sq = board.king(color)
        if king_sq is None:
            continue
        kf = chess.square_file(king_sq)
        kr = chess.square_rank(king_sq)

        # ── Pawn shield (only when king is on back two ranks) ─────────────────
        on_back_two = (color == chess.WHITE and kr <= 1) or (
            color == chess.BLACK and kr >= 6
        )
        if on_back_two:
            shield_r = kr + (1 if color == chess.WHITE else -1)
            for df in (-1, 0, 1):
                sf = kf + df
                if 0 <= sf <= 7:
                    p = board.piece_at(chess.square(sf, shield_r))
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        mg += sign * PAWN_SHIELD_BONUS[0]

        # ── Open / semi-open files near king ──────────────────────────────────
        own_pawns   = board.pieces(chess.PAWN, color)
        enemy_pawns = board.pieces(chess.PAWN, not color)
        for df in (-1, 0, 1):
            f = kf + df
            if not 0 <= f <= 7:
                continue
            file_bb   = chess.BB_FILES[f]
            has_own   = bool(own_pawns   & file_bb)
            has_enemy = bool(enemy_pawns & file_bb)
            if not has_own and not has_enemy:
                mg += sign * KING_OPEN_FILE_PENALTY[0]
            elif not has_own:
                mg += sign * KING_SEMIOPEN_FILE_PENALTY[0]

    return mg, 0


def mobility_score(board: chess.Board) -> tuple[int, int]:
    mg = eg = 0
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        own_occ = board.occupied_co[color]
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            for sq in board.pieces(pt, color):
                reachable = len(board.attacks(sq) & ~own_occ)
                if reachable <= 1:
                    mg += sign * TRAPPED_PIECE_PENALTY[0]
                    eg += sign * TRAPPED_PIECE_PENALTY[1]
                else:
                    bmg, beg = MOBILITY_BONUS[pt]
                    mg += sign * bmg * reachable
                    eg += sign * beg * reachable
    return mg, eg


def rook_bonuses(board: chess.Board) -> tuple[int, int]:
    mg = eg = 0
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        own_pawns   = board.pieces(chess.PAWN, color)
        enemy_pawns = board.pieces(chess.PAWN, not color)
        seventh     = 6 if color == chess.WHITE else 1
        for sq in board.pieces(chess.ROOK, color):
            file_bb   = chess.BB_FILES[chess.square_file(sq)]
            has_own   = bool(own_pawns   & file_bb)
            has_enemy = bool(enemy_pawns & file_bb)
            if not has_own and not has_enemy:
                mg += sign * ROOK_OPEN_FILE_BONUS[0]
                eg += sign * ROOK_OPEN_FILE_BONUS[1]
            elif not has_own:
                mg += sign * ROOK_SEMIOPEN_FILE_BONUS[0]
                eg += sign * ROOK_SEMIOPEN_FILE_BONUS[1]
            if chess.square_rank(sq) == seventh:
                mg += sign * ROOK_SEVENTH_RANK_BONUS[0]
                eg += sign * ROOK_SEVENTH_RANK_BONUS[1]
    return mg, eg


def bishop_pair(board: chess.Board) -> tuple[int, int]:
    mg = eg = 0
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        bishops = board.pieces(chess.BISHOP, color)
        if len(bishops) >= 2:
            has_light = bool(bishops & chess.BB_LIGHT_SQUARES)
            has_dark  = bool(bishops & chess.BB_DARK_SQUARES)
            if has_light and has_dark:
                mg += sign * BISHOP_PAIR_BONUS[0]
                eg += sign * BISHOP_PAIR_BONUS[1]
    return mg, eg


# ═══════════════════════════════════════════════════════════════════════════════
# 5 — Tapered combiner
# ═══════════════════════════════════════════════════════════════════════════════

def taper(mg: int, eg: int, phase: int) -> int:
    return (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE


# ═══════════════════════════════════════════════════════════════════════════════
# 6 — Top-level evaluate / explain
# ═══════════════════════════════════════════════════════════════════════════════

_FEATURES: tuple[tuple[str, object], ...] = (
    ("material",    material_balance),
    ("pst",         pst_score),
    ("pawns",       pawn_structure),
    ("king_safety", king_safety),
    ("mobility",    mobility_score),
    ("rooks",       rook_bonuses),
    ("bishops",     bishop_pair),
)


def evaluate(board: chess.Board) -> int:
    """Return centipawns from the side-to-move's perspective."""
    phase    = get_game_phase(board)
    total_mg = total_eg = 0
    for _, fn in _FEATURES:
        mg, eg = fn(board)  # type: ignore[operator]
        total_mg += mg
        total_eg += eg
    score = taper(total_mg, total_eg, phase)
    score = max(-MATE_SCORE + 1, min(MATE_SCORE - 1, score))
    if board.turn == chess.BLACK:
        score = -score
    return score


def explain(board: chess.Board) -> dict[str, int]:
    """Return per-feature tapered scores (White's perspective) for debugging."""
    phase  = get_game_phase(board)
    result: dict[str, int] = {"phase": phase}
    for name, fn in _FEATURES:
        mg, eg = fn(board)  # type: ignore[operator]
        result[name] = taper(mg, eg, phase)
    return result
