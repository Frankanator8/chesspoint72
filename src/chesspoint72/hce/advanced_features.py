"""Advanced HCE feature modules.

Eight theoretically-grounded evaluation components, each implemented as a
callable class with a ``calculate(board) -> tuple[int, int]`` method returning
(mg_score, eg_score) from White's perspective.

Module index
------------
EWPM  — Entropy-Weighted Precision Module
SRCM  — Spectral Resilience and Coordination Module
IDAM  — Informational Dynamics and Acceleration Module
OTVM  — Option-Theoretic Value Module
LMDM  — Liquidity and Market Depth Module
LSCM  — Lyapunov Stability and Chaos Module
CLCM  — Cognitive Load and Chunking Module
DESM  — Dynamic Elasticity and Structural Stress Module

Module-level singleton instances (``ewpm``, ``srcm``, … ``desm``) are exported
and registered directly in ``hce.py``'s ``_FEATURES`` tuple.
"""
from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

import chess

if TYPE_CHECKING:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normal_cdf(x: float) -> float:
    """Standard normal CDF via math.erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _mobility_counts(board: chess.Board, color: chess.Color) -> list[int]:
    """Return a list of pseudo-legal attack counts for every non-king, non-pawn piece."""
    own_occ = board.occupied_co[color]
    counts: list[int] = []
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, color):
            counts.append(len(board.attacks(sq) & ~own_occ))
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# 1. EWPM — Entropy-Weighted Precision Module
# ─────────────────────────────────────────────────────────────────────────────

class EWPM:
    """Entropy-Weighted Precision Module.

    Mathematical derivation
    -----------------------
    Treat each piece's pseudo-legal attack count as a raw logit ``xᵢ``.
    Convert to a probability distribution via Softmax with temperature T:

        p(mᵢ) = exp(xᵢ / T) / Σⱼ exp(xⱼ / T)

    Compute the Shannon entropy of that distribution:

        H(S) = -Σᵢ p(mᵢ) · log₂ p(mᵢ)

    The normalised Precision Index is:

        ρ = 1 - H(S) / log₂(N)      where N = number of mobile pieces

    ρ ∈ [0, 1].  A position with all pieces equally mobile (maximum entropy)
    gives ρ = 0; a position where one piece dominates gives ρ → 1.
    The centipawn bonus scales linearly with ρ.
    """

    _TEMPERATURE: float = 1.5
    _SCALE_MG: int = 25
    _SCALE_EG: int = 18

    def calculate(self, board: chess.Board) -> tuple[int, int]:
        mg = eg = 0
        for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
            counts = _mobility_counts(board, color)
            n = len(counts)
            if n < 2:
                continue
            try:
                import numpy as np
                x = np.array(counts, dtype=float) / self._TEMPERATURE
                x -= x.max()                          # numerical stability
                exp_x = np.exp(x)
                probs = exp_x / exp_x.sum()
                mask = probs > 0
                h = float(-np.sum(probs[mask] * np.log2(probs[mask])))
                h_max = math.log2(n)
                rho = 1.0 - h / h_max if h_max > 0 else 0.0
            except ImportError:
                # Fallback: coefficient of variation as precision proxy
                mean_ = sum(counts) / n
                if mean_ == 0:
                    continue
                variance = sum((c - mean_) ** 2 for c in counts) / n
                rho = min(1.0, math.sqrt(variance) / (mean_ + 1e-9))

            mg += sign * round(rho * self._SCALE_MG)
            eg += sign * round(rho * self._SCALE_EG)
        return mg, eg

    def __call__(self, board: chess.Board) -> tuple[int, int]:
        return self.calculate(board)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SRCM — Spectral Resilience and Coordination Module
# ─────────────────────────────────────────────────────────────────────────────

class SRCM:
    """Spectral Resilience and Coordination Module.

    Mathematical derivation
    -----------------------
    Model the piece network as an undirected graph G = (V, E) where:
      • V — squares occupied by own pieces (vertices)
      • E — pairs (i, j) where piece on i attacks square j or vice-versa

    Construct the Graph Laplacian:
        L = D - A
    where A is the adjacency matrix and D = diag(A · 1) is the degree matrix.

    The *Algebraic Connectivity* λ₂ (Fiedler value) is the second-smallest
    eigenvalue of L.  By the Cheeger inequality, higher λ₂ means the graph
    is harder to disconnect, which in chess terms corresponds to tighter
    piece coordination.

    Computed per side; White's advantage = λ₂(White) - λ₂(Black).
    Uses ``scipy.linalg.eigsh`` for efficiency; falls back to (0, 0) if
    scipy is unavailable.
    """

    _SCALE_MG: int = 12
    _SCALE_EG: int = 18

    def _algebraic_connectivity(self, board: chess.Board, color: chess.Color) -> float:
        try:
            import numpy as np
            from scipy.linalg import eigvalsh
        except ImportError:
            return 0.0

        squares = list(chess.scan_forward(board.occupied_co[color]))
        n = len(squares)
        if n < 2:
            return 0.0

        sq_idx = {sq: i for i, sq in enumerate(squares)}
        A = np.zeros((n, n), dtype=float)
        for sq in squares:
            attacks = board.attacks(sq)
            for target in attacks:
                if target in sq_idx:
                    i, j = sq_idx[sq], sq_idx[target]
                    A[i, j] = 1.0
                    A[j, i] = 1.0

        D = np.diag(A.sum(axis=1))
        L = D - A
        eigenvalues = eigvalsh(L)
        eigenvalues.sort()
        return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0

    def calculate(self, board: chess.Board) -> tuple[int, int]:
        lam_w = self._algebraic_connectivity(board, chess.WHITE)
        lam_b = self._algebraic_connectivity(board, chess.BLACK)
        diff = lam_w - lam_b
        mg = round(diff * self._SCALE_MG)
        eg = round(diff * self._SCALE_EG)
        return mg, eg

    def __call__(self, board: chess.Board) -> tuple[int, int]:
        return self.calculate(board)


# ─────────────────────────────────────────────────────────────────────────────
# 3. IDAM — Informational Dynamics and Acceleration Module
# ─────────────────────────────────────────────────────────────────────────────

class IDAM:
    """Informational Dynamics and Acceleration Module.

    Mathematical derivation
    -----------------------
    Treat the sequence of tapered evaluation scores as a 1-D trajectory in
    configuration space.  Define finite-difference kinematic quantities:

        Velocity:     vᵢ = eᵢ − eᵢ₋₁
        Acceleration: aᵢ = vᵢ − vᵢ₋₁
        Jerk:         jᵢ = aᵢ − aᵢ₋₁

    Mean absolute jerk over the window quantifies *evaluation turbulence*.
    A stable position (low |j̄|) receives a bonus; a turbulent position
    (high |j̄|) receives a penalty.  ``record(score)`` must be called from
    the outer ``evaluate()`` loop to populate the history deque.
    """

    _WINDOW: int = 8
    _JERK_SCALE_MG: int = 8
    _JERK_SCALE_EG: int = 5
    _CLAMP: int = 30

    def __init__(self) -> None:
        self._history: deque[float] = deque(maxlen=self._WINDOW)

    def record(self, score: float) -> None:
        """Feed the latest tapered score into the history buffer."""
        self._history.append(score)

    def calculate(self, board: chess.Board) -> tuple[int, int]:
        h = list(self._history)
        if len(h) < 4:
            return 0, 0

        velocities     = [h[i] - h[i - 1] for i in range(1, len(h))]
        accelerations  = [velocities[i] - velocities[i - 1] for i in range(1, len(velocities))]
        jerks          = [accelerations[i] - accelerations[i - 1] for i in range(1, len(accelerations))]

        mean_abs_jerk = sum(abs(j) for j in jerks) / max(1, len(jerks))
        # Normalise: 100 cp/move² of jerk is considered highly turbulent
        normalised = min(1.0, mean_abs_jerk / 200.0)
        # Stable (low jerk) = bonus; turbulent = penalty
        score_delta = round((0.5 - normalised) * 2 * self._CLAMP)
        mg = max(-self._CLAMP, min(self._CLAMP, round(score_delta * self._JERK_SCALE_MG / self._CLAMP)))
        eg = max(-self._CLAMP, min(self._CLAMP, round(score_delta * self._JERK_SCALE_EG / self._CLAMP)))
        return mg, eg

    def __call__(self, board: chess.Board) -> tuple[int, int]:
        return self.calculate(board)


# ─────────────────────────────────────────────────────────────────────────────
# 4. OTVM — Option-Theoretic Value Module
# ─────────────────────────────────────────────────────────────────────────────

class OTVM:
    """Option-Theoretic Value Module.

    Mathematical derivation
    -----------------------
    Apply the Black-Scholes framework to model the *option value* of strategic
    flexibility.  Parameters mapped to chess:

      S  = |material advantage in centipawns| + ε   (underlying asset price)
      T  = max(1, 50 − halfmove_clock) / 50          (normalised time-to-maturity)
      σ  = std-dev of per-piece mobility counts       (volatility proxy)
      r  = 0                                          (no risk-free rate analog)
      K  = 1                                          (near-zero strike)

    Black-Scholes call:
        d₁ = (ln(S/K) + (σ²/2)·T) / (σ·√T)
        d₂ = d₁ − σ·√T
        C  = S·N(d₁) − K·N(d₂)

    where N(·) is the standard normal CDF.  The option premium C is scaled to
    centipawns and given a sign reflecting which side holds the advantage.
    """

    _PREMIUM_SCALE_MG: int = 40
    _PREMIUM_SCALE_EG: int = 30
    _EPSILON: float = 1.0   # prevents log(0)

    def _bs_call(self, S: float, K: float, T: float, sigma: float) -> float:
        if sigma <= 0 or T <= 0:
            return max(0.0, S - K)
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return S * _normal_cdf(d1) - K * _normal_cdf(d2)

    def _side_premium(self, board: chess.Board, color: chess.Color) -> float:
        counts = _mobility_counts(board, color)
        if not counts:
            return 0.0
        n = len(counts)
        mean_ = sum(counts) / n
        variance = sum((c - mean_) ** 2 for c in counts) / n
        sigma = math.sqrt(variance) / max(1.0, mean_)   # coefficient of variation

        material = sum(
            len(board.pieces(pt, color)) * val
            for pt, val in {
                chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                chess.ROOK: 500, chess.QUEEN: 900,
            }.items()
        )
        S = float(material) + self._EPSILON
        T = max(1, 50 - board.halfmove_clock) / 50.0
        K = 1.0
        return self._bs_call(S, K, T, max(sigma, 0.01))

    def calculate(self, board: chess.Board) -> tuple[int, int]:
        premium_w = self._side_premium(board, chess.WHITE)
        premium_b = self._side_premium(board, chess.BLACK)
        raw = premium_w - premium_b

        # Normalise: a raw delta of 1000 is mapped to the full scale
        norm = raw / 1000.0
        mg = max(-self._PREMIUM_SCALE_MG, min(self._PREMIUM_SCALE_MG,
                 round(norm * self._PREMIUM_SCALE_MG)))
        eg = max(-self._PREMIUM_SCALE_EG, min(self._PREMIUM_SCALE_EG,
                 round(norm * self._PREMIUM_SCALE_EG)))
        return mg, eg

    def __call__(self, board: chess.Board) -> tuple[int, int]:
        return self.calculate(board)


# ─────────────────────────────────────────────────────────────────────────────
# 5. LMDM — Liquidity and Market Depth Module
# ─────────────────────────────────────────────────────────────────────────────

class LMDM:
    """Liquidity and Market Depth Module.

    Mathematical derivation
    -----------------------
    Map piece mobility to *market liquidity* — the ease with which a resource
    can be re-deployed.  Sort per-piece mobility counts descending:

        bid  = m[0]   (most mobile piece, primary market depth)
        ask  = m[1]   (second-most mobile piece, secondary depth)

    The Bid-Ask Spread:
        spread = (bid − ask) / max(1, bid)   ∈ [0, 1]

    A narrow spread (bid ≈ ask) means pieces are uniformly mobile — high
    liquidity.  A wide spread means one piece dominates, others are
    illiquid.  Liquidity score = (1 − spread), scaled to centipawns.
    """

    _SCALE_MG: int = 20
    _SCALE_EG: int = 15

    def _liquidity(self, board: chess.Board, color: chess.Color) -> float:
        counts = sorted(_mobility_counts(board, color), reverse=True)
        if len(counts) < 2:
            return 0.0
        bid, ask = counts[0], counts[1]
        spread = (bid - ask) / max(1, bid)
        return 1.0 - spread

    def calculate(self, board: chess.Board) -> tuple[int, int]:
        liq_w = self._liquidity(board, chess.WHITE)
        liq_b = self._liquidity(board, chess.BLACK)
        diff = liq_w - liq_b
        mg = round(diff * self._SCALE_MG)
        eg = round(diff * self._SCALE_EG)
        return mg, eg

    def __call__(self, board: chess.Board) -> tuple[int, int]:
        return self.calculate(board)


# ─────────────────────────────────────────────────────────────────────────────
# 6. LSCM — Lyapunov Stability and Chaos Module
# ─────────────────────────────────────────────────────────────────────────────

class LSCM:
    """Lyapunov Stability and Chaos Module.

    Mathematical derivation
    -----------------------
    A system is Lyapunov-stable if small perturbations do not grow
    exponentially.  We proxy the *Lyapunov exponent* λ̂ by measuring how
    sensitive the piece-network connectivity is to the removal of each piece:

        B      = Σᵢ |attacks(squareᵢ) & ~own_occ|     (baseline connectivity)
        δBᵢ    = contribution of piece i to B           (its attack count)
        λ̂      = mean(|δBᵢ| / max(1, B))   over all pieces

    High λ̂ → one piece dominates the network → chaotic/fragile position.
    Low λ̂  → load is distributed evenly      → stable/resilient position.

    Stability score = (1 − clamp(λ̂, 0, 1)), scaled to centipawns.
    """

    _SCALE_MG: int = 15
    _SCALE_EG: int = 10

    def _lyapunov(self, board: chess.Board, color: chess.Color) -> float:
        own_occ = board.occupied_co[color]
        contributions: list[int] = []
        baseline = 0
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            for sq in board.pieces(pt, color):
                c = len(board.attacks(sq) & ~own_occ)
                contributions.append(c)
                baseline += c
        if baseline == 0 or not contributions:
            return 0.0
        lam = sum(abs(c) / baseline for c in contributions) / len(contributions)
        return min(1.0, lam)

    def calculate(self, board: chess.Board) -> tuple[int, int]:
        lam_w = self._lyapunov(board, chess.WHITE)
        lam_b = self._lyapunov(board, chess.BLACK)
        # Stability = 1 - lambda; White advantage = (stability_W - stability_B)
        diff = (1.0 - lam_w) - (1.0 - lam_b)   # = lam_b - lam_w
        mg = round(diff * self._SCALE_MG)
        eg = round(diff * self._SCALE_EG)
        return mg, eg

    def __call__(self, board: chess.Board) -> tuple[int, int]:
        return self.calculate(board)


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLCM — Cognitive Load and Chunking Module
# ─────────────────────────────────────────────────────────────────────────────

class CLCM:
    """Cognitive Load and Chunking Module.

    Mathematical derivation
    -----------------------
    Expert players perceive positions in recognisable *chunks* (known
    patterns) rather than individual pieces.  We identify N_MAX = 6
    structural patterns via bitboard masks.

    Chunk Density:
        ρ_chunk = matched_patterns / N_MAX   ∈ [0, 1]

    The centipawn score is:
        chunk_score = round((ρ_chunk − 0.5) × SCALE)

    i.e., positions with more than half the patterns active get a bonus;
    positions below half the patterns get a penalty.

    Patterns checked (per colour)
    ──────────────────────────────
    1. Fianchetto-g — bishop on g2 (g7) with pawns on f2 + h2 (f7 + h7)
    2. Fianchetto-b — bishop on b2 (b7) with pawns on a2 + c2 (a7 + c7)
    3. File battery  — rook and queen share a file
    4. Diagonal battery — bishop and queen on a shared diagonal
    5. Connected rooks  — both rooks on the same rank, no piece between them
    6. Knight outpost   — knight on an advanced central square with pawn support
    """

    _N_MAX: int = 6
    _SCALE_MG: int = 18
    _SCALE_EG: int = 12

    # Outpost squares (d5/e5/c5/f5 for White, d4/e4/c4/f4 for Black)
    _W_OUTPOSTS = chess.BB_D5 | chess.BB_E5 | chess.BB_C5 | chess.BB_F5
    _B_OUTPOSTS = chess.BB_D4 | chess.BB_E4 | chess.BB_C4 | chess.BB_F4

    def _count_patterns(self, board: chess.Board, color: chess.Color) -> int:
        count = 0
        bishops  = board.pieces(chess.BISHOP, color)
        pawns    = board.pieces(chess.PAWN, color)
        rooks    = board.pieces(chess.ROOK, color)
        queens   = board.pieces(chess.QUEEN, color)
        knights  = board.pieces(chess.KNIGHT, color)

        if color == chess.WHITE:
            # 1. Fianchetto-g: bishop on g2, pawns on f2 and h2
            if (int(bishops) & chess.BB_G2) and (int(pawns) & chess.BB_F2) and (int(pawns) & chess.BB_H2):
                count += 1
            # 2. Fianchetto-b: bishop on b2, pawns on a2 and c2
            if (int(bishops) & chess.BB_B2) and (int(pawns) & chess.BB_A2) and (int(pawns) & chess.BB_C2):
                count += 1
            # 6. Knight outpost with pawn support
            outpost_knights = chess.SquareSet(int(knights) & self._W_OUTPOSTS)
            if outpost_knights:
                for sq in outpost_knights:
                    supporters = board.attackers(chess.WHITE, sq)
                    if int(supporters) & int(pawns):
                        count += 1
                        break
        else:
            # 1. Fianchetto-g: bishop on g7, pawns on f7 and h7
            if (int(bishops) & chess.BB_G7) and (int(pawns) & chess.BB_F7) and (int(pawns) & chess.BB_H7):
                count += 1
            # 2. Fianchetto-b: bishop on b7, pawns on a7 and c7
            if (int(bishops) & chess.BB_B7) and (int(pawns) & chess.BB_A7) and (int(pawns) & chess.BB_C7):
                count += 1
            # 6. Knight outpost with pawn support
            outpost_knights = chess.SquareSet(int(knights) & self._B_OUTPOSTS)
            if outpost_knights:
                for sq in outpost_knights:
                    supporters = board.attackers(chess.BLACK, sq)
                    if int(supporters) & int(pawns):
                        count += 1
                        break

        # 3. File battery: rook and queen on the same file
        if rooks and queens:
            for f in range(8):
                file_bb = chess.BB_FILES[f]
                if (int(rooks) & file_bb) and (int(queens) & file_bb):
                    count += 1
                    break

        # 4. Diagonal battery: bishop and queen share a diagonal
        if bishops and queens:
            for sq_b in bishops:
                b_diags = int(board.attacks(sq_b))
                for sq_q in queens:
                    if chess.BB_SQUARES[sq_q] & b_diags:
                        count += 1
                        break
                else:
                    continue
                break

        # 5. Connected rooks: both rooks on the same rank, no piece between
        rook_list = list(rooks)
        if len(rook_list) >= 2:
            r1, r2 = rook_list[0], rook_list[1]
            if chess.square_rank(r1) == chess.square_rank(r2):
                lo, hi = min(r1, r2), max(r1, r2)
                between_bb = 0
                for sq in range(lo + 1, hi):
                    between_bb |= chess.BB_SQUARES[sq]
                if not (board.occupied & between_bb):
                    count += 1

        return count

    def calculate(self, board: chess.Board) -> tuple[int, int]:
        mg = eg = 0
        for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
            matched = self._count_patterns(board, color)
            density = matched / self._N_MAX
            score = round((density - 0.5) * 2 * 1.0)  # -1 to +1
            mg += sign * round(score * self._SCALE_MG)
            eg += sign * round(score * self._SCALE_EG)
        return mg, eg

    def __call__(self, board: chess.Board) -> tuple[int, int]:
        return self.calculate(board)


# ─────────────────────────────────────────────────────────────────────────────
# 8. DESM — Dynamic Elasticity and Structural Stress Module
# ─────────────────────────────────────────────────────────────────────────────

class DESM:
    """Dynamic Elasticity and Structural Stress Module.

    Mathematical derivation
    -----------------------
    Model the pawn structure as a set of spring-mass systems.  For each pawn
    at square sq:

      • Stiffness  k  = number of friendly pawns that defend it
                        (diagonally behind), minimum 0.
      • Ideal file    = mean file index of all own pawns (centroid).
      • Displacement  Δx = |file(sq) − ideal_file|
      • Potential energy: U = ½ · (k + 1) · Δx²

    The (k + 1) term ensures isolated pawns (k = 0) still contribute stress.
    Total structural stress:

        U_total = Σᵢ Uᵢ   (summed over all own pawns)

    High U_total → deformed / doubled / isolated structure → penalty.
    U_total is clamped and mapped to centipawns with:

        stress_penalty = −round(U_total · SCALE)
    """

    _STRESS_SCALE_MG: float = 0.8
    _STRESS_SCALE_EG: float = 1.2
    _CAP: int = 60

    def _structural_stress(self, board: chess.Board, color: chess.Color) -> float:
        squares = list(board.pieces(chess.PAWN, color))
        if not squares:
            return 0.0

        files = [chess.square_file(sq) for sq in squares]
        ideal_file = sum(files) / len(files)

        total_u = 0.0
        for sq in squares:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            dx = abs(f - ideal_file)

            # Count defending pawns (diagonally behind this pawn)
            defend_rank = r - 1 if color == chess.WHITE else r + 1
            k = 0
            if 0 <= defend_rank <= 7:
                for df in (-1, 1):
                    dsq_file = f + df
                    if 0 <= dsq_file <= 7:
                        dsq = chess.square(dsq_file, defend_rank)
                        p = board.piece_at(dsq)
                        if p and p.piece_type == chess.PAWN and p.color == color:
                            k += 1

            total_u += 0.5 * (k + 1) * dx ** 2

        return total_u

    def calculate(self, board: chess.Board) -> tuple[int, int]:
        stress_w = self._structural_stress(board, chess.WHITE)
        stress_b = self._structural_stress(board, chess.BLACK)
        # White advantage: Black has more stress than White
        diff = stress_b - stress_w
        mg_raw = round(diff * self._STRESS_SCALE_MG)
        eg_raw = round(diff * self._STRESS_SCALE_EG)
        mg = max(-self._CAP, min(self._CAP, mg_raw))
        eg = max(-self._CAP, min(self._CAP, eg_raw))
        return mg, eg

    def __call__(self, board: chess.Board) -> tuple[int, int]:
        return self.calculate(board)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons — import these into hce.py
# ─────────────────────────────────────────────────────────────────────────────

ewpm = EWPM()
srcm = SRCM()
idam = IDAM()
otvm = OTVM()
lmdm = LMDM()
lscm = LSCM()
clcm = CLCM()
desm = DESM()
