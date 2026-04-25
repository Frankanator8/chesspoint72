"""Hand-Crafted Evaluation (HCE) package.

The primary entry point is ``evaluate(board)``, which returns a tapered
centipawn score from the side-to-move's perspective.  The legacy helpers
``material_score`` and ``pst_score`` are re-exported for callers that need
individual components.
"""
from .hce import evaluate, explain, get_game_phase
from .material import (
    BISHOP_PAIR_BONUS,
    KNIGHT_PAIR_PENALTY,
    PIECE_VALUES,
    ROOK_PAIR_PENALTY,
    material_score,
)
from .pst import pst_score
from .advanced_features import (
    EWPM, ewpm,
    SRCM, srcm,
    IDAM, idam,
    OTVM, otvm,
    LMDM, lmdm,
    LSCM, lscm,
    CLCM, clcm,
    DESM, desm,
)

__all__ = [
    # unified evaluator
    "evaluate",
    "explain",
    "get_game_phase",
    # legacy standalone helpers
    "material_score",
    "pst_score",
    "PIECE_VALUES",
    "BISHOP_PAIR_BONUS",
    "KNIGHT_PAIR_PENALTY",
    "ROOK_PAIR_PENALTY",
    # advanced feature classes
    "EWPM", "SRCM", "IDAM", "OTVM", "LMDM", "LSCM", "CLCM", "DESM",
    # advanced feature singletons
    "ewpm", "srcm", "idam", "otvm", "lmdm", "lscm", "clcm", "desm",
]
# Entropy-Weighted Precision Module (EWPM)Logic: Compute Shannon Entropy $H(S) = -\sum p(m_i) \log_2 p(m_i)$.Implementation: Convert raw centipawn evaluations into a probability distribution using a Softmax function with a temperature parameter $T$. Return the Precision Index $\rho$.2. Spectral Resilience and Coordination Module (SRCM)Logic: Represent the board as a graph $G=(V, E)$ where $V$ are occupied squares and $E$ are tactical interactions.Implementation: Construct the Graph Laplacian $L = D - A$. Use scipy.linalg.eigsh to find the Algebraic Connectivity ($\lambda_2$). Higher $\lambda_2$ should correlate with a "Coordination Bonus."3. Informational Dynamics and Acceleration Module (IDAM)Logic: Treat evaluation as a position in configuration space.Implementation: Use a deque to store the last $N$ evaluations. Calculate finite differences for Informational Velocity ($v_i$), Acceleration ($a_i$), and Jerk ($j_i$).4. Option-Theoretic Value Module (OTVM)Logic: Apply the Black-Scholes framework to strategic flexibility.Implementation: Calculate an "Option Premium" based on the "Time to Maturity" (moves until the 50-move rule or endgame) and "Volatility" (standard deviation of evaluations in the local search tree).5. Liquidity and Market Depth Module (LMDM)Logic: Assess the "re-deployability" of pieces.Implementation: Map square importance to piece mobility. Compute the Bid-Ask Spread as the delta between the primary and secondary move evaluations.6. Lyapunov Stability and Chaos Module (LSCM)Logic: Quantify the sensitivity of the evaluation to move-order perturbations.Implementation: Implement a function that simulates "noise" in the Principal Variation (PV) and measures the divergence rate $\lambda$.7. Cognitive Load and Chunking Module (CLCM)Logic: Model the difficulty of finding the move for a human.Implementation: Use bitboard masks to identify common tactical "chunks" (e.g., fianchetto structures, battery formations). Penalize positions with low "Chunk Density."8. Dynamic Elasticity and Structural Stress Module (DESM)Logic: Model pawn structures as mechanical systems.Implementation: Calculate the "Potential Energy" $U = \frac{1}{2}k\Delta x^2$ for pawn chains, where $\Delta x$ represents the displacement from ideal squares and $k$ represents the "stiffness" of the structural support.