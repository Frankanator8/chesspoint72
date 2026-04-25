"""Demo: advanced HCE feature modules on arbitrary FEN strings.

Usage
-----
    python scripts/demo_advanced_hce.py
    python scripts/demo_advanced_hce.py "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

The script demonstrates:
  1. Evaluating a single FEN with the full feature breakdown (all 15 features).
  2. Feeding 10 sequential positions through evaluate() to warm up IDAM and
     show how the Informational Dynamics module responds to evaluation momentum.
"""
from __future__ import annotations

import sys
import textwrap

import chess

from chesspoint72.hce.hce import evaluate, explain
from chesspoint72.hce.advanced_features import (
    ewpm, srcm, idam, otvm, lmdm, lscm, clcm, desm,
)

# ── A short game fragment (Ruy López, ~10 half-moves) used for the IDAM demo
_GAME_MOVES = [
    "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6",
    "b5a4", "g8f6", "e1g1", "f8e7",
]

_DIVIDER = "─" * 64


def _banner(title: str) -> None:
    print(f"\n{'═' * 64}")
    print(f"  {title}")
    print('═' * 64)


def _feature_table(breakdown: dict[str, int]) -> None:
    phase = breakdown.pop("phase", None)
    if phase is not None:
        print(f"  {'Game phase':<22} {phase:>6}  (0=endgame, 24=midgame)")
        print(_DIVIDER)

    classic     = ["material", "pst", "pawns", "king_safety", "mobility", "rooks", "bishops"]
    advanced    = ["ewpm", "srcm", "idam", "otvm", "lmdm", "lscm", "clcm", "desm"]
    label_map   = {
        "material":   "Material balance",
        "pst":        "Piece-square tables",
        "pawns":      "Pawn structure",
        "king_safety":"King safety",
        "mobility":   "Mobility",
        "rooks":      "Rook bonuses",
        "bishops":    "Bishop pair",
        "ewpm":       "EWPM  (Entropy/Precision)",
        "srcm":       "SRCM  (Spectral Coordination)",
        "idam":       "IDAM  (Infor. Dynamics)",
        "otvm":       "OTVM  (Option Value)",
        "lmdm":       "LMDM  (Liquidity/Depth)",
        "lscm":       "LSCM  (Lyapunov Stability)",
        "clcm":       "CLCM  (Chunking/Patterns)",
        "desm":       "DESM  (Structural Stress)",
    }

    print(f"  {'Feature':<30} {'Score (cp)':>10}")
    print(_DIVIDER)
    print("  Classic features")
    for key in classic:
        if key in breakdown:
            print(f"    {label_map[key]:<28} {breakdown[key]:>+7}")
    print()
    print("  Advanced features")
    for key in advanced:
        if key in breakdown:
            print(f"    {label_map[key]:<28} {breakdown[key]:>+7}")


def evaluate_fen(fen: str) -> None:
    board = chess.Board(fen)
    _banner(f"Position: {fen}")
    print(board)
    print()

    score = evaluate(board)
    side  = "White" if board.turn == chess.WHITE else "Black"
    print(f"  Total evaluation: {score:+d} cp  (from {side}'s perspective)\n")

    breakdown = explain(board)
    _feature_table(breakdown)
    print()


def idam_demo() -> None:
    _banner("IDAM demo — 10-move Ruy López fragment")
    print(textwrap.dedent("""\
      IDAM requires a history of evaluations to compute velocity / acceleration /
      jerk.  We play through 10 half-moves and call evaluate() at each step so
      the singleton accumulates scores in its deque.
    """))

    board = chess.Board()
    idam._history.clear()   # fresh slate for the demo

    for i, uci in enumerate(_GAME_MOVES, 1):
        move = chess.Move.from_uci(uci)
        board.push(move)
        score = evaluate(board)
        idam_mg, idam_eg = idam.calculate(board)
        print(f"  Move {i:2d} ({uci})  eval={score:+5d} cp  "
              f"IDAM mg={idam_mg:+4d}  eg={idam_eg:+4d}")

    print()


def individual_module_demo(fen: str) -> None:
    _banner("Individual module outputs")
    board = chess.Board(fen)
    modules = [
        ("EWPM", ewpm), ("SRCM", srcm), ("IDAM", idam), ("OTVM", otvm),
        ("LMDM", lmdm), ("LSCM", lscm), ("CLCM", clcm), ("DESM", desm),
    ]
    print(f"  {'Module':<8}  {'mg':>7}  {'eg':>7}")
    print(_DIVIDER)
    for name, mod in modules:
        mg, eg = mod.calculate(board)
        print(f"  {name:<8}  {mg:>+7}  {eg:>+7}")
    print()


if __name__ == "__main__":
    target_fen = sys.argv[1] if len(sys.argv) > 1 else chess.STARTING_FEN

    evaluate_fen(target_fen)

    # A richer middlegame position (Sicilian dragon, castled both sides)
    midgame_fen = (
        "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 0 9"
    )
    evaluate_fen(midgame_fen)

    idam_demo()

    individual_module_demo(midgame_fen)
