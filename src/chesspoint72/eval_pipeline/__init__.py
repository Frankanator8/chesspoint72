"""Chesspoint72 evaluation pipeline.

Implements the 9-stage testing framework from EVAL_PIPELINE.md:

    Stage 0  — Perft validation (move-generation correctness gate)
    Stage 1  — Hard disqualifiers (illegal moves, crashes, timeouts, determinism)
    Stage 2  — Search instrumentation (already wired into NegamaxSearch)
    Stage 3  — True baseline establishment (stub vs Baseline B sanity check)
    Stage 4  — Module-type benchmarks (NPS, TT hit rate, eval accuracy)
    Stage 5  — Isolated A/B tests (corrected LOS thresholds + 3 new tests)
    Stage 6  — Factorial interaction matrix (marginal contribution analysis)
    Stage 7  — Regime stress tests (phase, tactical, EPD suites)
    Stage 8  — Tournament backtest (round-robin Elo ladder)
    Stage 9  — Final module scoring (corrected composite formula)

Entry point:
    python -m chesspoint72.eval_pipeline.runner [--stage N] [--games N]
"""
