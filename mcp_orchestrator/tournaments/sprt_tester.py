"""Sequential Probability Ratio Test between two UCI engines.

We drive the engines through python-chess's `SimpleEngine` (so we get
proper clock management via `go wtime/btime/winc/binc`), play a stream of
games with alternating colors, and after each game recompute the GSPRT
log-likelihood ratio over the running W/D/L tally. We stop as soon as the
LLR crosses either boundary or we hit `max_games`.

The LLR formula is the generalised SPRT from Michel Van den Bergh
(fishtest), which works on the trinomial (W, D, L) model. It is the
standard choice for chess engine A/B testing.

Hypotheses:
    H0: Elo difference = elo0    (default 0)
    H1: Elo difference = elo1    (default 10)

Boundaries:
    upper = log((1 - beta) / alpha)
    lower = log(beta / (1 - alpha))
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import chess
import chess.engine


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #

_TC_RE = re.compile(r"^\s*([\d.]+)\s*(?:\+\s*([\d.]+))?\s*$")


@dataclass
class TimeControl:
    base: float
    inc: float = 0.0

    @classmethod
    def parse(cls, s: str) -> "TimeControl":
        m = _TC_RE.match(s or "")
        if not m:
            raise ValueError(
                f"invalid time_control {s!r}; expected 'base[+inc]' seconds, e.g. '10+0.1'"
            )
        return cls(base=float(m.group(1)), inc=float(m.group(2) or 0.0))


# --------------------------------------------------------------------------- #
# SPRT math
# --------------------------------------------------------------------------- #


def _elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def sprt_llr(W: int, D: int, L: int, elo0: float, elo1: float) -> float:
    """Generalized SPRT log-likelihood ratio over the trinomial W/D/L."""
    N = W + D + L
    if N < 2:
        return 0.0
    mean = (W + 0.5 * D) / N
    var = (W + 0.25 * D) / N - mean * mean
    if var <= 1e-9:
        return 0.0
    s0 = _elo_to_score(elo0)
    s1 = _elo_to_score(elo1)
    return N * (s1 - s0) * (2 * mean - s0 - s1) / (2 * var)


def elo_point_estimate(W: int, D: int, L: int) -> Optional[float]:
    N = W + D + L
    if N == 0:
        return None
    score = (W + 0.5 * D) / N
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / score - 1.0)


# --------------------------------------------------------------------------- #
# Game loop
# --------------------------------------------------------------------------- #


def _play_game(
    white: chess.engine.SimpleEngine,
    black: chess.engine.SimpleEngine,
    tc: TimeControl,
    starting_fen: str,
    move_cap: int = 400,
) -> str:
    """Play one game, return "1-0" / "0-1" / "1/2-1/2"."""
    board = chess.Board(starting_fen) if starting_fen else chess.Board()
    wt = bt = tc.base
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < move_cap:
        engine = white if board.turn == chess.WHITE else black
        limit = chess.engine.Limit(
            white_clock=max(wt, 0.01),
            black_clock=max(bt, 0.01),
            white_inc=tc.inc,
            black_inc=tc.inc,
        )
        t0 = time.monotonic()
        try:
            res = engine.play(board, limit)
        except chess.engine.EngineError:
            # Crash / illegal move → side-to-move loses.
            return "0-1" if board.turn == chess.WHITE else "1-0"
        elapsed = time.monotonic() - t0

        if board.turn == chess.WHITE:
            wt = wt - elapsed + tc.inc
            if wt < 0:
                return "0-1"
        else:
            bt = bt - elapsed + tc.inc
            if bt < 0:
                return "1-0"

        if res.move is None:
            break
        board.push(res.move)
        plies += 1
    return board.result(claim_draw=True)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


@dataclass
class SPRTResult:
    decision: str  # "H1_accepted" | "H0_accepted" | "inconclusive"
    reason: str
    W: int
    D: int
    L: int
    games: int
    llr: float
    elo_estimate: Optional[float]
    elapsed_s: float
    bounds: dict
    time_control: dict
    openings_used: int


# Balanced two-move openings. Keep the set small and deterministic so the
# engine is forced into varied positions without needing an external book.
DEFAULT_OPENINGS: tuple[str, ...] = (
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",   # 1.e4 e5
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",   # 1.e4 c5
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",   # 1.e4 d5
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",   # 1.d4 d5
    "rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",   # 1.d4 e6
    "rnbqkbnr/pppppp1p/6p1/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",   # 1.d4 g6
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",   # 1.d4 Nf6
    "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",     # 1.Nf3
)


def run_sprt(
    engine_a_path: str,
    engine_b_path: str,
    time_control: str,
    elo0: float = 0.0,
    elo1: float = 10.0,
    alpha: float = 0.05,
    beta: float = 0.05,
    max_games: int = 1000,
    openings: Optional[list[str]] = None,
    on_game: Optional[Callable[[dict], None]] = None,
) -> SPRTResult:
    tc = TimeControl.parse(time_control)
    upper = math.log((1 - beta) / alpha)
    lower = math.log(beta / (1 - alpha))
    book = list(openings) if openings else list(DEFAULT_OPENINGS)

    W = D = L = 0
    t_start = time.monotonic()
    decision = "inconclusive"
    reason = f"reached max_games={max_games}"

    eng_a = chess.engine.SimpleEngine.popen_uci(engine_a_path)
    try:
        eng_b = chess.engine.SimpleEngine.popen_uci(engine_b_path)
    except Exception:
        eng_a.quit()
        raise

    try:
        for g in range(max_games):
            fen = book[g % len(book)]
            # Alternate colors every game for fairness.
            a_is_white = (g % 2 == 0)
            white, black = (eng_a, eng_b) if a_is_white else (eng_b, eng_a)

            result_str = _play_game(white, black, tc, fen)
            if result_str == "1-0":
                if a_is_white:
                    W += 1
                else:
                    L += 1
            elif result_str == "0-1":
                if a_is_white:
                    L += 1
                else:
                    W += 1
            else:
                D += 1

            llr = sprt_llr(W, D, L, elo0, elo1)
            if on_game:
                try:
                    on_game({
                        "g": g + 1, "W": W, "D": D, "L": L,
                        "llr": llr, "result": result_str,
                        "a_was_white": a_is_white,
                    })
                except Exception:
                    pass

            if llr >= upper:
                decision = "H1_accepted"
                reason = f"LLR {llr:.3f} ≥ upper {upper:.3f} (engine A stronger than elo1)"
                break
            if llr <= lower:
                decision = "H0_accepted"
                reason = f"LLR {llr:.3f} ≤ lower {lower:.3f} (engine A not stronger than elo0)"
                break
    finally:
        for e in (eng_a, eng_b):
            try:
                e.quit()
            except Exception:
                try:
                    e.close()
                except Exception:
                    pass

    games = W + D + L
    return SPRTResult(
        decision=decision,
        reason=reason,
        W=W, D=D, L=L,
        games=games,
        llr=sprt_llr(W, D, L, elo0, elo1),
        elo_estimate=elo_point_estimate(W, D, L),
        elapsed_s=round(time.monotonic() - t_start, 3),
        bounds={
            "elo0": elo0, "elo1": elo1,
            "alpha": alpha, "beta": beta,
            "upper": upper, "lower": lower,
        },
        time_control={"base_s": tc.base, "inc_s": tc.inc},
        openings_used=len(book),
    )


def result_to_dict(r: SPRTResult) -> dict:
    return asdict(r)
