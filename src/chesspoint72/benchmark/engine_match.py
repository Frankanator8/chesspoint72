"""Head-to-head match between two external UCI engines.

Each engine is launched by executing ``run.sh`` inside its folder.  A fixed
per-move time limit is used so the match does not depend on the host machine's
total clock budget.

CLI:
    python -m chesspoint72.benchmark.engine_match \\
        --engine1 /path/to/engine1 --engine2 /path/to/engine2 \\
        [--games 100] [--movetime 1.0] [--seed 42] [--quiet]
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chess
import chess.engine


# --------------------------------------------------------------------------- #
# Game loop
# --------------------------------------------------------------------------- #


def _play_game(
    white: chess.engine.SimpleEngine,
    black: chess.engine.SimpleEngine,
    movetime_s: float,
    move_cap: int = 400,
) -> str:
    """Play one game from the starting position; return '1-0'/'0-1'/'1/2-1/2'."""
    board = chess.Board()
    limit = chess.engine.Limit(time=movetime_s)
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < move_cap:
        engine = white if board.turn == chess.WHITE else black
        try:
            result = engine.play(board, limit)
        except chess.engine.EngineError:
            return "0-1" if board.turn == chess.WHITE else "1-0"
        if result.move is None:
            break
        board.push(result.move)
        plies += 1
    return board.result(claim_draw=True)


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #


@dataclass
class MatchResult:
    games: int
    e1_wins: int
    draws: int
    e2_wins: int
    elapsed_s: float

    def summary(self, name1: str, name2: str) -> str:
        n = max(self.games, 1)
        w1_pct = 100.0 * self.e1_wins / n
        d_pct = 100.0 * self.draws / n
        w2_pct = 100.0 * self.e2_wins / n
        return (
            f"\n=== Engine Match: {name1} vs {name2} ===\n"
            f"  Games:          {self.games}\n"
            f"  {name1:<14} wins:  {self.e1_wins:>3}  ({w1_pct:.1f}%)\n"
            f"  Draws:                    {self.draws:>3}  ({d_pct:.1f}%)\n"
            f"  {name2:<14} wins:  {self.e2_wins:>3}  ({w2_pct:.1f}%)\n"
            f"  Elapsed: {self.elapsed_s:.1f}s\n"
        )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def run_match(
    engine1_dir: str | Path,
    engine2_dir: str | Path,
    games: int = 100,
    movetime_s: float = 1.0,
    seed: int | None = None,
    progress: bool = True,
    out=sys.stdout,
) -> MatchResult:
    dir1 = Path(engine1_dir).resolve()
    dir2 = Path(engine2_dir).resolve()
    name1 = dir1.name
    name2 = dir2.name
    rng = random.Random(seed)

    e1_wins = e2_wins = draws = 0
    t0 = time.monotonic()

    eng1 = chess.engine.SimpleEngine.popen_uci(["bash", "run.sh"], cwd=dir1)
    try:
        eng2 = chess.engine.SimpleEngine.popen_uci(["bash", "run.sh"], cwd=dir2)
    except Exception:
        eng1.quit()
        raise

    try:
        for g in range(games):
            e1_is_white = rng.choice([True, False])
            white, black = (eng1, eng2) if e1_is_white else (eng2, eng1)

            result_str = _play_game(white, black, movetime_s)

            if result_str == "1-0":
                if e1_is_white:
                    e1_wins += 1
                else:
                    e2_wins += 1
            elif result_str == "0-1":
                if e1_is_white:
                    e2_wins += 1
                else:
                    e1_wins += 1
            else:
                draws += 1

            if progress:
                color1 = "W" if e1_is_white else "B"
                color2 = "B" if e1_is_white else "W"
                print(
                    f"  game {g+1:>3}/{games}  {name1}({color1}) vs {name2}({color2})"
                    f"  -> {result_str}  [W={e1_wins} D={draws} L={e2_wins}]",
                    file=out,
                    flush=True,
                )
    finally:
        for eng in (eng1, eng2):
            try:
                eng.quit()
            except Exception:
                try:
                    eng.close()
                except Exception:
                    pass

    return MatchResult(
        games=games,
        e1_wins=e1_wins,
        draws=draws,
        e2_wins=e2_wins,
        elapsed_s=round(time.monotonic() - t0, 2),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="chesspoint72-bench",
        description="Head-to-head match between two external UCI engines via run.sh.",
    )
    p.add_argument("--engine1", required=True, help="Path to engine 1 folder.")
    p.add_argument("--engine2", required=True, help="Path to engine 2 folder.")
    p.add_argument("--games", type=int, default=100, help="Number of games (default 100).")
    p.add_argument(
        "--movetime", type=float, default=1.0,
        help="Seconds per move (default 1.0).",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility.")
    p.add_argument("--quiet", action="store_true", help="Suppress per-game output.")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_match(
        engine1_dir=args.engine1,
        engine2_dir=args.engine2,
        games=args.games,
        movetime_s=args.movetime,
        seed=args.seed,
        progress=not args.quiet,
    )
    name1 = Path(args.engine1).resolve().name
    name2 = Path(args.engine2).resolve().name
    print(result.summary(name1, name2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
