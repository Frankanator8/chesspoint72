from __future__ import annotations
import argparse, random, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import chess
import chess.engine

def _play_game(white: chess.engine.SimpleEngine, black: chess.engine.SimpleEngine, movetime_s: float, move_cap: int = 400) -> str:
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
        # LIVE OUTPUT PATCH: Print the move directly to the terminal
        move_san = board.san(result.move)
        color_prefix = "W" if board.turn == chess.WHITE else "B"
        print(f"{color_prefix}:{move_san} ", end="", flush=True)
        
        board.push(result.move)
        plies += 1
    print() # Newline when game ends
    return board.result(claim_draw=True)

@dataclass
class MatchResult:
    games: int; e1_wins: int; draws: int; e2_wins: int; elapsed_s: float
    def summary(self, name1: str, name2: str) -> str:
        n = max(self.games, 1)
        return (f"\n=== Engine Match: {name1} vs {name2} ===\n"
                f"  Games: {self.games} | {name1} wins: {self.e1_wins} | Draws: {self.draws} | {name2} wins: {self.e2_wins}\n")

def run_match(engine1_dir: str | Path, engine2_dir: str | Path, games: int = 100, movetime_s: float = 1.0, seed: int | None = None, progress: bool = True, out=sys.stdout) -> MatchResult:
    dir1, dir2 = Path(engine1_dir).resolve(), Path(engine2_dir).resolve()
    name1, name2 = dir1.name, dir2.name
    rng = random.Random(seed)
    e1_wins = e2_wins = draws = 0
    t0 = time.monotonic()

    print(f"Booting {name1} (giving PyTorch up to 60s)...")
    eng1 = chess.engine.SimpleEngine.popen_uci(["bash", "run.sh"], cwd=dir1, timeout=60.0)
    print(f"Booting {name2}...")
    try:
        eng2 = chess.engine.SimpleEngine.popen_uci(["bash", "run.sh"], cwd=dir2, timeout=60.0)
    except Exception:
        eng1.quit()
        raise

    try:
        for g in range(games):
            e1_is_white = rng.choice([True, False])
            white, black = (eng1, eng2) if e1_is_white else (eng2, eng1)
            print(f"--- Game {g+1}/{games} started ---")
            result_str = _play_game(white, black, movetime_s)
            
            if result_str == "1-0":
                e1_wins += 1 if e1_is_white else 0; e2_wins += 0 if e1_is_white else 1
            elif result_str == "0-1":
                e2_wins += 1 if e1_is_white else 0; e1_wins += 0 if e1_is_white else 1
            else:
                draws += 1
            
            if progress:
                print(f"Result: {result_str} [W={e1_wins} D={draws} L={e2_wins}]\n")
    finally:
        for eng in (eng1, eng2):
            try: eng.close()
            except Exception: pass
    return MatchResult(games, e1_wins, draws, e2_wins, round(time.monotonic() - t0, 2))

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--engine1", required=True); p.add_argument("--engine2", required=True)
    p.add_argument("--games", type=int, default=100); p.add_argument("--movetime", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None); p.add_argument("--quiet", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)
    print(run_match(args.engine1, args.engine2, args.games, args.movetime, args.seed, not args.quiet).summary(Path(args.engine1).resolve().name, Path(args.engine2).resolve().name))
    return 0

if __name__ == "__main__":
    sys.exit(main())
