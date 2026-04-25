from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path

# Import the match runner from the provided script
from chesspoint72.benchmark.engine_match import run_match

POSITIONS: list[str | None] = [
    None,
    "r6r/p1pkq1p1/2pp1nQp/2b1p3/4PB2/2N4P/PPP2PP1/R4RK1 w - - 0 15",
    "r2q1rk1/pbpn1ppp/1p1p1b2/1B1P4/8/2N2N2/PPP2PPP/R2QR1K1 w - - 0 13",
    "r3k2r/pp1qppbp/3p1np1/8/2PQP3/2N1B3/PP3PPP/R3K2R w KQkq - 2 11",
    "r4rk1/1p4p1/p1p1pn1p/b3Nq2/3Pp2P/1P2P1P1/PB3PK1/1R1Q3R w - - 0 23",
    "2kr2nr/pppq1pNp/3p2n1/8/4PP2/4bQ2/PPP3PP/RN2K2R w KQ - 0 12",
    "r1b2rk1/ppp2ppp/2q5/2bn4/8/2P1B3/PP2BPPP/RN1Q1RK1 w - - 0 11",
    "r2q1rk1/ppp2ppp/2n1b3/3n4/2BB4/2P5/PP3PPP/RN1Q1RK1 w - - 2 11",
    "r1q2rk1/pp1nbpp1/n3p2p/8/1Pp2B2/P1N1P2P/2B2PP1/R2Q1RK1 w - - 0 16",
    "r1b2rk1/ppppqppp/8/3nP3/3P4/8/PP2BPPP/RN1QK2R w KQ - 3 11",
]

def run_tournament(
        engine_dirs: list[str],
        output_file: str,
        games_per_match: int = 10,
        movetime_s: float = 1.0,
        seed: int | None = None
) -> None:

    current_round_engines = [Path(d).resolve() for d in engine_dirs]
    round_number = 1
    rng = random.Random(seed)

    with open(output_file, 'w') as f:
        f.write("=== CHESS ENGINE TOURNAMENT INITIALIZED ===\n")
        f.write(f"Engines: {[e.name for e in current_round_engines]}\n")
        f.write(f"Format: {games_per_match} games per match | {movetime_s}s movetime\n\n")
        f.flush()

        while len(current_round_engines) > 1:
            f.write(f"================ ROUND {round_number} ================\n")
            f.flush()
            print(f"\n--- Starting Round {round_number} ---")

            next_round_engines = []

            for i in range(0, len(current_round_engines), 2):
                if i + 1 < len(current_round_engines):
                    e1 = current_round_engines[i]
                    e2 = current_round_engines[i+1]

                    match_header = f"Match: {e1.name} vs {e2.name}"
                    f.write(match_header + "\n")
                    f.flush()
                    print(match_header)

                    result = run_match(
                        engine1_dir=e1,
                        engine2_dir=e2,
                        games=games_per_match,
                        movetime_s=movetime_s,
                        seed=seed,
                        progress=False,
                        positions=POSITIONS,
                    )

                    f.write(result.summary(e1.name, e2.name))
                    f.flush()

                    if result.e1_wins > result.e2_wins:
                        winner = e1
                    elif result.e2_wins > result.e1_wins:
                        winner = e2
                    else:
                        f.write("Result is a tie. Running 1-game sudden death...\n")
                        f.flush()
                        print(f"Tie between {e1.name} and {e2.name}. Running sudden death...")
                        sd_result = run_match(
                            engine1_dir=e1,
                            engine2_dir=e2,
                            games=1,
                            movetime_s=movetime_s,
                            seed=seed,
                            progress=False,
                            positions=POSITIONS,
                        )

                        if sd_result.e1_wins > sd_result.e2_wins:
                            winner = e1
                        elif sd_result.e2_wins > sd_result.e1_wins:
                            winner = e2
                        else:
                            winner = rng.choice([e1, e2])
                            f.write("Sudden death drawn. Winner selected randomly.\n")
                            f.flush()

                    f.write(f"Winner of match: {winner.name}\n\n")
                    f.flush()
                    print(f"Winner: {winner.name}")
                    next_round_engines.append(winner)

                else:
                    e_bye = current_round_engines[i]
                    f.write(f"Bye: {e_bye.name} automatically advances to next round.\n\n")
                    f.flush()
                    print(f"Bye: {e_bye.name}")
                    next_round_engines.append(e_bye)

            current_round_engines = next_round_engines
            round_number += 1

        champion = current_round_engines[0]
        f.write("========================================\n")
        f.write(f"TOURNAMENT CHAMPION: {champion.name}\n")
        f.write("========================================\n")
        f.flush()
        print(f"\nTournament complete. Champion: {champion.name}")
        print(f"All data has been written to {output_file}")

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a single-elimination bracket for chess engines.")
    parser.add_argument("--engines", nargs='+', required=True, help="List of engine directory paths")
    parser.add_argument("--output", default="tournament_results.txt", help="Text file to output the bracket data")
    parser.add_argument("--games", type=int, default=10, help="Games per match")
    parser.add_argument("--movetime", type=float, default=1.0, help="Time per move in seconds")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if len(args.engines) < 2:
        print("Error: You must provide at least two engines to run a tournament.")
        return 1

    run_tournament(
        engine_dirs=args.engines,
        output_file=args.output,
        games_per_match=args.games,
        movetime_s=args.movetime,
        seed=args.seed
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())