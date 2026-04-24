from __future__ import annotations

import argparse

import chess

from chesspoint72.app.controller import GameConfig, GameController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chesspoint72: Pygame UCI chess renderer")
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        help="Path to a UCI engine binary (example: /opt/homebrew/bin/stockfish)",
    )
    parser.add_argument(
        "--engine-color",
        choices=["white", "black"],
        default="black",
        help="Side played by the engine when enabled",
    )
    parser.add_argument(
        "--movetime",
        type=float,
        default=0.2,
        help="Engine think time in seconds per move",
    )
    parser.add_argument(
        "--fen",
        type=str,
        default=None,
        help="Optional initial FEN",
    )
    parser.add_argument(
        "--square-size",
        type=int,
        default=96,
        help="Square size in pixels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GameConfig(
        engine_path=args.engine,
        engine_color=chess.WHITE if args.engine_color == "white" else chess.BLACK,
        think_time=args.movetime,
        square_size=args.square_size,
        initial_fen=args.fen,
    )
    GameController(config).run()


if __name__ == "__main__":
    main()

