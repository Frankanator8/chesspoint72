from __future__ import annotations
import argparse, inspect, sys, pathlib
import chess
import chess.engine
from chesspoint72.app.controller import GameConfig, GameController

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chesspoint72: Pygame UCI chess renderer")
    p.add_argument("--engine", type=str, default=None, help="Directory containing run.sh")
    p.add_argument("--engine-color", choices=["white", "black"], default="black")
    p.add_argument("--movetime", type=float, default=0.2)
    p.add_argument("--fen", type=str, default=None)
    p.add_argument("--square-size", type=int, default=96)
    p.add_argument("--evaluator", choices=["stub", "hce", "material", "nnue"], default=None)
    p.add_argument("--depth", type=int, default=4)
    return p.parse_args(argv)

def _get_accepted_fields() -> set[str]:
    """Dynamically determine GameConfig fields for compatibility."""
    fields = getattr(GameConfig, "__dataclass_fields__", None)
    if fields: return set(fields)
    sig = inspect.signature(GameConfig)
    return {n for n, p in sig.parameters.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    engine = None
    engine_cwd = None

    if args.engine:
        # Resolve the directory. If user points to a file, take the parent directory.
        raw_path = pathlib.Path(args.engine).resolve()
        engine_cwd = raw_path if raw_path.is_dir() else raw_path.parent

        # Validation to prevent the bash error
        if not (engine_cwd / "run.sh").exists():
            print(f"Error: 'run.sh' not found in {engine_cwd}", file=sys.stderr)
            return 1

        print(f"Booting {engine_cwd.name} (giving PyTorch up to 60s)...")
        try:
            # Exact loading logic from engine_match.py
            engine = chess.engine.SimpleEngine.popen_uci(["bash", "run.sh"], cwd=engine_cwd, timeout=60.0)
        except Exception as e:
            print(f"Failed to boot engine: {e}", file=sys.stderr)
            return 1

    # Configuration mapping
    req = {
        "engine": engine,
        "engine_path": ["bash", "run.sh"] if args.engine else None,
        "cwd": str(engine_cwd) if engine_cwd else None,
        "evaluator": args.evaluator,
        "depth": args.depth,
        "engine_color": chess.WHITE if args.engine_color == "white" else chess.BLACK,
        "think_time": args.movetime,
        "square_size": args.square_size,
        "initial_fen": args.fen,
    }

    valid = _get_accepted_fields()
    config = GameConfig(**{k: v for k, v in req.items() if k in valid})

    try:
        GameController(config).run()
    except Exception as e:
        print(f"Runtime Error: {e}", file=sys.stderr)
        return 1
    finally:
        if engine:
            engine.quit()

    return 0

if __name__ == "__main__":
    sys.exit(main())