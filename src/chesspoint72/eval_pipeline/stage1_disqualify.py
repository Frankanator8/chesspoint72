"""Stage 1 — Hard disqualifiers.

Binary pass/fail checks that every candidate engine must pass before any
competitive game is played.  Failure on any check is an immediate rejection —
the engine does not proceed to Stage 2 or beyond.

Checks (per EVAL_PIPELINE.md):
  1. Illegal move rate   — exactly 0.0   (500 random games)
  2. Crash rate          — exactly 0.0   (200 games with exception monitor)
  3. Timeout rate        — < 1%          (100 games; flag if >1 move exceeds 3×)
  4. Determinism         — True          (same position × 10 seeds → identical moves)
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import chess

from chesspoint72.eval_pipeline.engine_config import EngineConfig, build_engine_for_test
from chesspoint72.eval_pipeline.game_runner import OPENINGS


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass
class DisqualifyResult:
    engine_name: str
    illegal_move_rate: float   # 0.0 required
    crash_rate: float          # 0.0 required
    timeout_rate: float        # < 0.01 required
    is_deterministic: bool     # True required
    passed: bool
    notes: list[str] = field(default_factory=list)

    def print_report(self) -> None:
        print(f"\n=== Stage 1 — Hard Disqualifiers: {self.engine_name} ===")
        checks = [
            ("Illegal move rate", f"{self.illegal_move_rate:.4f}", self.illegal_move_rate == 0.0),
            ("Crash rate",        f"{self.crash_rate:.4f}",        self.crash_rate == 0.0),
            ("Timeout rate",      f"{self.timeout_rate:.4f}",      self.timeout_rate < 0.01),
            ("Determinism",       str(self.is_deterministic),      self.is_deterministic),
        ]
        for name, val, ok in checks:
            print(f"  {name:<22} {val:<10} {'PASS' if ok else 'FAIL'}")
        for note in self.notes:
            print(f"  NOTE: {note}")
        print(f"\n  Verdict: {'QUALIFIED' if self.passed else 'DISQUALIFIED'}\n")


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #

def _check_illegal_moves(config: EngineConfig, n_games: int = 500) -> tuple[float, list[str]]:
    """Play *n_games* and count moves that are not in chess.Board.legal_moves."""
    engine = build_engine_for_test(config)
    illegal = total = 0
    notes: list[str] = []

    for i in range(n_games):
        opening = OPENINGS[i % len(OPENINGS)]
        game_board = chess.Board(opening)
        plies = 0
        while not game_board.is_game_over(claim_draw=True) and plies < 300:
            fen = game_board.fen()
            try:
                move_obj = engine.get_best_move(fen)
            except Exception:
                break
            if move_obj is None:
                break
            chess_move = chess.Move(
                move_obj.from_square, move_obj.to_square,
                int(move_obj.promotion_piece) if move_obj.promotion_piece else None,
            )
            total += 1
            if chess_move not in game_board.legal_moves:
                illegal += 1
                notes.append(
                    f"Illegal move {chess_move.uci()} in position {fen[:30]}..."
                )
                break
            game_board.push(chess_move)
            plies += 1

    rate = illegal / max(total, 1)
    return rate, notes


def _check_crash_rate(config: EngineConfig, n_games: int = 200) -> tuple[float, list[str]]:
    """Play *n_games* and count games where an unhandled exception occurs."""
    engine = build_engine_for_test(config)
    crashes = notes = []
    crash_count = 0

    for i in range(n_games):
        opening = OPENINGS[i % len(OPENINGS)]
        game_board = chess.Board(opening)
        plies = 0
        game_crashed = False
        while not game_board.is_game_over(claim_draw=True) and plies < 300:
            fen = game_board.fen()
            try:
                move_obj = engine.get_best_move(fen)
            except Exception as exc:
                crash_count += 1
                notes = notes + [f"Game {i+1}: {type(exc).__name__}: {exc}"]
                game_crashed = True
                break
            if move_obj is None:
                break
            chess_move = chess.Move(
                move_obj.from_square, move_obj.to_square,
                int(move_obj.promotion_piece) if move_obj.promotion_piece else None,
            )
            if chess_move not in game_board.legal_moves:
                break
            game_board.push(chess_move)
            plies += 1

    return crash_count / n_games, notes


def _check_timeout_rate(
    config: EngineConfig, n_games: int = 100, tolerance: float = 3.0
) -> tuple[float, list[str]]:
    """Count moves that exceed *tolerance*× the configured time limit."""
    engine = build_engine_for_test(config)
    limit = config.time_limit
    timeouts = total = 0
    notes: list[str] = []

    for i in range(n_games):
        opening = OPENINGS[i % len(OPENINGS)]
        game_board = chess.Board(opening)
        plies = 0
        while not game_board.is_game_over(claim_draw=True) and plies < 100:
            fen = game_board.fen()
            t0 = time.monotonic()
            try:
                move_obj = engine.get_best_move(fen)
            except Exception:
                break
            elapsed = time.monotonic() - t0
            total += 1
            if elapsed > limit * tolerance:
                timeouts += 1
                notes.append(
                    f"Timeout: {elapsed:.2f}s (limit {limit}s × {tolerance})"
                )
            if move_obj is None:
                break
            chess_move = chess.Move(
                move_obj.from_square, move_obj.to_square,
                int(move_obj.promotion_piece) if move_obj.promotion_piece else None,
            )
            if chess_move not in game_board.legal_moves:
                break
            game_board.push(chess_move)
            plies += 1

    return timeouts / max(total, 1), notes


def _check_determinism(config: EngineConfig, n_runs: int = 10) -> tuple[bool, list[str]]:
    """Run the same position *n_runs* times and assert identical moves."""
    engine = build_engine_for_test(config)
    test_fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    moves_seen: list[str] = []

    for _ in range(n_runs):
        try:
            move_obj = engine.get_best_move(test_fen)
        except Exception as exc:
            return False, [f"Exception during determinism check: {exc}"]
        moves_seen.append(move_obj.to_uci_string() if move_obj else "None")

    is_det = len(set(moves_seen)) == 1
    notes: list[str] = []
    if not is_det:
        notes.append(f"Non-deterministic moves: {set(moves_seen)}")
    return is_det, notes


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def run_stage1(config: EngineConfig, verbose: bool = True) -> DisqualifyResult:
    """Run all four hard-disqualifier checks on *config*.

    Returns DisqualifyResult. If ``passed`` is False, the engine must not
    proceed to Stage 3 or any competitive testing.
    """
    all_notes: list[str] = []

    if verbose:
        print(f"\nStage 1: checking {config.name}...")

    illegal_rate, n1 = _check_illegal_moves(config, n_games=500)
    all_notes.extend(n1[:3])  # cap to 3 examples

    crash_rate, n2 = _check_crash_rate(config, n_games=200)
    all_notes.extend(n2[:3])

    timeout_rate, n3 = _check_timeout_rate(config, n_games=100)
    all_notes.extend(n3[:3])

    is_det, n4 = _check_determinism(config, n_runs=10)
    all_notes.extend(n4)

    passed = (
        illegal_rate == 0.0
        and crash_rate == 0.0
        and timeout_rate < 0.01
        and is_det
    )

    result = DisqualifyResult(
        engine_name=config.name,
        illegal_move_rate=illegal_rate,
        crash_rate=crash_rate,
        timeout_rate=timeout_rate,
        is_deterministic=is_det,
        passed=passed,
        notes=all_notes,
    )

    if verbose:
        result.print_report()

    return result
