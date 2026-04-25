"""Head-to-head match between two evaluators.

Each side uses an evaluator pulled from the engine factory's registry, wrapped
in a small alpha-beta search that runs directly on a python-chess board. The
real ``NegamaxSearch`` requires a concrete ``Board`` ABC implementation with
working ``make_move``/``unmake_move``/``generate_legal_moves`` — we don't
have one yet, so the benchmark uses python-chess as the move generator and
calls each evaluator through its ``evaluate_position`` interface. That keeps
the *evaluator* under test (the actual subject of the comparison) while
sidestepping the missing Board impl.

CLI:
    python -m chesspoint72.benchmark.battle_royale \\
        --engine1 nnue --engine2 material --games 20 --depth 3
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import chess

from chesspoint72.engine.core.evaluator import Evaluator
from chesspoint72.engine.factory import build_evaluator


_INF = 10**9
_MATE = 10**8


# Same balanced 2-move openings used by the SPRT runner; varied games
# without needing an external book.
_OPENINGS: tuple[str, ...] = (
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppppp1p/6p1/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",
    "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
)


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


def _eval_white_pov(evaluator: Evaluator, board: chess.Board) -> int:
    """Run the evaluator on *board* and return centipawns from White's POV."""
    return evaluator.evaluate_position(board)


def _alphabeta(
    board: chess.Board,
    evaluator: Evaluator,
    depth: int,
    alpha: int,
    beta: int,
) -> int:
    """Negamax-formulated alpha-beta from the side-to-move's perspective."""
    if board.is_checkmate():
        return -_MATE
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0
    if depth == 0:
        score = _eval_white_pov(evaluator, board)
        return score if board.turn == chess.WHITE else -score

    best = -_INF
    for move in board.legal_moves:
        board.push(move)
        try:
            score = -_alphabeta(board, evaluator, depth - 1, -beta, -alpha)
        finally:
            board.pop()
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def _pick_move(board: chess.Board, evaluator: Evaluator, depth: int) -> chess.Move:
    """Return the best move at *depth* for the side to move."""
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError("no legal moves")
    best_move = legal[0]
    best_score = -_INF
    alpha, beta = -_INF, _INF
    for move in legal:
        board.push(move)
        try:
            score = -_alphabeta(board, evaluator, depth - 1, -beta, -alpha)
        finally:
            board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
    return best_move


# --------------------------------------------------------------------------- #
# Game loop
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
        score = self.e1_wins + 0.5 * self.draws
        pct = 100.0 * score / n
        return (
            f"\n=== Battle Royale: {name1} vs {name2} ===\n"
            f"  Games:   {self.games}\n"
            f"  {name1:<10} wins:  {self.e1_wins}\n"
            f"  Draws:               {self.draws}\n"
            f"  {name2:<10} wins:  {self.e2_wins}\n"
            f"  Score for {name1}:   {score}/{n}  ({pct:.1f}%)\n"
            f"  Elapsed: {self.elapsed_s:.1f}s\n"
        )


def _play_game(
    e1: Evaluator,
    e2: Evaluator,
    e1_is_white: bool,
    depth: int,
    starting_fen: str,
    move_cap: int,
) -> str:
    """Play one game; return '1-0' / '0-1' / '1/2-1/2' from White's POV."""
    board = chess.Board(starting_fen)
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < move_cap:
        side_white = board.turn == chess.WHITE
        evaluator = e1 if (side_white == e1_is_white) else e2
        try:
            move = _pick_move(board, evaluator, depth)
        except ValueError:
            break
        board.push(move)
        plies += 1
    return board.result(claim_draw=True)


def run_match(
    engine1: str,
    engine2: str,
    games: int,
    depth: int,
    move_cap: int = 200,
    progress: bool = True,
    out=sys.stdout,
) -> MatchResult:
    e1 = build_evaluator(engine1)
    e2 = build_evaluator(engine2)
    e1_wins = e2_wins = draws = 0
    t0 = time.monotonic()

    for g in range(games):
        opening = _OPENINGS[g % len(_OPENINGS)]
        e1_is_white = (g % 2 == 0)
        result = _play_game(e1, e2, e1_is_white, depth, opening, move_cap)

        if result == "1-0":
            if e1_is_white:
                e1_wins += 1
            else:
                e2_wins += 1
        elif result == "0-1":
            if e1_is_white:
                e2_wins += 1
            else:
                e1_wins += 1
        else:
            draws += 1

        if progress:
            print(
                f"  game {g+1:>3}/{games}  {engine1}{'(W)' if e1_is_white else '(B)'} "
                f"vs {engine2}{'(B)' if e1_is_white else '(W)'}  -> {result}  "
                f"[W={e1_wins} D={draws} L={e2_wins}]",
                file=out,
                flush=True,
            )

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
        prog="chesspoint72-battle",
        description="Head-to-head match between two registered evaluators.",
    )
    p.add_argument("--engine1", required=True, help="Evaluator key (e.g. 'nnue').")
    p.add_argument("--engine2", required=True, help="Evaluator key (e.g. 'material').")
    p.add_argument("--games", type=int, default=10, help="Number of games (default 10).")
    p.add_argument("--depth", type=int, default=3, help="Search depth per move (default 3).")
    p.add_argument(
        "--move-cap", type=int, default=200,
        help="Hard ply cap to prevent infinite drifting games (default 200).",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress per-game output.")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_match(
        engine1=args.engine1,
        engine2=args.engine2,
        games=args.games,
        depth=args.depth,
        move_cap=args.move_cap,
        progress=not args.quiet,
    )
    print(result.summary(args.engine1, args.engine2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
