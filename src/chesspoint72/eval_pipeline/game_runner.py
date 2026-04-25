"""In-process game runner for the eval pipeline.

Plays a single game between two EngineInstance objects using a shared
chess.Board as the source of truth for position and legality.
"""
from __future__ import annotations

import chess

from chesspoint72.eval_pipeline.engine_config import EngineInstance

# Balanced two-move openings: same set used by sprt_tester.py and battle_royale.py,
# extended to 16 for better opening variety in long test runs.
OPENINGS: tuple[str, ...] = (
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",   # 1.e4 e5
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",   # 1.e4 c5
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",   # 1.e4 d5
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",   # 1.d4 d5
    "rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",   # 1.d4 e6
    "rnbqkbnr/pppppp1p/6p1/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",   # 1.d4 g6
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",   # 1.d4 Nf6
    "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",     # 1.Nf3
    "rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2",   # 1.c4 e5
    "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 1 3",  # 1.d4 Nf6 2.c4 e6
    "rnbqkbnr/pp1p1ppp/2p5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",  # 1.e4 c6 2.Nf3 e5
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # 1.e4 e5 2.Nf3 Nc6
    "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3",  # 1.e4 e5 2.Nc3 Nf6
    "rnbqkbnr/ppp2ppp/3p4/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 3",   # 1.e4 e5 2.Nc3 d6
    "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",    # 1.e4 e5 2.d4 Nc6
    "rnbqkb1r/ppp1pppp/3p1n2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 1 3",    # 1.d4 d6 2.e4 Nf6
)


def play_game(
    engine_white: EngineInstance,
    engine_black: EngineInstance,
    opening_fen: str,
    move_cap: int = 300,
) -> str:
    """Play one game; return '1-0', '0-1', or '1/2-1/2' from White's POV.

    Uses a shared chess.Board for legality checking. Each engine's internal
    board is synchronised via set_position_from_fen before each search call.
    An illegal move returned by an engine is treated as an instant loss.
    """
    game_board = chess.Board(opening_fen)
    plies = 0

    while not game_board.is_game_over(claim_draw=True) and plies < move_cap:
        engine = engine_white if game_board.turn == chess.WHITE else engine_black
        fen = game_board.fen()

        try:
            move_obj = engine.get_best_move(fen)
        except Exception:
            return "0-1" if game_board.turn == chess.WHITE else "1-0"

        if move_obj is None:
            break

        chess_move = chess.Move(
            from_square=move_obj.from_square,
            to_square=move_obj.to_square,
            promotion=int(move_obj.promotion_piece) if move_obj.promotion_piece else None,
        )

        if chess_move not in game_board.legal_moves:
            # Illegal move → immediate loss for the side that played it
            return "0-1" if game_board.turn == chess.WHITE else "1-0"

        game_board.push(chess_move)
        plies += 1

    return game_board.result(claim_draw=True)
