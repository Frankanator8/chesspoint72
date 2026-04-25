from __future__ import annotations

import threading
from typing import Callable


def explain_move_async(
    fen: str,
    move_san: str,
    score_cp: int,
    depth: int,
    pv_san: list[str],
    callback: Callable[[str], None],
) -> None:
    """Call the Claude API in a background thread to explain why a move is strong."""
    def _worker() -> None:
        try:
            import anthropic
            client = anthropic.Anthropic()
            score_str = f"{score_cp / 100:+.2f}"
            pv_str = " ".join(pv_san[:5]) if pv_san else "N/A"
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=120,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Chess position FEN: {fen}\n"
                        f"Engine played: {move_san} (eval {score_str}, depth {depth})\n"
                        f"PV: {pv_str}\n"
                        "In 1-2 sentences, explain concisely why this move is strong."
                    ),
                }],
            )
            callback(msg.content[0].text.strip())
        except Exception:
            callback("")
    threading.Thread(target=_worker, daemon=True).start()
