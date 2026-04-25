from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Literal

import chess
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from chesspoint72.web.game_session import (
    create_hve_session,
    delete_hve_session,
    get_hve_session,
)

router = APIRouter()


def _sse(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _run_engine_turn(sid: str) -> None:
    sess = get_hve_session(sid)
    board = sess.state.board
    if board.is_game_over(claim_draw=True):
        await sess.queue.put(("game_over", {"result": board.result(claim_draw=True)}))
        return
    if board.turn != sess.engine_color():
        return

    await sess.queue.put(("thinking", {"started_at_ms": int(time.time() * 1000)}))

    def _pick() -> str:
        assert sess.engine is not None
        mv = sess.engine.request_best_move(board)
        return mv.uci()

    loop = asyncio.get_running_loop()
    t0 = time.monotonic()
    try:
        uci = await loop.run_in_executor(None, _pick)
    except Exception as e:
        await sess.queue.put(("game_over", {"result": "*", "error": str(e)}))
        return

    elapsed_ms = max(int((time.monotonic() - t0) * 1000), 1)
    await sess.queue.put(("thinking", {"finished_in_ms": elapsed_ms}))

    ok = sess.state.push_uci(uci)
    if not ok:
        await sess.queue.put(("game_over", {"result": "*", "error": "engine produced illegal move"}))
        return

    await sess.queue.put(
        (
            "engine_move",
            {
                "uci": uci,
                "fen": sess.state.board.fen(),
                "turn": "white" if sess.state.board.turn == chess.WHITE else "black",
            },
        )
    )

    if sess.state.board.is_game_over(claim_draw=True):
        await sess.queue.put(
            ("game_over", {"result": sess.state.board.result(claim_draw=True)})
        )


@router.post("/hvse/new")
async def hvse_new(body: dict[str, Any]) -> dict[str, Any]:
    engine_id = str(body.get("engine_id") or "").strip()
    if not engine_id:
        raise HTTPException(status_code=400, detail="engine_id is required")

    human_color: Literal["white", "black"] = body.get("human_color") or "white"
    if human_color not in ("white", "black"):
        raise HTTPException(status_code=400, detail="human_color must be white|black")

    depth = int(body.get("depth") or 4)
    think_time = float(body.get("think_time") or 0.5)
    hce_modules = body.get("hce_modules")
    hce_modules = str(hce_modules).strip() if hce_modules is not None else None

    try:
        sess = create_hve_session(
            engine_id=engine_id,
            human_color=human_color,
            depth=depth,
            think_time=think_time,
            hce_modules=hce_modules,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to start engine: {e}")

    # If engine is White and human chose Black, kick off engine immediately.
    if sess.state.board.turn == sess.engine_color():
        asyncio.create_task(_run_engine_turn(sess.session_id))

    snap = sess.snapshot()
    return {"session_id": sess.session_id, "fen": snap["fen"], "turn": snap["turn"]}


@router.post("/hvse/{sid}/move")
async def hvse_move(sid: str, body: dict[str, Any]) -> dict[str, Any]:
    try:
        sess = get_hve_session(sid)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown session")

    if sess.state.board.is_game_over(claim_draw=True):
        return {
            "ok": False,
            "fen": sess.state.board.fen(),
            "game_over": True,
            "result": sess.state.board.result(claim_draw=True),
        }

    uci = str(body.get("uci") or "").strip()
    if not uci:
        raise HTTPException(status_code=400, detail="uci is required")

    # Human can only move on their turn.
    if sess.state.board.turn == sess.engine_color():
        return {"ok": False, "fen": sess.state.board.fen(), "game_over": False, "result": "*"}

    ok = sess.state.push_uci(uci)
    if not ok:
        return {"ok": False, "fen": sess.state.board.fen(), "game_over": False, "result": "*"}

    # Schedule engine response.
    asyncio.create_task(_run_engine_turn(sid))

    game_over = sess.state.board.is_game_over(claim_draw=True)
    result = sess.state.board.result(claim_draw=True) if game_over else "*"
    return {"ok": True, "fen": sess.state.board.fen(), "game_over": game_over, "result": result}


@router.get("/hvse/{sid}/state")
async def hvse_state(sid: str) -> dict[str, Any]:
    try:
        sess = get_hve_session(sid)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown session")
    return sess.snapshot()


@router.delete("/hvse/{sid}")
async def hvse_delete(sid: str) -> dict[str, Any]:
    delete_hve_session(sid)
    return {"ok": True}


@router.get("/hvse/{sid}/stream")
async def hvse_stream(sid: str) -> StreamingResponse:
    try:
        sess = get_hve_session(sid)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown session")

    async def event_generator():
        while True:
            event_type, data = await sess.queue.get()
            yield _sse(event_type, data)
            if event_type == "game_over":
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")

