from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import chess
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from chesspoint72.app.builtin_engine import BuiltinEngineClient
from chesspoint72.engine.uci.client import UciEngineClient
from chesspoint72.web.game_session import ENGINE_CATALOG, new_id


router = APIRouter()


def _sse(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@dataclass
class EveSession:
    session_id: str
    engine1_id: str
    engine2_id: str
    depth: int
    created_at_s: float = field(default_factory=lambda: time.time())

    board: chess.Board = field(default_factory=chess.Board)
    e1: object | None = None
    e2: object | None = None
    queue: asyncio.Queue[tuple[str, dict[str, Any]]] = field(default_factory=asyncio.Queue)
    stop_flag: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None


_EVE_STORE: dict[str, EveSession] = {}


def _build_engine(engine_id: str, *, depth: int) -> object:
    meta = ENGINE_CATALOG.get(engine_id)
    if meta is None:
        raise ValueError(f"unknown engine: {engine_id!r}")
    if meta["type"] == "builtin":
        eng = BuiltinEngineClient(
            evaluator=meta.get("evaluator"),
            hce_modules=None,
            depth=depth,
            think_time=0.2,
        )
    else:
        eng = UciEngineClient(command=meta["command"], think_time=0.2)
    eng.start()
    return eng


async def _runner(sess: EveSession) -> None:
    await sess.queue.put(("move", {"fen": sess.board.fen(), "uci": None, "ply": 0}))
    ply = 0

    while not sess.stop_flag.is_set() and not sess.board.is_game_over(claim_draw=True):
        ply += 1
        engine = sess.e1 if sess.board.turn == chess.WHITE else sess.e2
        assert engine is not None

        def _pick() -> str:
            mv = engine.request_best_move(sess.board)
            return mv.uci()

        loop = asyncio.get_running_loop()
        try:
            uci = await loop.run_in_executor(None, _pick)
        except Exception as e:
            await sess.queue.put(("game_over", {"result": "*", "error": str(e)}))
            return

        try:
            sess.board.push_uci(uci)
        except Exception:
            await sess.queue.put(("game_over", {"result": "*", "error": "illegal move"}))
            return

        await sess.queue.put(("move", {"fen": sess.board.fen(), "uci": uci, "ply": ply}))

    await sess.queue.put(("game_over", {"result": sess.board.result(claim_draw=True)}))


@router.post("/eve/new")
async def eve_new(body: dict[str, Any]) -> dict[str, Any]:
    engine1_id = str(body.get("engine1_id") or "").strip()
    engine2_id = str(body.get("engine2_id") or "").strip()
    if not engine1_id or not engine2_id:
        raise HTTPException(status_code=400, detail="engine1_id and engine2_id are required")
    depth = max(int(body.get("depth") or 4), 1)

    sid = new_id("eve")
    sess = EveSession(session_id=sid, engine1_id=engine1_id, engine2_id=engine2_id, depth=depth)
    try:
        sess.e1 = _build_engine(engine1_id, depth=depth)
        sess.e2 = _build_engine(engine2_id, depth=depth)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    _EVE_STORE[sid] = sess
    sess.task = asyncio.create_task(_runner(sess))
    return {"session_id": sid, "fen": sess.board.fen(), "turn": "white"}


@router.post("/eve/{sid}/stop")
async def eve_stop(sid: str) -> dict[str, Any]:
    sess = _EVE_STORE.get(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail="unknown session")
    sess.stop_flag.set()
    return {"ok": True}


@router.get("/eve/{sid}/stream")
async def eve_stream(sid: str) -> StreamingResponse:
    sess = _EVE_STORE.get(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail="unknown session")

    async def event_generator():
        while True:
            event_type, data = await sess.queue.get()
            yield _sse(event_type, data)
            if event_type == "game_over":
                # Cleanup.
                try:
                    if sess.e1: sess.e1.stop()
                    if sess.e2: sess.e2.stop()
                finally:
                    _EVE_STORE.pop(sid, None)
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")

