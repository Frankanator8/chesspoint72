from __future__ import annotations

import asyncio
import importlib.util
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import chess

from chesspoint72.app.builtin_engine import BuiltinEngineClient
from chesspoint72.engine.uci.client import UciEngineClient
from chesspoint72.models import GameState


EngineType = Literal["builtin", "uci"]


_AIENGINES_DIR = Path(__file__).resolve().parents[1] / "aiengines"


def discover_aiengine_uci_entries() -> dict[str, dict[str, Any]]:
    """Discover runnable UCI engines under src/chesspoint72/aiengines/.

    We prefer wrapper scripts shipped inside the aiengines folders:
    - Paul's engines: `run.sh`
    - Jonathan/Calix: `calix.sh`
    """
    entries: dict[str, dict[str, Any]] = {}

    # Paul engines (often depend on torch/NNUE).
    if importlib.util.find_spec("torch") is not None:
        for run_sh in _AIENGINES_DIR.glob("paul/**/run.sh"):
            # Example: .../paul/engine_classic/run.sh -> id paul-classic
            parent = run_sh.parent.name  # engine_classic
            engine_name = parent.replace("engine_", "").replace("_", "-")
            engine_id = f"paul-{engine_name}"
            entries[engine_id] = {
                "type": "uci",
                # Some repo scripts may not have +x set; running via bash is more robust.
                "command": ["bash", str(run_sh)],
                "label": f"Paul · {engine_name.replace('-', ' ').title()} (UCI)",
            }

    # Jonathan Calix versions.
    for calix_sh in _AIENGINES_DIR.glob("jonathan/**/calix.sh"):
        # Example: .../jonathan/v1/calix.sh -> id jonathan-calix-v1
        vdir = calix_sh.parent.name  # v1/v2/v3
        if not vdir.startswith("v"):
            continue
        engine_id = f"jonathan-calix-{vdir}"
        entries[engine_id] = {
            "type": "uci",
            "command": ["bash", str(calix_sh)],
            "label": f"Jonathan · Calix {vdir.upper()} (UCI)",
        }

    # Frank engines: add lightweight wrappers under the frank folders if present.
    for frank_run in _AIENGINES_DIR.glob("frank/**/run.sh"):
        vdir = frank_run.parent.name
        if not vdir.startswith("v"):
            continue
        engine_id = f"frank-{vdir}"
        entries[engine_id] = {
            "type": "uci",
            "command": ["bash", str(frank_run)],
            "label": f"Frank {vdir.upper()} (UCI)",
        }

    return dict(sorted(entries.items()))


ENGINE_CATALOG: dict[str, dict[str, Any]] = {
    # Built-ins (still useful for demos).
    "material": {"type": "builtin", "evaluator": "material"},
    "hce": {"type": "builtin", "evaluator": "hce"},
    "nnue": {"type": "builtin", "evaluator": "nnue"},
    "stub": {"type": "builtin", "evaluator": "stub"},
    # AI engines discovered from the repo.
    **discover_aiengine_uci_entries(),
}


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(12)}"


@dataclass
class HveSession:
    session_id: str
    engine_id: str
    human_color: Literal["white", "black"]
    depth: int
    think_time: float
    hce_modules: str | None = None
    created_at_s: float = field(default_factory=lambda: time.time())

    state: GameState = field(default_factory=GameState)
    engine: object | None = None
    queue: asyncio.Queue[tuple[str, dict[str, Any]]] = field(
        default_factory=asyncio.Queue
    )

    def engine_color(self) -> chess.Color:
        return chess.BLACK if self.human_color == "white" else chess.WHITE

    def start_engine(self) -> None:
        meta = ENGINE_CATALOG.get(self.engine_id)
        if meta is None:
            raise ValueError(f"unknown engine_id: {self.engine_id!r}")
        etype: EngineType = meta["type"]
        if etype == "builtin":
            self.engine = BuiltinEngineClient(
                evaluator=meta.get("evaluator"),
                hce_modules=self.hce_modules,
                depth=self.depth,
                think_time=self.think_time,
            )
        else:
            self.engine = UciEngineClient(
                command=meta["command"],
                think_time=self.think_time,
            )
        self.engine.start()

    def stop_engine(self) -> None:
        if self.engine is None:
            return
        try:
            self.engine.stop()
        finally:
            self.engine = None

    def snapshot(self) -> dict[str, Any]:
        b = self.state.board
        return {
            "session_id": self.session_id,
            "fen": b.fen(),
            "turn": "white" if b.turn == chess.WHITE else "black",
            "human_color": self.human_color,
            "engine_id": self.engine_id,
            "game_over": b.is_game_over(claim_draw=True),
            "result": b.result(claim_draw=True) if b.is_game_over(claim_draw=True) else "*",
            "moves": list(b.move_stack[i].uci() for i in range(len(b.move_stack))),
            "san": b.san(b.peek()) if b.move_stack else None,
        }


SESSION_STORE: dict[str, HveSession] = {}


def create_hve_session(
    *,
    engine_id: str,
    human_color: Literal["white", "black"],
    depth: int,
    think_time: float,
    hce_modules: str | None,
) -> HveSession:
    sid = new_id("hvse")
    sess = HveSession(
        session_id=sid,
        engine_id=engine_id,
        human_color=human_color,
        depth=max(int(depth), 1),
        think_time=max(float(think_time), 0.05),
        hce_modules=hce_modules,
    )
    sess.start_engine()
    SESSION_STORE[sid] = sess
    return sess


def get_hve_session(sid: str) -> HveSession:
    sess = SESSION_STORE.get(sid)
    if sess is None:
        raise KeyError(sid)
    return sess


def delete_hve_session(sid: str) -> None:
    sess = SESSION_STORE.pop(sid, None)
    if sess is None:
        return
    sess.stop_engine()

