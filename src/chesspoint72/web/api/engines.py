from __future__ import annotations

from fastapi import APIRouter

from chesspoint72.web.game_session import ENGINE_CATALOG

router = APIRouter()


_ALLOWED_PREFIXES = ("victor-", "paul-")


@router.get("/engines")
def list_engines() -> list[dict]:
    engines: list[dict] = []
    for engine_id, meta in ENGINE_CATALOG.items():
        if not engine_id.startswith(_ALLOWED_PREFIXES):
            continue
        label = meta.get("label") or f"{engine_id} (UCI)"
        item: dict = {"id": engine_id, "label": label, "type": "uci"}
        engines.append(item)

    return engines

