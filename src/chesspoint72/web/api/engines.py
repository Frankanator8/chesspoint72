from __future__ import annotations

from fastapi import APIRouter

from chesspoint72.web.game_session import ENGINE_CATALOG

router = APIRouter()


@router.get("/engines")
def list_engines() -> list[dict]:
    # UI plan expects module metadata for HCE.
    hce_meta = {
        "supports_modules": True,
        "module_groups": {
            "classic": [
                "material",
                "pst",
                "pawns",
                "king_safety",
                "mobility",
                "rooks",
                "bishops",
            ],
            "advanced": ["ewpm", "srcm", "idam", "otvm", "lmdm", "lscm", "clcm", "desm"],
        },
    }

    engines: list[dict] = []
    for engine_id, meta in ENGINE_CATALOG.items():
        label = meta.get("label")
        if not label:
            if meta["type"] == "builtin":
                label = f"{engine_id.upper()} (Built-in)"
            else:
                label = f"{engine_id} (UCI)"
        item: dict = {"id": engine_id, "label": label}
        if meta["type"] == "uci":
            item["type"] = "uci"
        if engine_id == "hce":
            item.update(hce_meta)
        engines.append(item)

    return engines

