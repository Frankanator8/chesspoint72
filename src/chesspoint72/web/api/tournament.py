from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()

_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
_RESULTS_FILE = _ASSETS_DIR / "tournament_results.txt"


def _parse_results(text: str) -> dict[str, Any]:
    engines: list[str] = []
    rounds: list[dict[str, Any]] = []
    champion: str | None = None

    current_round: dict[str, Any] | None = None
    pending_match: dict[str, Any] | None = None

    for line in text.splitlines():
        line = line.strip()

        # Engines list
        m = re.match(r"Engines:\s*\[(.+)\]", line)
        if m:
            engines = [e.strip().strip("'\"") for e in m.group(1).split(",")]
            continue

        # Round header
        m = re.match(r"={4,}\s*ROUND\s+(\d+)\s*={4,}", line)
        if m:
            current_round = {"round": int(m.group(1)), "matches": []}
            rounds.append(current_round)
            pending_match = None
            continue

        # Match declaration (before score details)
        m = re.match(r"Match:\s*(\S+)\s+vs\s+(\S+)", line)
        if m and current_round is not None:
            pending_match = {
                "e1": m.group(1),
                "e2": m.group(2),
                "winner": None,
                "games": None,
                "w1": None,
                "draws": None,
                "w2": None,
                "bye": False,
            }
            current_round["matches"].append(pending_match)
            continue

        # Score line: "Games: G | eA wins: W | Draws: D | eB wins: L"
        m = re.match(
            r"Games:\s*(\d+)\s*\|\s*\S+\s+wins:\s*(\d+)\s*\|\s*Draws:\s*(\d+)\s*\|\s*\S+\s+wins:\s*(\d+)",
            line,
        )
        if m and pending_match is not None:
            pending_match["games"] = int(m.group(1))
            pending_match["w1"] = int(m.group(2))
            pending_match["draws"] = int(m.group(3))
            pending_match["w2"] = int(m.group(4))
            continue

        # Winner line
        m = re.match(r"Winner of match:\s*(\S+)", line)
        if m and pending_match is not None:
            pending_match["winner"] = m.group(1)
            pending_match = None
            continue

        # Bye
        m = re.match(r"Bye:\s*(\S+)\s+automatically advances", line)
        if m and current_round is not None:
            current_round["matches"].append(
                {"e1": m.group(1), "e2": None, "winner": m.group(1), "games": None, "w1": None, "draws": None, "w2": None, "bye": True}
            )
            continue

        # Champion
        m = re.match(r"(?:Champion|Final winner|CHAMPION)[:\s]+(\S+)", line, re.I)
        if m:
            champion = m.group(1)

    return {"engines": engines, "rounds": rounds, "champion": champion}


@router.get("/tournament/bracket")
def get_bracket() -> dict[str, Any]:
    if not _RESULTS_FILE.exists():
        raise HTTPException(status_code=404, detail="tournament_results.txt not found")
    text = _RESULTS_FILE.read_text(encoding="utf-8")
    return _parse_results(text)
