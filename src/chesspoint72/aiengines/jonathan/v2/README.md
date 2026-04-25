# Calix v2 — Local scoring-based Module Selector Agent

Same engine plumbing as v1 (registry → agent → search → UCI loop). The
behavioural difference is in [`agent.py`](agent.py): v2 keeps the same
mode/clock/position rules as v1, but implements selection as a tiny
deterministic **utility scoring** problem rather than a straight if/elif
cascade.

There are **no API calls** and no network dependencies.

## Running

```bash
python -m chesspoint72.aiengines.jonathan.v2.main --agent-mode aware
# or via the wrapper for GUI launchers:
src/chesspoint72/aiengines/jonathan/v2/calix.sh
```
