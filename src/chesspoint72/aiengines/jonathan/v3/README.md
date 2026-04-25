# Calix v3 — Local rule-table Module Selector Agent

Same engine plumbing as v1 (registry → agent → search → UCI loop). The
behavioural difference is in [`agent.py`](agent.py): v3 keeps the same
mode/clock/position rules as v1, but implements selection as a
deterministic **rule table** (apply one mode preset rule, then apply any
matching adjustment rules).

There are **no API calls** and no network dependencies.

## Running

```bash
python -m chesspoint72.aiengines.jonathan.v3.main --agent-mode aware
# or via the wrapper for GUI launchers:
src/chesspoint72/aiengines/jonathan/v3/calix.sh
```

