# Calix v3 — Module Selector Agent driven by Claude Sonnet 4.6

Same engine plumbing as v1 (registry → agent → search → UCI loop). The
only behavioural difference is in [`agent.py`](agent.py): instead of a
deterministic rule cascade, the agent calls **Claude Sonnet 4.6** through
the Anthropic API, hands it the registry + the AgentContext, and asks
for an `EngineConfig` back as a JSON object validated against a strict
schema (`output_config.format`).

## How the LLM call is structured

- **Model:** `claude-sonnet-4-6`.
- **System prompt:** describes Calix, the hard config constraints, the
  three mode presets, the position-aware adjustments, and the full
  module registry. Stable across calls in a session, so it sits behind
  a `cache_control: ephemeral` breakpoint to amortise cost.
- **User prompt:** the live `AgentContext` (mode, FEN, clock,
  `can_add_modules`, plus cheap structural hints).
- **Output:** JSON conforming to a JSON schema covering every field of
  `EngineConfig`, returned via `output_config.format`. The schema uses
  enums for `evaluator_name` and `move_ordering` and `additionalProperties: false`
  so a malformed response fails fast rather than corrupting search state.
- **Effort:** `low`. The schema does most of the constraint work; we
  don't need deep reasoning, only correct selection.

## Resilience

The engine never blocks on the API:

- Missing `ANTHROPIC_API_KEY` → fall back to v1's rule cascade.
- `anthropic` package not installed → fall back.
- Network / timeout / 5xx → fall back.
- Response that fails JSON parse or schema validation → fall back.

In every fallback case, an `info string` line on stderr tells the operator
the rule cascade ran.

## Memoisation

The per-`go` UCI loop would hit the API on every move; instead
`select_modules` caches the result keyed by `(mode, can_add_modules,
registry name set)`. One API call per session is the steady state.

## Running

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m chesspoint72.aiengines.jonathan.v3.main --agent-mode aware
# or via the wrapper for GUI launchers:
src/chesspoint72/aiengines/jonathan/v3/calix.sh
```

