# `PruningConfig` Interface Contract

The forward-pruning module reads, but does not define, a `PruningConfig`
value. The struct itself is owned by a separate prompt. This file is the
authoritative spec of every field that prompt must provide, the type the
field must hold, the role it plays at runtime, and the valid range of values.

Treat this as the contract: any field with a different name, type, or
semantics breaks the pruning module without warning, because the module
accesses these as plain attributes (no validation layer).

## Required fields

| Field | Type | Role | Valid range / typical default |
|---|---|---|---|
| `nmp_enabled` | `bool` | Master switch for null-move pruning. False fully disables NMP. | `True` in production; flip to `False` for A/B testing. |
| `futility_enabled` | `bool` | Master switch for futility pruning at frontier nodes (depth == 1). | `True`. |
| `razoring_enabled` | `bool` | Master switch for razoring at depths 2–4. | `True`. |
| `lmr_enabled` | `bool` | Master switch for late-move reductions. | `True`. |
| `futility_margin` | `int` (centipawns) | Slack added to `static_eval` before comparing against `alpha`. Below this slack, quiet moves are pruned at depth 1. | `100`–`300`. Default `200`. Larger = more conservative (prunes less). |
| `razoring_margins` | `Sequence[int]`, length `>= 3` (centipawns) | Per-depth slack for razoring. Indexed by `depth - 2`, so element 0 covers depth 2, element 1 depth 3, element 2 depth 4. Each margin is the gap below alpha at which the engine drops into QS. | Strictly increasing with depth. Typical `(300, 500, 900)`. Element `i` should be roughly the largest single-move material swing the engine wants to *not* prune through. |
| `lmr_min_depth` | `int` | Minimum remaining depth at which LMR may activate. Below this, LMR is skipped — reductions on shallow nodes are too risky. | `>= 2`. Default `3`. |
| `lmr_min_move_index` | `int` | LMR activates only when the move's 0-based index in the ordered list is **greater than** this value. The first `lmr_min_move_index + 1` moves are always searched at full depth. | `>= 0`. Default `3` (so the first 4 moves are never reduced). |

## Field-level invariants the next prompt must guarantee

1. `razoring_margins` must be subscriptable with integer indices `0`, `1`, `2`. A `tuple[int, int, int]` is the cleanest fit; any longer sequence works as well. A `list` is fine but **must not be mutated at runtime** — the pruning module assumes per-call stability.
2. The four `_enabled` flags must always be plain `bool`. Truthy proxies (e.g. `int`, `Optional[bool]`) work in the obvious cases but will silently change behaviour if `None` ever leaks through.
3. `lmr_min_depth` and `lmr_min_move_index` must be integers, not floats. They are compared with `<` and `<=` against the search's integer depth/index.
4. All margins (`futility_margin`, every entry of `razoring_margins`) are in **centipawns** and assumed positive. Negative margins effectively disable the corresponding technique because the gate `static_eval + margin < alpha` will almost always be false.

## Fields the pruning module does NOT read

Anything else the next prompt wants to add to `PruningConfig` (telemetry hooks, per-mode toggles, NNUE-aware margins) is fine — the pruning module ignores unknown attributes. New fields the module *should* read in the future will require a corresponding update here.
