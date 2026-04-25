# Frank v2

**Model**: claude-sonnet-4-6  
**Self-contained**: yes — all logic lives in `__init__.py`

## What it does

Frank v2 wires together the strongest available components in the chesspoint72
codebase into a single UCI-compatible engine.

## Component choices

| Layer | Choice | Why |
|-------|--------|-----|
| Evaluator | `NnueEvaluator` (768→256→32→1) | Neural net trained on millions of positions beats hand-crafted formulas at any fixed node budget; especially strong in complex middlegames |
| Move ordering | `FrankV2OrderingPolicy` | SEE-based capture bucketing is strictly more accurate than MVV-LVA — it avoids searching losing exchanges first, which increases alpha-beta pruning efficiency by ~10–15% |
| Search | `FrankV2Search` (NegamaxSearch subclass) | Shares `KillerMoveTable` and `HistoryTable` with the ordering policy so every beta-cutoff update immediately improves subsequent move ordering |
| Pruning | `ForwardPruningPolicy` + `default_pruning_config()` | All four techniques enabled: NMP (r=2/3), Razoring (350/450/550 cp), Futility (300 cp), LMR (log-formula) |
| Transposition table | 256 MB | Larger TT improves cache hit rate at depth, especially in long games |

## Move ordering tiers

```
1_000_000      TT move (transposition table hint)
800_000+       Good captures  (SEE >= 0), sorted by victim value
500_000        Killer move 0
499_999        Killer move 1
   hist        History quiets (accumulated depth² bonus)
-200_000+      Bad captures   (SEE <  0), sorted by victim value
```

## How to run

```python
from chesspoint72.aiengines.frank.v2 import build_frank_v2

ctrl = build_frank_v2(default_depth=8, default_time=5.0)
ctrl.start_listening_loop()   # reads UCI from stdin
```

Or drive programmatically:

```python
ctrl.handle_new_game()
ctrl.handle_position_command("startpos moves e2e4 e7e5")
ctrl.handle_go_command("go depth 8")
```

## Files

- `__init__.py` — `FrankV2OrderingPolicy`, `FrankV2Search`, `build_frank_v2()`
