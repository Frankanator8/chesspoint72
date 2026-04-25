# Paul's Engine Suite

**Models used:** Claude Sonnet 4.6 (initial 4 engines), Claude Opus 4.7 (3 hybrid engines)  
**Design phase token estimate:** ~18k input / ~6k output (Sonnet phase) + ~30k input / ~12k output (Opus hybrid phase)  
**All code is self-contained within this folder** (shared infrastructure imports from `chesspoint72.engine.*` but all Paul-specific search logic lives here).

---

## The 5-Model NNUE Ensemble

Each `.pt` file is a dictionary with keys `state_dict`, `h1`, `h2`. The loader in `engine/evaluators/nnue/evaluator.py` reads `h1`/`h2` dynamically and falls back to 256x32 for legacy checkpoints.

| Key | File | Architecture | Specialty |
|---|---|---|---|
| `nnue_baseline` | `nnue_weights.pt` | 256×32 | Generalist — full dataset mix |
| `nnue_tank` | `nnue_tank_final.pt` | 512×64 | Positional depth |
| `nnue_tactician` | `nnue_tactician_final.pt` | 256×32 | Tactical sharpness |
| `nnue_speedster` | `nnue_speedster_final.pt` | 64×16 | NPS / blitz speed |
| `nnue_finisher` | `nnue_finisher_final.pt` | 256×32 | Endgame precision |

### Training data filters (from Colab session)

- **Generalist**: Full Lichess/CCRL game mix, no position filter. Balanced across all phases.
- **Tank**: Long time-control games (≥15+10) only. Positions filtered to middlegame (move 10–40, material > 60). Forces the network to learn deep positional understanding.
- **Tactician**: Positions containing tactical motifs — pins, forks, discovered attacks, sacrifices — extracted via Stockfish annotation (eval swing ≥ 150cp in 1 move). Oversampled sharp positions.
- **Speedster**: Blitz and bullet games (≤3+2). Positions from moves 8–25. Architecture kept intentionally tiny (64×16) so forward pass dominates search time minimally.
- **Finisher**: Endgame positions only — fewer than 14 total pieces on board, excluding queen endings. Oversampled R+P and K+P structures. Trained with finer centipawn labels from 7-piece Syzygy tablebase probes where available.

---

## Engine Implementations

### `engine_classic.py` — Paul-Classic (control)
- **Evaluator**: `nnue_baseline` (256×32 generalist)
- **Search**: Standard `NegamaxSearch` + default pruning config
- **Depth**: 5 | **Time**: 5s default
- Purpose: the control engine. All others are benchmarked against this.

### `engine_grinder.py` — Paul-Grinder
- **Evaluator**: `nnue_tank` (512×64 positional)
- **Search**: `GrinderSearch` — `NegamaxSearch` subclass adding **root-level PVS**
- **Pruning**: Conservative (futility margin 150cp, LMR starts at move index 5, tight razoring margins)
- **Depth**: 8 | **Time**: 10s default
- PVS at root: first move searched with full window; all subsequent moves use a null window and only re-search on improvement. Reduces re-search overhead without compromising accuracy when move ordering is good.

### `engine_chaos.py` — Paul-Chaos
- **Evaluator**: `nnue_tactician` (256×32 tactical)
- **Search**: `ChaosSearch` — adds **aspiration windows** to iterative deepening
- **Pruning**: Aggressive NMP (R=3/4 vs default 2/3), standard LMR/futility
- **Depth**: 6 | **Time**: 5s default
- Aspiration windows: each depth starts with a ±50cp window around the prior depth's score. Failed-low/high results widen the window exponentially (delta×2 each time, max 4 retries) before falling back to full window. Rewards the tactician network's ability to identify sharp lines quickly.

### `engine_bullet.py` — Paul-Bullet
- **Evaluator**: `nnue_speedster` (64×16 blitz)
- **Search**: Standard `NegamaxSearch` + bullet pruning config
- **Pruning**: Razoring **disabled**, aggressive LMR (starts at move 2, depth 2), NMP R=3/4, futility 400cp
- **Depth**: 4 | **Time**: 1s default
- Razoring disabled because it triggers a full QS call per node at depths 2-4 — at shallow depths this overhead outweighs the pruning benefit. Aggressive LMR compensates by reducing the tree width instead.

---

## Hybrid Engines (Opus 4.7 design phase)

These three combine multiple specialists or apply unconventional search architectures. Every one has its own search class, its own move-ordering policy, and its own pruning config — no code reuse across them.

### `engine_sentry.py` — Paul-Sentry (Iron Sentry)
- **Evaluator**: `DualPhaseEvaluator` — `nnue_tank` (middlegame) ↔ `nnue_finisher` (endgame), switched per node based on non-king piece count (≤12 → endgame).
- **Move ordering**: `SafetyMoveOrdering` — TT-best > **promotions** > captures > quiet. Promotion priority drives K+P endgame conversion.
- **Pruning** (`sentry_pruning_config`): the most conservative profile in the suite. Razoring **off**, NMP R=1/2 (minimum allowed), futility margin 75cp, LMR delayed until move index 8.
- **Search** (`SentrySearch`): root-PVS plus a **minimum-depth guarantee**. The first 4 plies run with `allotted_time` temporarily set to infinity, so a Sentry move is never the result of a single-ply panic abort. Above the floor, the real time budget is restored.
- **Depth**: 9 | **Time**: 12s default
- Personality: never blunders, slow grind. The dual-eval improves accuracy at phase boundaries but the search behaviour is uniformly conservative.

### `engine_cannon.py` — Paul-Cannon (Glass Cannon)
- **Evaluator**: `nnue_tactician` (256×32 tactical sniper)
- **Move ordering**: `CaptureFirstOrdering` — TT-best > captures (sorted by **victim piece value** descending: Q > R > B/N > P) > promotions > quiet. Direct lookup on `board.py_board` for victim type.
- **Pruning** (`cannon_pruning_config`): ultra-aggressive. Razoring **off**, NMP R=4/5 (vs default 2/3), futility margin 500cp, LMR from move index 1 at depth 2.
- **Search** (`CannonSearch`): two unique pieces stacked together —
  1. **Narrow aspiration windows (Δ=30)**: tighter than Chaos's Δ=50. The tactician's eval is sharply peaked on tactical positions, so a smaller window cuts off faster and only re-searches when the score actually diverges.
  2. **Quiescence-with-checks**: standard qsearch only generates captures, missing forced mates. The first qsearch ply (tracked via a `_qs_depth` counter) also generates checking moves. Deeper qsearch plies revert to captures-only to prevent pathological check-chains.
- **Depth**: 6 | **Time**: 3s default
- Personality: tactical knockouts on a short clock. Any single calculation error is fatal.

### `engine_chameleon.py` — Paul-Chameleon (the flagship)
- **Evaluator**: `PhaseSwitchingEvaluator` — same `nnue_tank` ↔ `nnue_finisher` swap as Sentry, also based on non-king piece count.
- **Move ordering**: `PhaseAwareOrdering` — *different priority list per phase*:
  - Middlegame: TT-best > captures > quiet
  - Endgame: TT-best > **pawn pushes** > captures > **king moves** > quiet
- **Pruning**: **two configs** held in parallel; swapped per `find_best_move` call —
  - `chameleon_middlegame_config`: balanced (default-style) margins
  - `chameleon_endgame_config`: **NMP DISABLED** (zugzwang risk in K+P endings is real), tighter futility/razoring, LMR delayed
- **Search** (`ChameleonSearch`): both pruning policies are `attach_search`-bound at construction time, so the per-move swap is a pure pointer swap. The bootstrap policy is the middlegame one; `find_best_move` overwrites `pruning_policy` and `pruning_config` based on phase before delegating to the standard parent search.
- **Depth**: 7 | **Time**: 8s default
- Personality: two engines in one. Aggressive in the middlegame, careful and zugzwang-safe in the endgame. **Sentry vs Chameleon distinction**: both share the dual evaluator, but Sentry holds a uniformly conservative shell across phases while Chameleon's whole personality (eval + ordering + pruning) shifts at the threshold.
