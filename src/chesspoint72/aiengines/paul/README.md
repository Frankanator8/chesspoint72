# Paul's Engine Suite

**Model used:** Claude Sonnet 4.6  
**Design phase token estimate:** ~18k input / ~6k output  
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
