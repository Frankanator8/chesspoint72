# chesspoint72

A chess engine research platform built by a team of five. It includes a Pygame
chessboard UI, five distinct AI engine families (Frank, Paul, Jonathan, Minal,
Victor), a benchmarking suite, and an MCP-powered validation pipeline that
enforces correctness and Elo gain before any engine change is accepted.

---

## What's inside

```
src/chesspoint72/
├── aiengines/          # The five engine families (frank, paul, jonathan, minal, victor)
├── app/                # Pygame UI — GameController, GameConfig, BuiltinEngineClient
├── benchmark/          # Engine-vs-engine match runner and battle royale tournament
├── engine/             # Core framework: search, move ordering, pruning, UCI, factory
│   ├── core/           # Abstract interfaces (Search, Evaluator, Board, Move, ...)
│   ├── boards/         # python-chess board wrapper
│   ├── ordering/       # Move picker, MVV-LVA, SEE, history tables
│   ├── pruning/        # NMP, LMR, Futility, Razoring
│   ├── search/         # Negamax with iterative deepening, aspiration windows
│   └── uci/            # UCI protocol controller and subprocess client
├── eval_pipeline/      # 9-stage automated evaluation (perft → SPRT → tournament)
├── hce/                # Hand-crafted evaluation: PST, king safety, advanced modules
└── ui/                 # Board renderer, sidebar, move explainer
mcp_orchestrator/       # MCP server exposing perft, SPRT, and tactics tools
tests/                  # pytest test suite
```

---

## Install

Requires **Python 3.10+**.

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

`torch` is optional — only needed for NNUE-based engines (Paul family).

---

## Play chess

**Human vs human**
```bash
python -m chesspoint72.main
```

**Human vs built-in engine** (HCE, runs in-process)
```bash
python -m chesspoint72.main --evaluator hce --depth 4 --engine-color black
```

**Human vs external UCI engine** (e.g. Stockfish)
```bash
python -m chesspoint72.main --engine /path/to/stockfish --engine-color black --movetime 0.5
```

**Start from a custom position**
```bash
python -m chesspoint72.main --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
```

### UI controls
- Click a piece to select it, click a destination to move.
- Click the selected piece again to deselect.
- Pawns auto-promote to queen.

### All CLI options

| Flag | Default | Description |
|---|---|---|
| `--engine PATH` | — | Directory containing a `run.sh` UCI launcher |
| `--engine-color` | `black` | Which side the engine plays (`white` / `black`) |
| `--movetime` | `0.2` | Engine think time in seconds |
| `--evaluator` | — | Built-in evaluator: `stub`, `hce`, `material`, `nnue` |
| `--hce-modules` | — | Comma-separated HCE modules (aliases: `classic`, `advanced`, `all`) |
| `--depth` | `4` | Max search depth for built-in engine |
| `--fen` | — | Starting position in FEN notation |
| `--square-size` | `96` | Board square size in pixels |

---

## The engines

All engines speak UCI and live under `src/chesspoint72/aiengines/`.

### Frank
The team's primary experimental engine. Uses the shared Negamax framework with
full forward pruning (NMP, LMR, Futility, Razoring) and a custom move orderer
that feeds history-table feedback back into ordering.

- **Active variant**: `v1`
- **Evaluator**: HCE classic modules + CLCM
- **Run**: `python -m chesspoint72.aiengines.frank.v1`

### Paul
Seven specialised variants plus three hybrids, all built on the same NNUE
backbone. Each variant tunes a different part of the search.

| Variant | What it tunes |
|---|---|
| `classic` | Baseline NNUE at depth 5 |
| `bullet` | Aggressive LMR, depth 4 |
| `grinder` | PVS at root, depth 8 |
| `chaos` | Aspiration windows, depth 6 |
| `sentry` | Dual-phase eval (NNUE ↔ finisher) |
| `cannon` | Narrow aspiration (Δ=30), qsearch with checks |
| `chameleon` | Phase-switching eval + phase-aware ordering (flagship) |

- **Run**: `bash src/chesspoint72/aiengines/paul/engine_classic/run.sh`

### Jonathan (Calix)
A registry-driven engine that uses an agent to pick which modules to enable at
runtime based on position context.

- **Modes**: `blind` (minimal context), `aware` (default), `autonomous` (full HCE)
- **Run**: `python -m chesspoint72.aiengines.jonathan.v1`

### Minal
Three progressive versions of a clean HCE-based engine.

- **Run**: `python -m chesspoint72.aiengines.minal.v1` (or `v2`, `v3`)

### Victor
Intentionally weak baselines used as low-bar benchmarks.

- `v1` — random beam (~600 ELO), no real search
- `v2` — 1-ply lookahead, avoids hanging pieces
- **Run**: `python -m chesspoint72.aiengines.victor.v1`

---

## Run an engine match

Runs two engines head-to-head and reports wins / draws / losses.

```bash
python -m chesspoint72.benchmark.engine_match \
  --engine1 src/chesspoint72/aiengines/frank/v1 \
  --engine2 src/chesspoint72/aiengines/paul/engine_classic \
  --games 20 --movetime 0.5
```

Run a round-robin tournament across multiple engines:

```bash
python -m chesspoint72.benchmark.battle_royale
```

---

## MCP validation tools

The MCP orchestrator exposes three tools used by AI agents (and humans) to
validate engine changes. Start the server with:

```bash
python -m mcp_orchestrator.mcp_server
```

### `run_perft` — move generation correctness
Sends `go perft <depth>` to a UCI engine and checks node counts.
Must pass before any move-generation change is merged.

### `play_sprt_match` — statistical Elo comparison
Runs a Sequential Probability Ratio Test between two engines.
Returns a `decision` of `H1_accepted` (engine A is better), `H0_accepted`
(no improvement), or `inconclusive`. An LLR-proven Elo gain is required before
accepting any search or evaluation change.

### `run_tactics` — puzzle regression check
Solves a set of EPD tactical puzzles and reports accuracy.
Must not regress after any evaluation change.

---

## Tests

```bash
pytest
```

The test suite covers the UI controller, game state legality, UCI protocol
communication, engine correctness, HCE module loading, pruning, and the
Jonathan agent registry.

---

## Development rules

1. **Move generation change** → must pass `run_perft` with zero illegal moves.
2. **Search or eval change** → must show Elo gain via `play_sprt_match` LLR.
3. **Eval change** → must not regress on `run_tactics`.
4. **Every AI-assisted task** → log model used, token count, and summary in `AI_PROCESS.md`.
