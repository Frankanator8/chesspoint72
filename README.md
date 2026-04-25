# chesspoint72

AI-Orchestrated Chess Engine Evaluation System

## Project Overview

This project explores how multiple AI-assisted chess engines can be designed, iterated, and evaluated through structured experimentation. Instead of building a single monolithic engine, we built a modular system of independently AI-generated engines and evaluated them against each other in controlled matches.

The core goal was not just to build a chess engine, but to study:

- How effectively AI can assist in engineering complex systems
- How different AI-generated implementations compare in performance
- How structured human oversight improves AI output quality

We used Claude extensively as a co-engineer across multiple stages of development — not just as a code generator.

---

## Team & Planning

We started by using **Gemini Deep Research** to survey the chess engine landscape. Key finding: chess engines are inherently modular (evaluation, search, move ordering, hashing, etc.), so building independent swappable modules would be far more efficient than generating full engines from scratch each time.

This shaped our entire model-tier strategy:

| Task | Model |
|---|---|
| Well-defined modules (Zobrist hashing, move ordering, point values) | Claude Haiku / Sonnet |
| Novel module ideas → ideation | Gemini Deep Research |
| Novel module ideas → implementation | Claude Opus 4.7 |
| Base architecture (foundational layer) | Claude Opus 4.6 |

The goal was to maximize credit efficiency: use cheaper models for mechanical coding we already understood, and reserve high-power models for design decisions and novel algorithms.

---

## 1. Base Architecture

We first used AI to design a shared base framework with **Claude Opus 4.6** — a deliberate choice, since this layer is foundational to everything else and correctness here multiplies across all engines.

The shared base includes:

- Chess board representation
- Move validation layer
- UCI-compatible interface
- Game loop and engine orchestration system

All modules plug into the same shared interface, ensuring every engine is interoperable and testable in the same environment.

---

## 2. Modular Engine Development

Instead of prompting AI to generate full chess engines repeatedly, we split the system into independent modules and assigned different team members + AI sessions to build variations of:

- Evaluation functions
- Move ordering strategies
- Search depth logic (minimax variations, pruning differences, heuristics)
- Randomized vs deterministic play styles

This created a diverse ecosystem of AI-generated engines and prevented us from rebuilding the same thing over and over and wasting tokens.

**How we diversified across team members:**

Each team member prompted their AI differently to build engines from the existing module library — each engine lives in its own self-contained folder for parallelization:

- **Full reign** — AI given a goal, no constraints on which modules to use
- **Module list** — AI given the list of available modules and told to choose
- **Module list + extend** — AI given the list and told to add new modules as it saw fit

This produced engines with genuinely different "personalities." We also varied the model used: GPT-5.3, Claude Opus, and Claude Sonnet all contributed engines.

---

## 3. MCP Infrastructure

A custom MCP server gives all team members (using Claude) unified access to four evaluation tools. These run autonomously so we don't have to watch test output by hand:

| Tool | What it does |
|---|---|
| `run_perft` | Verifies the engine follows every rule of chess perfectly — counts all legal moves to a given depth |
| `play_sprt_match` | Pits two engine versions against each other in a 100-game arena; uses SPRT Log-Likelihood Ratio to prove (or disprove) an Elo gain mathematically |
| `run_tactics` | Gives the engine a deck of difficult chess puzzles and measures how many it solves within one second |
| `metrics_summary` | Tracks tokens used, cost in cents, and estimated hours of manual testing saved by letting AI run these tests autonomously |

Any change to move generation must pass `run_perft`. Any claimed search improvement must show a positive SPRT LLR from `play_sprt_match` — we don't accept "it feels faster."

---

## 4. Iterative Prompting & Critical Evaluation

We did not accept AI output blindly. For each module:

1. Generate initial AI code
2. Review for correctness, move legality, and efficiency
3. Iterate prompts to fix:
   - Illegal move generation bugs
   - Inefficient search loops
   - Poor evaluation heuristics
4. Refactor AI output into cleaner, more testable components

Each engine went through multiple refinement cycles. AI-generated code was treated as untrusted until validated by the MCP tools above.

Where we got stuck, we switched up the model: Gemini prompts fed into Claude, or escalating from Sonnet to Opus when a problem resisted simpler approaches.

---

## 5. Engine Tournament & Results

A major part of the project is empirical benchmarking: engine vs engine battles.

We ran structured experiments where:

- Engine A vs Engine B (different AI-generated implementations)
- Controlled time settings and move limits
- Multiple games per pairing to reduce randomness

Metrics collected:

- Win/loss/draw ratios
- Average game length
- Blunder frequency (illegal or suboptimal moves)
- Search efficiency (time per move)

This lets us compare AI design choices empirically, not just theoretically.

**Run a tournament:**

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src && \
python3 -m src.chesspoint72.benchmark.tournament \
  --movetime 1 --seed 67 \
  --engines src/chesspoint72/aiengines/frank/v1 \
            src/chesspoint72/aiengines/frank/v2 \
            src/chesspoint72/aiengines/frank/v3 \
  --games 10
```

---

## 6. Engineering Quality

**Documentation**

Each module includes inline comments explaining AI-generated logic, function-level docstrings, and this README covers architecture and experiment setup.

**Testing**

- Unit tests for move legality
- Self-play simulation tests
- Match-based validation via the engine tournament system

**Code Review**

All AI-generated code was reviewed before integration, refactored into a unified style, and checked for shared interface compliance.

---

## 7. Research & Prior Art

We referenced:

- Standard chess engine architecture (minimax, alpha-beta pruning)
- UCI (Universal Chess Interface) protocol design
- Existing open-source engines (e.g., Stockfish behavior principles)
- Academic concepts in adversarial search and heuristics
- Sebastian Lague's tournament-style engine evaluation approach
- Gemini Deep Research for surveying the algorithmic landscape

This informed both our base engine design and our evaluation strategy for AI-generated variants.

---

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
