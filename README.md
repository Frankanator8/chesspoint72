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

## Features

- 2D chessboard renderer in Pygame
- Click-to-move interaction with legal move validation
- UCI engine support (for example Stockfish)
- CLI options for engine side, think time, board size, and starting FEN

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Run

Human vs human:

```bash
python -m chesspoint72.main
```

Human vs UCI engine (engine plays black):

```bash
python -m chesspoint72.main --engine /opt/homebrew/bin/stockfish --engine-color black --movetime 0.2
```

Human vs built-in HCE engine with selected modules:

```bash
python -m chesspoint72.main --evaluator hce --hce-modules classic,ewpm,desm --engine-color black --depth 4 --movetime 0.2
```

`--hce-modules` accepts comma-separated module names and aliases:
- aliases: `classic`, `advanced`, `all`
- modules: `material,pst,pawns,king_safety,mobility,rooks,bishops,ewpm,srcm,idam,otvm,lmdm,lscm,clcm,desm`

Engine from custom FEN:

```bash
python -m chesspoint72.main --engine /opt/homebrew/bin/stockfish --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
```

## Controls

- Left click a piece to select it.
- Left click a destination square to move.
- Click selected piece again to clear selection.
- Pawn promotions are auto-queen for mouse-only flow.

## Tests

```bash
pytest
```
