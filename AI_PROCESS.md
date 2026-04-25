# AI Orchestration & Prompt Ledger

## 2026-04-24 - AGENTS.md generation
- Task prompt: Analyze repository architecture/workflows/conventions/integration points and generate root `AGENTS.md` for AI coding agents.
- Discovery prompt used: "Create a concise analysis plan for generating AGENTS.md for this chesspoint72 repository..."
- Files consulted: `README.md`, `CLAUDE.md`, `pyproject.toml`, engine/UI/modules, MCP orchestrator, and test suite.
- Token efficiency (estimated): ~2.1k tokens consumed to synthesize repository guidance vs. repeated per-task rediscovery overhead; expected net savings on subsequent agent tasks due to centralized instructions.

## 2026-04-24 — Forward Pruning Module (Phases 1–5, separate folder)

**Prompt shape:** five-phase brief — Examine → Plan → Implement (NMP, Razoring, Futility, LMR, Zugzwang) → Integrate → Correctness tests — plus three required output artifacts. Constrained to a separate folder; existing `src/` tree must remain untouched.

**Approach:**
- Built `forward_pruning/` as a self-contained module: `pruning.py` (algorithmic core, no config struct), `search_modified.py` (copy of `negamax.py` with pruning calls inserted), `python_chess_board.py` (concrete `Board` adapter — needed because the upstream tree only ships an empty `_StubBoard`), `_test_support.py` (eval / ordering / config shim for tests only), `tests/test_correctness.py` (perft baseline + search-equivalence + 3 tactics + zugzwang).
- Pruning module is engine-agnostic — exposes the protocols `ZugzwangProbe`, `NullMoveBoard`, `SearchHost` rather than importing concrete classes.
- Search-equivalence test (pruning ON vs OFF must agree on best move) chosen as the substantive perft check, since pruning lives in *search*, not move-gen.

**Token efficiency notes:**
- Single five-phase prompt ≈ 1100 input tokens drove a complete implementation + tests in one session — much denser than splitting into per-technique prompts. Cost: a ~700-line response across ~5 file writes plus one debug cycle (3 puzzle replacements before the tactical suite was sound). The puzzle-construction iteration was the only token waste — would mitigate next time by hand-checking puzzle geometry up front.
- Reusing the existing `python-chess` dependency for the `Board` adapter saved ~300 lines of bitboard scaffolding versus rolling our own move generator.
- Memory: did not save anything to `memory/` — task is one-off and the module is self-documenting.

## 2026-04-25 — Stockfish 16+ Move Ordering (SEE + History Tables + Staged Picker)

**Prompt shape:** deep-research + build brief — "research Stockfish 16+ move ordering (CaptureHistory, ContinuationHistory), then build a Python equivalent with tiered Pick Best selection sort and bitboard-optimised SEE." Two parallel sub-agents: codebase exploration + Stockfish source research.

**Files created:**
- `src/chesspoint72/engine/ordering/see.py` — SEE with pre-computed leaper tables + loop-based ray tracing (x-ray reveals via occupancy masking). Uses Stockfish 16 exact piece values: P=208, N=781, B=825, R=1276, Q=2538.
- `src/chesspoint72/engine/ordering/history_tables.py` — `ButterflyHistory` [2][64][64] (cap=7183), `CaptureHistory` [12][64][6] (cap=10692), `ContinuationHistory` [768][768] flat int16 array (~1.2MB, cap=30000), all using Stockfish's gravity/aging formula: `val + clamped − val·|clamped|/cap`.
- `src/chesspoint72/engine/ordering/move_picker.py` — `MovePicker` staged iterator: MAIN_TT → CAPTURE_INIT → GOOD_CAPTURE (SEE threshold = −score/18) → QUIET_INIT (partial_insertion_sort at −3560·depth) → GOOD_QUIET (≥−14000) → BAD_CAPTURE → BAD_QUIET. Separate evasion and qsearch stage paths. `_partial_insertion_sort` ports Stockfish's exact C++ swap-expand-then-insert idiom.

**Key design decisions:**
- Ray tracing chosen over magic-bitboard tables: correct, readable, no pre-generated magic number dependency. Acceptable for Python-speed SEE.
- `array.array('h', ...)` for all history tables: int16 avoids 28-byte Python int object per entry; gravity update clamps before writing.
- Parallel `_moves`/`_scores` lists (never reallocated) + in-place swap = zero object creation in pick_best hot path.
- Continuation history uses flat 768×768 layout; search passes integer context keys (piece_idx×64+sq); CONT_HIST_SENTINEL=−1 for uninitialised frames.

**Token efficiency:** two parallel research agents (~32k tokens combined) provided complete Stockfish source-level detail (exact caps, formulas, magic numbers, stage enum values) in one pass. Implementation prompt was self-contained, requiring no follow-up clarification cycles.

**Verification:** 8-category smoke test passing — gravity_update, all three history tables, partial_insertion_sort, pick_best, SEE attack tables (rook/bishop/knight/pawn), see_ge cases (PxP, NxP-recaptured, QxR-recaptured, RxR-even, x-ray-support).

## 2026-04-25 — Engine Component Registries (Evaluator / Move Ordering / Pruning)

**Prompt shape:** inspect whether a central registry already exists for evaluator, pruning policy, and move-ordering policy; if partial/missing, add the missing registries and expose the canonical function(s).

**Approach:**
- Audited `src/chesspoint72/engine/factory.py`; confirmed `_EVALUATOR_REGISTRY` existed while pruning and move-ordering were hardwired.
- Added `_MOVE_ORDERING_REGISTRY` and `_PRUNING_POLICY_REGISTRY` with `build_move_ordering_policy(...)` and `build_pruning_policy(...)` builders.
- Added `list_registered_components()` to provide one introspection point for all three concerns.
- Kept behavior stable by default (`stub` move ordering, `forward` pruning), and threaded optional CLI args (`--move-ordering`, `--pruning-policy`) through `_parse_cli` and `build_controller`.

**Token efficiency (estimated):** ~1.5k tokens total (audit + patch + targeted verification) to generalize factory wiring across two additional seams with minimal code churn and no search-behavior drift under defaults.

## 2026-04-25 - Frank v3 engine profile assembly

**Prompt shape:** inspect existing engine modules (HCE, NNUE, ordering, pruning), select strongest practical combination, and implement it self-contained under `src/chesspoint72/aiengines/frank/v3`.

**Approach:**
- Added `frank/v3` package with its own UCI entrypoint and builder (`engine.py`, `__main__.py`, `__init__.py`).
- Implemented `FrankV3MoveOrderingPolicy` in `ordering.py` to replace factory stub ordering with TT-first and tactical capture prioritization.
- Built robust evaluator chain in Frank v3: prefer NNUE when loadable, otherwise fallback to HCE (`classic,advanced` default).
- Added runnable and documentation artifacts: `src/chesspoint72/aiengines/frank/v3/README.md` and updated `src/chesspoint72/aiengines/frank/README.md`.
- Added focused regression tests in `tests/test_frank_v3.py`.

**Token efficiency (estimated):** ~7k total tokens to audit existing architecture and ship a runnable profile with tests in one pass, avoiding broad factory refactors and preserving current defaults.

## 2026-04-25 - Frank v3 Python-only launcher follow-up

**Prompt shape:** after user deleted `.sh` wrappers, migrate Frank v3 usage/docs to Python-only entrypoints.

**Approach:**
- Removed remaining `.sh` launcher references from `src/chesspoint72/aiengines/frank/v3/README.md`.
- Standardized guidance on `python3 -m chesspoint72.aiengines.frank.v3` and installed console scripts (`chesspoint72-frank-v3`, `chesspoint72-engine`) for tooling that needs executable paths.

**Token efficiency (estimated):** ~1k tokens for targeted grep + doc patch; no engine-logic churn.
## 2026-04-25 — Calix engine (agent-driven module selector)

**Prompt shape:** seven-phase brief — Read repo → Create engine → Module Selector Agent → Three info modes (blind / aware / autonomous) → UCI entrypoint → Registry scanner + `# @capability:` tag convention → Tests + README. Mid-session correction: "do not allow the agent to generate or register new modules; the autonomous `can_add_modules=True` flag must remain inert."

**Files created in `src/chesspoint72/aiengines/jonathan/`:**
- `__init__.py` — package marker / map.
- `registry.py` — walks `src/chesspoint72`, parses `# @capability:` tags within the first 40 lines, returns `ModuleDescriptor[]`. Best-effort AST parse of `default_*_config` factories surfaces keyword defaults so the agent can introspect.
- `agent.py` — `AgentContext`, `EngineConfig`, position-inspection helpers (`_is_endgame`, `_is_tactical`), and `select_modules` rule cascade keyed by mode + position + clock. `with_runtime_position` refreshes hints per `go` for non-blind modes.
- `policies.py` — local `StubMoveOrderingPolicy`, `StubPruningPolicy`, and `CapturesFirstOrderingPolicy` (TT-first → captures → quiets, no shared search state required).
- `main.py` — CLI flags (`--agent-mode {blind|aware|autonomous}`, `--depth`, `--time`), `CalixController` re-runs the agent at every `go` so the configuration tracks the live game, activation log written to stderr to keep stdout UCI-clean.
- `README.md` — operator-facing docs covering all three modes and the no-stub-generation guarantee.

**Modules tagged with `# @capability:`** (13 total, discovered by scanner):
`engine.boards.pychess`, `engine.core.transposition`, `engine.evaluators.nnue.evaluator`, `engine.ordering.{move_picker, move_sorter, see}`, `engine.pruning.{algorithms, config, policy}`, `engine.search.negamax.negamax`, `engine.uci.controller`, `hce.{material, hce}`.

**Defensive upstream patch:** `engine/__init__.py` was wrapping its NNUE re-exports in an unconditional import; without `torch` installed even the existing test suite failed at collection time. Replaced the eager import with a `try/except ImportError` that nulls out the symbols. Public API unchanged for users who have torch.

**User mid-session correction:** initial draft of `agent.py` honoured the literal Phase 7/test-3 instruction by writing a stub module file under `aiengines/jonathan/modules/` whenever a desired capability was missing in autonomous mode. After the user countermanded that ("don't allow the agent to generate and register new modules"), `_generate_stub_module` and the surrounding branch were removed; the autonomous-mode test was rewritten to assert the inverse — no synthesised modules, no `chesspoint72.aiengines.jonathan.modules.*` activations.

**Validation surface:** 23 new tests in `tests/test_calix_{registry,agent_modes,integration}.py`, full suite (40 tests including 17 pre-existing) green. Move-generation unchanged ⇒ `run_perft` not required. Search algorithms unchanged ⇒ `play_sprt_match` not required (Calix uses the upstream `NegamaxSearch` and `ForwardPruningPolicy` as-is, only injecting different config knobs). Tactical strength baseline unchanged ⇒ `run_tactics` not required.

**Token efficiency notes:** single seven-phase prompt drove the entire build in one pass. The user correction landed mid-implementation and required only a localised excision (one helper, one branch, one test rewrite) — total revision cost ≈ 600 tokens against the ≈ 9k-token implementation. Letting the registry scanner discover modules from `# @capability:` tags rather than hardcoding a catalogue meant the agent code didn't have to be rewritten when the tag list grew. The biggest avoidable cost was the initial torch debug cycle: a full repo install surfaced the latent NNUE-eager-import bug that I would have hit on first test run anyway, so the cycle was net useful but could have been short-circuited by reading `engine/__init__.py` upfront.
