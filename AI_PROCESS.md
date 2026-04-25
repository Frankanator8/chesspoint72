# AI Orchestration & Prompt Ledger

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
