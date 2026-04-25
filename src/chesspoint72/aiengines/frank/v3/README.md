# Frank v3

Frank v3 is a self-contained engine profile under `chesspoint72/aiengines/frank/v3`.

## What It Builds

- Search: `NegamaxSearch` with transposition table and forward pruning (`NMP`, razoring, futility, LMR)
- Ordering: `FrankV3MoveOrderingPolicy` (TT-first, tactical captures via MVV-LVA, promotion and center bias)
- Evaluator preference chain:
  1. NNUE (`NnueEvaluator`) when available and loadable
  2. HCE fallback (`classic,advanced` by default)

## Usage

Run as a UCI engine:

```bash
python3 -m chesspoint72.aiengines.frank.v3
```

Common options:

```bash
python3 -m chesspoint72.aiengines.frank.v3 --depth 6 --time 0.3
python3 -m chesspoint72.aiengines.frank.v3 --no-nnue --hce-modules classic,advanced
```

Alternative (after editable install):

```bash
python3 -m pip install -e .
chesspoint72-frank-v3 --depth 6 --time 0.3
```

Optional environment variables:

- `CHESSPOINT72_NNUE_WEIGHTS`: path to NNUE `.pt` weights
- `CHESSPOINT72_FRANK_HCE_MODULES`: fallback HCE module list

Validation tooling note (Python-only):
- Prefer module execution directly: `python3 -m chesspoint72.aiengines.frank.v3`
- For tools requiring an executable path, use installed console scripts after `python3 -m pip install -e .`:
  - candidate: `chesspoint72-frank-v3`
  - baseline: `chesspoint72-engine`

## Notes Required By Team Convention

- Model used: GPT-5-Codex
- Tokens used (estimated): ~7k for analysis + implementation + test wiring
- Self-contained scope: all Frank v3 orchestration code is in this folder, reusing shared engine seams (`boards`, `search`, `pruning`, `uci`) without cross-folder ownership changes.
