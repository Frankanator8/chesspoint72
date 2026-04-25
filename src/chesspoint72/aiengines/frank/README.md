# Frank

Self-contained chess engine variant. Composes the strongest configuration
the existing chesspoint72 modules allow under a pure-Python time budget.

Active variant: **`v1`** (`src/chesspoint72/aiengines/frank/v1/`).

## Configuration (v1)

- **Search**: `NegamaxSearch` (iterative-deepening fail-soft α-β + qsearch + TT)
- **Forward pruning**: full `default_pruning_config()` — NMP, LMR, Futility, Razoring
- **Move ordering**: `FrankMoveOrdering` — TT-first, MVV-LVA captures,
  history-sorted quiets (shares `HistoryTable` with the search so β-cutoff
  bookkeeping flows back into ordering). This replaces the stub the shared
  factory wires in by default — the largest single source of free Elo.
- **Evaluator**: HCE classic (`material, pst, pawns, king_safety, mobility,
  rooks, bishops`) + `clcm`. NNUE, IDAM, EWPM, SRCM, OTVM, LMDM, LSCM, DESM
  are deliberately **excluded** — see the rationale in
  [`/Users/hanyangliu/.claude/plans/in-chesspoint72-we-have-frolicking-beacon.md`](../../../../../.claude/plans/in-chesspoint72-we-have-frolicking-beacon.md).

## Run

```bash
python -m chesspoint72.aiengines.frank.v1
```

Speaks UCI on stdio.

## Logging convention

Make sure to note down the model used, tokens used, and that everything is
self-contained within this folder.

- Model: <fill in per session>
- Tokens used: <fill in per session>
