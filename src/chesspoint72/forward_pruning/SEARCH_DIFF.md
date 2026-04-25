# Phase 4 — Search Integration Diff

Two artifacts:
1. `SEARCH_DIFF.patch` — full unified diff between
   `src/chesspoint72/engine/search/negamax/negamax.py` and
   `src/chesspoint72/forward_pruning/search_modified.py`.
   Apply with `patch -p0 < SEARCH_DIFF.patch` from the repo root.
2. This file — focused logical diff: only the algorithmically-meaningful
   changes inside `search_node`, with surrounding context.

The class was renamed (`NegamaxSearch` → `PrunedNegamaxSearch`) and a
`pruning_config` constructor parameter was added; everything else outside
`search_node` is mechanically equivalent to the upstream file.

> **Note (2026-04-24):** Upstream `engine/` was restructured (commit
> `9779d2d refactor: separate engine interfaces from implementations`) and a
> separate upstream `feature/forward_pruning` branch was merged. The patch
> in this folder is now a diff against the upstream baseline as it existed
> when the pruning module was first written; its line numbers will not
> apply cleanly to current upstream until a reviewer reconciles the two
> forward-pruning efforts.

---

## Constructor — accept the config

```diff
 def __init__(
     self,
     evaluator: Evaluator,
     transposition_table: TranspositionTable,
     move_ordering_policy: MoveOrderingPolicy,
     pruning_policy: PruningPolicy,
+    pruning_config,
 ) -> None:
     super().__init__(evaluator, transposition_table,
                      move_ordering_policy, pruning_policy)
+    self.pruning_config = pruning_config
```

`pruning_config` is unannotated by design — the `PruningConfig` struct is
authored by a separate module (see `INTERFACE_CONTRACT.md`).

---

## `search_node` — pruning insertion points

### After TT probe, before move generation

```diff
     if alpha >= beta:
         return tt_entry.score

+    in_check = self._board.is_king_in_check()
     static_eval = self.evaluator_reference.evaluate_position(self._board)
+
+    # ----------------- (1) Null-move pruning ----------------- #
+    nmp_score = pruning.try_null_move_pruning(
+        self, self._board, depth, beta, in_check, self.pruning_config
+    )
+    if nmp_score is not None:
+        return nmp_score
+
+    # --------------------- (2) Razoring --------------------- #
+    razor_score = pruning.try_razoring(
+        self, depth, alpha, static_eval, self.pruning_config
+    )
+    if razor_score is not None:
+        return razor_score
+
     prune_score = self.pruning_policy.try_prune(
         self._board, depth, alpha, beta, static_eval
     )
     if prune_score is not None:
         return prune_score
```

The original `pruning_policy.try_prune` extension hook is **kept** so
existing custom policies are not disturbed.

### Inside the move loop — futility (before make_move) and LMR (around the recursive call)

```diff
     best_score = -_INF
     best_move: Move | None = None
-    for move in moves:
+    for move_index, move in enumerate(moves):
+        move_is_quiet = (not move.is_capture) and (move.promotion_piece is None)
+
+        # ---- (3) Futility pruning (move-level, depth==1, quiet only) ---- #
+        if move_is_quiet and pruning.is_futile(
+            depth, alpha, static_eval, in_check, move_is_quiet, self.pruning_config
+        ):
+            continue
+
         self._board.make_move(move)
         self._ply += 1
-        score = -self.search_node(-beta, -alpha, depth - 1)
+
+        gives_check = self._board.is_king_in_check()
+
+        # --------------------- (4) LMR --------------------- #
+        if pruning.should_apply_lmr(
+            depth, move_index, move_is_quiet, in_check, gives_check, self.pruning_config
+        ):
+            r = pruning.lmr_reduction(depth, move_index)
+            reduced_depth = max(1, depth - 1 - r)
+            score = -self.search_node(-alpha - 1, -alpha, reduced_depth)
+            if score > alpha:
+                # Re-search at full depth — the reduced search raised alpha.
+                score = -self.search_node(-beta, -alpha, depth - 1)
+        else:
+            score = -self.search_node(-beta, -alpha, depth - 1)
+
         self._ply -= 1
         self._board.unmake_move()
```

### Quiescence search

Untouched. Phase 4 explicitly forbids modifying QS, so
`PrunedNegamaxSearch.quiescence_search` is a verbatim copy of the upstream
implementation.

---

## Order of insertion in the node body

Final order of operations inside `search_node` after the changes:

1. Depth-0 → quiescence.
2. Time check.
3. TT probe.
4. Compute `in_check` and `static_eval`.
5. **NMP** → return beta if cutoff.
6. **Razoring** → return QS score if cutoff.
7. Existing `pruning_policy.try_prune` hook (preserved).
8. Move generation + ordering.
9. Move loop:
   - **Futility** filter (depth==1, quiet, before make_move).
   - make_move.
   - **LMR** decision (after make_move so we can read `gives_check`).
   - Recurse — reduced first if LMR fires; re-search full-depth if alpha rises.
   - Update best/alpha/beta.
10. TT store.
