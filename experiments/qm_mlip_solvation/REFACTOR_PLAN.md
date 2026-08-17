# Hot-path refactor plan — apply AFTER the full-367 pH-0 sweep completes

The sweep runs `scripts/unified_pipeline.py` across ~37 GPUs; new worker spawns pick
up file changes mid-run, so these edits are **deferred** to avoid mixing code versions
into `logs/ph0_sweep/`. Each is behavior-preserving. **Verification gate (run before
committing any of these):** pick 5 reactions spanning classes (e.g. rxn00695 phosphate,
rxn00579/glycosyl, an isomerase, a truncated, a clean), run with the SAME env flags
(`AUTO_TRUNCATE=1 PH0_AUTO=1`) before and after, and assert identical `ΔG` to 0.1 kJ.

## Already done this session (landed, safe)
- Moved 21 exploration one-offs → `backup/qm_exploration_scripts/`; `README_MODULES.md`.
- Wired isomerase gate + H-mass-balance guard into `ph0_auto` / `unified_pipeline`.

## 1. Split `run_reaction` (unified_pipeline.py ~L240-357) into 3 phases
It currently does prep + per-species G + assembly in one 120-line body. Extract:
- `prepare_reaction(rx, log) -> rx` — the `AUTO_TRUNCATE` block (L242-262) + `PH0_AUTO`
  block (L263-286). Pure reaction→reaction transform; no QM. Easily unit-testable.
- `species_free_energies(pu, rx, keep, pool, log) -> (G, sig)` — the per-species loop
  (L315-331) incl. the `requested_explicit` / `is_spectator_anion` closures (move to
  module-level helpers taking `rx`).
- `assemble_dG(rx, G, sig, log) -> row` — ΔG sum + n_H+ term + exact-Alberty pKa
  transform + U_samp quadrature + UNRESOLVED flag + row dict (L332-357).
`run_reaction` becomes: `rx = prepare_reaction(...); G,sig = species_free_energies(...);
return assemble_dG(...)`.

## 2. Remove dead `seeds` threading
`implicit_G` recomputes `_, keep, pool = sampling_budget(smi)` (L122) and ignores its
`seeds` arg; `explicit_G` uses `N_EXPLICIT_SEEDS`, not its `seeds` arg. Drop `seeds`
from both signatures + `run_reaction` + `main` (`--seeds/--keep/--pool` CLI args are
also unused by the convergent sampler — keep `--only`, drop the rest or mark deprecated).

## 3. Refresh the stale module docstring
Lines 2-22 describe a 3-reaction proof-of-concept ("reproduces all three at once",
"step3b/step5c/step7b"). It is now the production router over 367 TECRDB + 20,802
ModelSEED reactions. Rewrite to match `README_MODULES.md`.

## 4. Rename exploration-era hot-path module names (imports → must be atomic)
- `step4e_targeted.py` → `conformers.py` (exports `pool_confs`, `boltz`).
- `step7b_charge_balanced_waters.py` → `explicit_clusters.py` (exports `bare_geom`).
Update the two `from ...import` lines in `unified_pipeline.py` + any tool refs in the
same commit. (Deferred hardest because a half-applied rename breaks every worker.)

## 5. Factor the two env-gated auto-routing blocks
`AUTO_TRUNCATE` and `PH0_AUTO` blocks share the shape: env-gate → try import → transform
→ log → except-fallback. A small `_apply_stage(rx, enabled, fn, log, label)` helper would
dedupe, but it is low value and can be skipped if it hurts readability of the science.
