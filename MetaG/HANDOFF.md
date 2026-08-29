# MetaG — session handoff (2026-08-29)

## Where we are

**MetaG is a working, validated package** (refactored out of `experiments/qm_mlip_solvation`, which stays
put — a MACE-Polar-1 vs UMA agent is active there, do NOT move it).

- Pure-logic core (routing, uncertainty, symmetry, solvation) + QM backend + `pipeline.score_reaction()`.
- **6 non-GPU test modules green**; end-to-end smoke **passed on 4 diverse reactions** (incl. the pH-0 and
  phosphatase-anchor paths). Run: `PYTHONPATH=. <uma-py> tests/test_*.py`.
- Uncertainty layer is CV-honest: symmetric ±2.15σ interval with heavy-tail floor, nested-CV coverage 95.3%,
  class bias as separate point metadata (asymmetric de-bias was CV-rejected). Anchors report `dG_raw`
  alongside `dG`.
- Reference docs: `docs/modelseed_coverage.md` (the ModelSEED thermodynamics findings).

## Key findings this session (the strategic frame)

1. **UMA doesn't beat dGPredictor on TECRDB (11.6 vs held-out 6.5) — and shouldn't be expected to.** TECRDB
   is dGP's home turf; the goal is ModelSEED generalization, not TECRDB MAE.
2. **ModelSEED's `deltag` = eQuilibrator (Noor 2013) + Jankowski GC (2008)** — NOT dGPredictor; all three are
   the same TECRDB-fit additivity family.
3. **Coverage crosstab (43,669 rxns):** ~46% imbalanced (unscoreable by anyone incl. UMA); UMA's real unique
   target = **10,862 balanced-but-GC-silent** reactions; validation overlap = 12,770 balanced+estimate.
4. **The incumbents don't agree on ModelSEED:** GC vs eQ disagree by >10 kJ on **36%**, >30 kJ on **13%**,
   worse with molecular size — direct, experiment-free evidence additivity doesn't generalize there. So even
   `deltag` is an unreliable reference on the frontier.

## Prioritized next steps

1. **[READY TO RUN] Adjudicate the divergent reactions.** `analysis/divergent_inputs.json` holds 14
   tractable reactions where GC and eQ disagree by >30 kJ (built from ModelSEED structures, MetaG input
   format). Run MetaG `score_reaction` on each (GPUs 1-7 free; GPU 0 busy) and compare `dG`/`ci95` vs the
   stored `gc`/`eq`. Insight: does UMA land near one method, split them, or expose both? This is the highest-
   value first ModelSEED experiment — the incumbents contradict each other, so a first-principles adjudicator
   has clear value and needs no experimental ground truth. Builder: `analysis/build_divergent_set.py`.
2. **OOD gate + structural classifier (review item #3).** Now a *blocker* for trusting ModelSEED intervals:
   classify class/σ from structure (charge change, anion SMARTS) not the enzyme note, and widen σ when a
   reaction is unlike the TECRDB calibration set. Without it the ModelSEED intervals are only honest
   in-distribution. `mech_class` is note-based today (rxn falls to `other/clean` default).
3. **Cycle-consistency validation harness.** State-function check across ModelSEED reactions sharing
   compounds — the ground-truth-free way to validate the context-dependent routing (truncation/pH-0/anchor
   can score the same metabolite differently in different reactions → non-closure = bug).
4. **Scale to the GC-silent target (10,862).** Score UMA's unique reactions with OOD-flagged intervals; this
   is the actual deliverable — a coverage-complete, internally-consistent, honestly-bounded ΔG layer.

## Gotchas
- Envs: UMA python `~/miniforge3/envs/uma/bin/python` (torch+fairchem+rdkit); xtb binary for solvation.
- One-job-per-GPU (OOM otherwise); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- `score_reaction(pu, reaction)` takes a dict directly (no file needed); `run_reaction(pu, key,...)` is the
  file-harness wrapper. `exp` is optional.
