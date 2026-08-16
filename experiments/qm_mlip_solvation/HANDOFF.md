# HANDOFF — QM reaction-ΔG pipeline (2026-08-16)

Authoritative reference = `PIPELINE_REFERENCE.md`. Reasoning log = `EXPLORATION_LOG.md`.
Checklist = `TODO.md`. Repo = own git (`Ray16/qm_metabolites_reaction_energy`, master),
all pushed through commit 6106e23. Memory updated (tecrdb-hard-regimes, tecrdb-empirical-failure-map).

## RUNNING IN BACKGROUND (survives this session; setsid nohup)
- **Full-367 best-pipeline run** (`tools/run_full367.sh`, 8-GPU strict queue, AUTO_TRUNCATE=1).
  ~9/367 done when this was written; ETA ~4-6 h; HEALTHY (0 errors, no OOM). Resumable
  (skips logs with a ΔG). Logs: `logs/full367/<rxn>.log`.
  -> FIRST THING NEW SESSION: `python tools/full367_table.py` refreshes the rolling
     predicted-vs-measured table `artifacts/full367_results.md` (MAE, per-mode, sorted by |err|).
     The rolling monitor from this session dies at session end — just re-run that script, or
     re-arm a monitor. When done, `python tools/analyze_sweep.py` (point it at full367) for
     per-category MAE = the unbiased full-367 UMA+truncation number (roadmap step 2).
- **DFT C-X floor probe** (`tests/dft_cx_floor.py`, CPU): wB97M-V/def2-TZVPD (OMol25's own level)
  on the truncated rxn00605 core, to test if DFT electronics fixes the +16 glycosyl residual.
  donorCap done (DFT≈UMA within 1.7 kJ, neutral); charged glucose-phosphate species still
  computing (slow). Log: `logs/dft_cx_floor.log`. If DFT ΔE_elec closes the residual -> the
  C-X floor is UMA electronic error -> truncation+DFT-electronics hybrid is the fix.

## WHAT WORKS (6 general heuristics, NOTHING fitted to the DB; TECRDB = validation only)
1. **AUTO_TRUNCATE** (`truncate.build_truncated_reaction`, env AUTO_TRUNCATE=1) — removes conserved
   backbone -> kills catastrophic cancellation. rxn00605 -45 -> +16.5 automatic. SAFE via
   n_H+-conservation guard (rejects mis-detected truncations -> full fallback; caught rxn00545).
2. Auto-convergent sampling (bounded). 3. Per-reaction UQ + resolution flag (U_samp is a LOWER
   bound — misses cancellation/electronic error). 4. Spectator-anion guard. 5. pH-0 + pKa transform
   (created/destroyed anions; acetylcholine err -2.9). 6. Systematic truncation (MCS atom-map).

## HEADLINE RESULT
Top-10 dGPredictor-disagreement subset: **UMA+truncation MAE 22** vs retrained-dGP 61, and better
than AIMNet2 (27) / xtb-ALPB (32). Figure `figures/qm_vs_dgpredictor_top10.png` (3 series;
old MACE-POLAR QC composite removed). Roadmap: (1) beat dGP on hard cases DONE ->
(2) validate full-367 (RUNNING) -> (3) extend to ModelSEED.

## FAILURE MAP (empirical, full-molecule baseline; structural flags in tecrdb367_failure_flags.json)
huge/floppy 55% (MAE ~49, cancellation -> AUTO_TRUNCATE) | Mg-prone 24% (MAE 43 -> explicit-Mg,
viable: hydration within 6%, binding TODO) | anion-change 16% (MAE 38 -> pH-0) | isomerase 15%
(MAE 13, near-eq -> flag, concentration-limited) | clean 26%.

## THE ONE FRONTIER LIMIT
**C-X electronic floor** (glycosyl/nucleotidyl/thioester reactive bond, +16..+25 after truncation;
real nucleotidyl rxn01675/01005 still -76/-39). UMA electronic error; AIMNet2≈UMA. Needs DFT/CC ->
the DFT probe now running is the attack.

## NEXT (priority)
1. Read DFT probe result -> if it fixes rxn00605, build truncation+DFT-electronics hybrid.
2. Full-367 table/analysis when done = step-2 validation number.
3. AUTO_TRUNCATE hardening: real atom-mapper (RXNMapper) for the ~44% that fall back
   (atom-splitting NTP->NDP+PPi, multi-coeff) -> would truncate + speed up most of them.
4. Mg-phosphate binding (ligand-substitution; free -3 phosphate needs pH-0/explicit).

## GPUs / ENVS
uma env = `/homes/rzhu/miniforge3/envs/uma/bin/python` (pyscf 2.14 now installed there too).
Launch pattern: `AUTO_TRUNCATE=1 RXN_FILE=... CONV_MAX=5 CUDA_VISIBLE_DEVICES=N setsid nohup
<py> scripts/unified_pipeline.py --only <rxn> > log 2>&1 </dev/null &`. NEVER oversubscribe
GPUs (1 job/GPU; OOM lesson) — use the strict-queue pattern in run_full367.sh.
