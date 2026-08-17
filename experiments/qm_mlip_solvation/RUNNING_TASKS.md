# Running tasks — resume point (2026-08-17 PM)

Launched to refine Part 1 (pH-0 base extension) + harvest Part 2 (DLPNO glycosyl). Come back and check
outputs; everything is resumable.

## STATUS: A launched 2026-08-17 PM (8 GPUs active). B done. C not launched (ready).
- Check A: `tools/ph0bases_collect.py` (below) once `logs/ph0bases_sweep/*.log` are all present (41).
- Driver: `logs/ph0bases_sweep_driver.log`; per-shard `logs/ph0bases_sweep/g{0..7}.log`.

## What's running / to check

### A. Broad PH0_BASES sweep (GPU, sharded across 8 GPUs) — RUNNING
- **Goal:** real aggregate of the pH-0 base fix over ALL 41 amine-change TECRDB reactions
  (`scripts/ph0bases_sweep_ids.json`; 13 redox/ring + 28 non-redox), + confirm no guard regressions.
- **How:** `tools/ph0bases_sweep.sh` shards the 41 ids across CUDA 0-7, each a `unified_pipeline.py`
  run with `PH0_AUTO=1 PH0_BASES=1` (+ COFACTOR_RING/AUTO_TRUNCATE as the file dictates). Logs ->
  `logs/ph0bases_sweep/<rid>.log`. Uses per-N pKa (step 1, below).
- **Check:** `tools/ph0bases_collect.py` -> per-reaction before/after err + aggregate MAE + guard set.
- **Baseline to compare:** the pre-fix errors are in the ringcofactor/ph0_sweep logs
  (see ERROR_SOURCE_DECOMP.md); deamination was −38..−50, aspartase −26.

### B. Per-N-environment pKa (step 1, cheap, code)
- **Done in** `scripts/ph0_auto.py` `_classify_cations`: ammonia 9.25, α-amino-acid amine 9.6,
  primary aliphatic amine 10.6, guanidinium 12.5, imidazole ~6.5. Was uniform 9.7.
- **Purpose:** fix the 2 outliers. Expectation: helps histidine (imidazole near pH 7); diaminopentanoate
  (−62) is likely a SPECIES/sampling issue (β-amino-ketone product), NOT pKa — verify, don't assume.

### C. DLPNO glycosyl class run (CPU/ORCA, parallel, optional)
- **Goal:** harvest Part 2 — corrected ΔG for the ~15 glycosyl reactions with cc-pVTZ + Boltzmann
  ensemble (pilot showed −12.6 kJ at cc-pVDZ on rxn01362, a lower bound). See `DLPNO_GLYCOSYL_PILOT.md`.
- **Status:** [launched? see below] — CPU-only, does not contend with the GPU sweep.

## Deferred (step 2, not started)
- Truncate the amino-acid substrate to a small core to attack the ~12 kJ floppy-scatter floor. Only do
  this if the sweep shows the floppy floor is the dominant residual after per-N pKa.

## Key state / where things live
- Fix code: `scripts/ph0_auto.py` (build_ph0_reaction_v2, _amine_cn_change gate, per-N pKa),
  `scripts/unified_pipeline.py` (PH0_BASES routing + base-pKa in the Alberty loop).
- Validated single-run: `logs/ph0bases_val.log` (glu −43→−15, ala −50→−6, aspartase −26→+2; guards no-op).
- Memory: `ph0-base-extension-deamination-fix`, `error-source-decomposition`.
- NOT yet default in the worker — decide after the sweep + outlier fixes hold.
