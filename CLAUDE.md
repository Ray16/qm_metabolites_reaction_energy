# Repo conventions & performance rules (thermodynamic_calc)

## NEVER use `conda run -n <env> <cmd>` in a loop or hot path
`conda run` re-activates the environment on **every** call — measured **~30 s of
pure overhead per invocation** (vs **0.57 s** for the same `xtb` single point via
the direct binary). In a threaded/parallel loop this compounds into a hang
(e.g. 5 seeds × 8 threads × 30 s = machine thrash). This has burned us once
(2026-08-14, step4e xtb solvation stalled 10+ min at 0% GPU).

**Instead, call the binary by its full path** and set thread limits:
```python
XTB_BIN = "/homes/rzhu/miniforge3/envs/xtb/bin/xtb"   # or f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb"
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
subprocess.run([XTB_BIN, xyz, "--gfn", "2", "--chrg", str(q), "--sp", "--alpb", "water"],
               cwd=tmpdir, env=ENV, capture_output=True, text=True, timeout=120)
```
`conda run` is fine ONCE at the top of a shell script; never per-item in Python.

## Set thread limits for CPU tools when parallelizing
CPU QM tools (xtb, etc.) default to multi-threaded (all cores). Running N of them
concurrently oversubscribes the CPU and everything crawls. Always set
`OMP_NUM_THREADS=1` (and `MKL_NUM_THREADS=1`) on each worker when you fan out, and
keep total workers ≤ physical cores.

## UMA / batching (see PROGRESS.md, memory)
- Always BATCH UMA inference (`batched_relax.py`); never sequential per-structure BFGS.
- `AtomicData.from_ase` MUST get `r_data_keys=["spin","charge"]` or every structure
  is computed NEUTRAL. charge/spin in `atoms.info` must be `int`.

## Settled method decisions — QM reaction-ΔG (experiments/qm_mlip_solvation)
Terse DECISIONS only (not an experiment log — results/status live in
`experiments/qm_mlip_solvation/{PROGRESS,EXPLORATION_LOG,TODO}.md`).
**When a stage completes, record the decision here.**

- Engine: UMA `uma-s-1p2` (OMol25), `uma` env; charge/spin `int` in `atoms.info`.
- ΔG = ΔE_elec(UMA gas) + thermal(UMA Hessian) + ΔΔGsolv(xtb) + n_H⁺·G(H⁺,aq,pH7).
- corr backend (thermal+solv): the FAST SPLIT — `thermal_solv.corr_fast` = UMA-Hessian
  thermal (batched finite-diff on GPU) + `xtb --sp --cosmo` solvation single point.
  REPLACES the CPU-bound `xtb --ohess --cosmo` bundle. Accuracy-neutral for reaction
  ΔG (bare species Δ +4-7 kJ vs --ohess; nucleotidyl ΔG −9.5 vs −12.5, within 3 kJ)
  and ~10× faster (GPU-utilizing). CAVEAT: on floppy explicit-water clusters the
  UMA finite-diff Hessian's soft-water modes do NOT cancel like GFN2's analytical
  Hessian → it pins occupancy at the cap. Harmless for ΔG (waters cancel), fatal for
  absolute per-species occupancy — which is why we DON'T use occupancy self-selection
  (below). Always BATCH the ladder relaxation + run backends on separate GPUs.
- Sampling: ETKDG pool → batched UMA single-point rank → relax lowest ~10
  (energy-targeted); Boltzmann ensemble (not min); drop unconverged stragglers.
  keep=10 = fast default (cross-seed std ~6-8 kJ); keep~24 for tight final numbers
  (std ~3). A speed knob — sample more when accuracy matters.
- Solvation model: implicit continuum (ALPB/COSMO) is FINE when no compact
  high-charge-density anion is created/destroyed (redox, glycosyl). For compact
  anions (e.g. PPi) implicit over-solvates → use EXPLICIT first-shell waters
  (cluster-continuum): UMA electronics + xtb(RRHO+COSMO) correction. Solved
  nucleotidyl (implicit −28..−52 → +4, exp +1.9). NOTE: CPCM-X was designed
  FASTER than COSMO(-RS), NOT more accurate — don't crown it from one reaction.
- Water COUNT for explicit solvation: a DETERMINISTIC coordination rule
  (`water_count.py`), NOT a self-selected occupancy peak. Rule = 2 waters per hard
  anionic O (carboxylate/phosphate/sulfonate/sulfate O⁻), 1 per soft S⁻, 1 per
  cationic N–H donor. Rationale: reaction ΔG needs the count (a) ENOUGH — saturate the
  FIRST shell of the compact anionic site — and (b) CONSISTENT (same species → same n
  so waters cancel). A noisy self-selected peak breaks (b). MORE IS NOT SAFER: the
  window is bounded — too few = under-solvated bias, first-shell = accurate, TOO MANY =
  re-exploded conformer noise + wandering waters that stop cancelling + worse-than-
  continuum bulk model + degraded thermal Hessian. Target first-shell coordination and
  STOP; verify per reaction with a cheap ΔG(n) vs ΔG(n+1/site) probe
  (`water_count.converged_enough`, catches BOTH under- and over-watering), not padding.
- ABANDONED (removed): occupancy self-selection via grand potential — former step7
  (pinned monomer-cycle), step7c (cluster-cycle grand potential), step8 (peak
  calibration). The self-selected peak is a noisy, method-dependent observable
  (fast/UMA-Hessian pins at cap; GFN2 --ohess bounces 4/6/4 for a −2 phosphate) AND
  irrelevant to ΔG (insensitive to n, waters cancel). Charge-balanced fixed count
  (n=WPC·|charge|, step7b) still valid for charge-conserving reactions.

## Repo
`thermodynamic_calc/` is its own git repo (remote `qm_metabolites_reaction_energy`,
branch `master`, SSH). Commit + push after each meaningful step; the daily cron
(`daily_commit.sh`, 23:47) now also pushes.
