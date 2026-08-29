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

## Multi-GPU sweeps: ONE job per GPU at a time (per-GPU sequential shards)
The 15 GB cards OOM if two UMA jobs share one GPU (a big reaction like folate needs >7 GB +
model). Do NOT dispatch N jobs with a global throttle + `GPU = i % NG` — a new job lands on a GPU
whose previous job is still running → OOM (burned 2026-08-18: ~16/82 jobs died mid-sweep). Instead
give each GPU its OWN shard and run it SEQUENTIALLY (`idx=g; while ...; idx+=NG`), so exactly one job
occupies a GPU at a time. Template: `tools/production_sweep.sh` (resumable: skips a rid whose log
already has a ΔG line). Also export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Note: `pgrep -f
unified_pipeline.py` also matches your own shell command — count real jobs with
`nvidia-smi --query-compute-apps=pid` instead.

## Big reactions OOM on the ~15 GB cards: chunk the UMA batches (result-preserving)
The cards are effectively ~14.6 GB (a torch OOM prints "GPU has a total capacity of 14.57 GiB" even
though `nvidia-smi` reports 32 GB — it's partitioned/misleading). Big floppy cofactor/folate
reactions (folate-linked, e.g. rxn00692/rxn15964) exceed that at the default conformer-batch sizes.
Fix WITHOUT changing physics: reduce how many conformers are batched per GPU forward pass — same
conformers, same energies, byte-identical ΔG, smaller footprint. Env knobs (added 2026-08-18):
`UMA_ENERGY_CHUNK` (batched_relax.batched_energies, default 256), `UMA_FIRE_CHUNK`
(batched_relax.batched_fire force eval, default 0=one batch), `UMA_HESS_CHUNK`
(thermal_solv._forces_batched, default 128). Also `torch.cuda.empty_cache()` runs per species in
`unified_pipeline.run_reaction` so a later big species doesn't OOM on fragmentation from earlier ones.
rxn00692 (huge folate) went from OOM to ~5 GB. For the full sweep, apply small chunks ONLY to big
reactions (auto-detect by heavy-atom count) so the ~90% that fit keep default chunks and aren't slowed.

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
- RRHO rotational term: geometry (linear vs nonlinear → drop 5 vs 6 external modes) and
  the rotational symmetry number σ are DETECTED from the geometry (`scripts/mol_symmetry.py`,
  numpy-only proper-rotation count), NOT hard-coded. The old `geometry='nonlinear', σ=1`
  for every species (a) gave water (σ=2, 77 net-water TECRDB rxns) a fixed −RT·ln2 ≈ 1.7 kJ
  rotational-entropy error that does NOT cancel when water is created/destroyed, and (b)
  was PATHOLOGICAL for linear species (CO₂/O₂/H₂/N₂: a spurious 3rd rotational axis with
  I≈0 → CO₂ off by ~29 kJ) — 0 such in TECRDB but ubiquitous in ModelSEED. Detector exact on
  water/CO₂/O₂/H₂/N₂/CO/HCN/NH₃/CH₄ (`tests/test_mol_symmetry.py`); undercounts conservatively
  on rare high-symmetry floppy species (benzene/H₃PO₄), never overcounts. Pass geometry/σ
  explicitly only to override (testing).
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
- **Coherent auto-router (all opt-in flags, all SELF-GATING, all default-on for production):**
  AUTO_TRUNCATE (spectator truncation, radius-sensitivity + mass-balance guarded, falls back to full) ->
  COFACTOR_RING (`cofactor_truncate.py` canonical cores: NAD(P) nicotinamide ring + GSH cysteine-thiol,
  table-driven, couples compose) -> PH0_AUTO (`ph0_auto.py`: neutral-species QM + exact-Alberty pKa;
  gated OFF for isomerases via is_isomerization; REFUSED -> baseline by an H-mass-balance guard when the
  neutralised reaction isn't H-balanced at n_H+=0, which kills the ±1150 kJ net-proton-redox garbage).
  Each gate makes "always on" safe (gated ≥ baseline). Order: cofactor-ring BEFORE truncate BEFORE pH-0.
- **Cofactor ring-truncation (the redox fix):** the NAD −49 kJ bias is the floppy full-cofactor tail
  (not anion solvation — pH-0 alone does nothing for redox). Replace NAD(P)+/H with 1-methylnicotinamide
  ±dihydro, GSH/GSSG with capped-cysteine thiol/disulfide — ISODESMIC (tail cancels), experiment-free.
  Redox class 35.5 -> 16.5. TODO: add FAD (flavin) + CoA (pantetheine-thioester) rows -> attacks the tail.
- **CURRENT ACCURACY (2026-08-21, 362 TECRDB, clean coherent `logs/production/` sweep + rxn01211 routing fix + adenine N9-H data fix):** **MAE 13.2**
  (median 9.6, bias ~0). vs retrained-dGP MAE 5.6 (FIT to TECRDB; its edge is fitting, won't survive
  off-distribution). Prior "15.3" read stale fragmented logs; 13.5 is the honest coherent number.
- **WHERE THE RESIDUAL LIVES (physics root-cause, 2026-08-21, all validated vs INDEPENDENT references —
  never TECRDB):** UMA ELECTRONIC = gold-standard for every class (gas ΔE vs DLPNO-CCSD(T) within ~2 kJ,
  incl. the P-N phosphoramidate); THERMAL (RRHO) fine; solvation FUNCTIONAL error CANCELS in balanced
  reactions (COSMO vs FreeSolv); ionic strength = no gap (exp are standard ΔrG'° at I=0, pipeline too).
  => the residual is the aqueous free energy of REAL solutes = SOLVATION. BUT (2026-08-21) NO CHEAP fix
  survives validation: (1) CHARGED-solute (phosphagen +60, phosphate scatter) — explicit small-cluster
  FAILED (its −55 was min-over-seeds SELECTION BIAS -> −91.7; does not converge; does not reproduce exp
  ion solvation), implicit MODEL SWAP is DEAD (xtb-COSMO/ALPB/ORCA-SMD all agree within ~15-20 kJ), and we
  lack reliable ion ΔGhyd to validate (single-ion convention problem; need MNSol DB). Root cause not cleanly
  established beyond "not electronic/thermal/model-choice"; would need reliable ion data + explicit FEP,
  bounded ~2-3 kJ prize. (2) neutral flexible-conformer (hydratase +16) needs explicit dynamics (open form
  is not a gas minimum). DO NOT build electronic / functional / ionic-strength / cheap-cluster / model-swap
  corrections — all proven not the cause or not helpful (= fitting). **13.5 is near the honest floor for a
  CALIBRATED method; the proven-clean method IS the deliverable; next value = ModelSEED GENERALIZATION, not
  more TECRDB squeezing.** See experiments/.../RUNNING_TASKS.md.

## Repo
`thermodynamic_calc/` is its own git repo (remote `qm_metabolites_reaction_energy`,
branch `master`, SSH). Commit + push after each meaningful step; the daily cron
(`daily_commit.sh`, 23:47) now also pushes.
