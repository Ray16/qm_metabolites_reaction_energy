# PROGRESS — self-contained status snapshot

Read this first. It captures *where we are, what's decided, and what's next* so the
state survives even if external notes are lost. Companion docs:
`EXPLORATION_LOG.md` (full methodology + term-by-term ledgers), `README.md` (plan),
`artifacts/*.json` (raw numbers). Last updated: 2026-08-15.

## ▶ PICK UP HERE (as of 2026-08-15, explicit-water + DB-triage phase)
Redox (err ~3) and glycosyl (err ~13, COSMO) work on implicit solvation. The
NUCLEOTIDYL failure (PPi over-solvated by continuum) is now SOLVED with explicit
first-shell waters. Current thrust: generalize the water-count rule across the DB.

- **✅ Nucleotidyl SOLVED (step7b, charge-balanced explicit waters):** implicit
  −28..−52 → WPC1 −32.3 → WPC2 +5.9 → WPC3 +4.0, converged vs exp +1.9. n=WPC·|charge|
  per species; μ_water CANCELS because the reaction conserves charge (waters balance
  both sides). This is the informative convergence table — KEEP.
- **✅ Cluster-cycle grand potential (step7c):** references added waters to their own
  same-size water cluster G_wc(n) (Bryantsev/Ho) → occupancy PEAKS naturally, no
  μ_water, no pinning. Validates step7b (G_wc cancels for same-charge species). The
  obsolete pinned monomer-cycle step7 was REMOVED (wrong equation).
- **✅ Library triage (`library_solvation_triage.py`):** 30,498 scoreable pH-7
  compounds. 40% NEED_EXPLICIT (carboxylate 7741, phosphate 4580), **20% compact
  polyanions** (PPi-hardest), 11% BORDERLINE (soft/delocalized/cation), 49%
  IMPLICIT_OK; 6501 R-group fragments excluded. artifacts/library_solvation_triage.*
- **▶ RUNNING (GPU4):** `step8_calibration_set.py --nmax 8 --seeds 3` — 2-3 curated
  small reps per category through the cluster-cycle ladder; compares self-selected
  PEAK ⟨n⟩ to the coordination rule. READ logs/step8_full.log →
  artifacts/step8_calibration_set.json when done → per-group water-count table.
- **▶ RUNNING (GPU5):** step7c PPi ladder (nmax=9). NOTE noisy large-molecule peaks
  (MePPP −3 peaked at 4 < MeP −2 at 7 = backwards → under-seeding big flexible mols).

- **⚠ EFFICIENCY (TOP next-session task):** GPUs at **0% util** — CPU-bound on
  `xtb --ohess`. Fix = UMA-Hessian thermal on GPU (once per species, solute-only) +
  `xtb --sp --cosmo` solvation (no Hessian). See TODO ▶ TOP PRIORITY. `cache.py`
  (method-tagged results cache) written, NOT yet wired in.

- **KEY DISTINCTION:** the 40% NEED_EXPLICIT is an UPPER BOUND — implicit is fine when
  the anion CANCELS across the reaction (glycosyl's phosphate is a spectator). Explicit
  water only needed when a compact high-charge-density anion is CREATED/DESTROYED.

## Goal
Can a modern foundation MLIP (**UMA / OMol25**) + explicit-solvation machinery
predict **absolute** biochemical reaction ΔG'° accurately — on the hard reactions
where dGPredictor / eQuilibrator / the old xTB composite fail — and can it scale to
the **whole ModelSEED database (~50k reactions, ~30k compounds)**?

## The method (assembled ΔG, term by term)
```
ΔG_aq = ΔE_elec (UMA gas)              # near-DFT electronic reaction energy
      + ΔG_thermal (UMA Hessian→ideal-gas G)  # ZPE+thermal+entropy
      + ΔΔG_solv (xTB-ALPB water)      # solvation, on UMA geometries
      + n_H⁺·G(H⁺,aq,pH7)             # only if net proton released/consumed
```
**Cancellation strategy (why it works):** absolute ΔG of flexible polyanions is
hopeless directly (conformer noise ±50 kJ). We beat it by (a) **spectator
truncation** — replace the large identical-on-both-sides moiety (NAD tail, UDP,
glutathione backbone) with a small cap so its energy + conformer noise cancel;
(b) picking **Δn_molecules=0** reactions so trans/rot entropy cancels; (c) keeping
the same phosphate anion on both sides so ALPB anion error cancels.

## Results so far
| step | reaction(s) | result | vs incumbents | status |
|---|---|---|---|---|
| 0 | charge ladder → PO₄³⁻ | UMA runs on all polyanions | — | ✅ coverage |
| 1 | 6 isomerizations (naive) | MAE 23.7, conf-spread 49 kJ | — | ❌ noise wall reproduced |
| 2 | redox rxn00070/86 (truncated) | conf-noise 49→7 kJ | — | ✅ cancellation works |
| 3 | + xTB-ALPB solvation | −6.0 kJ | delicate 1000-kJ cancel | ◑ mixed |
| 3b | + thermal + cysteine model | **15.8 vs exp 18/12, MAE ~3** | dGP off ~90, QM ~38 | ✅ strong |
| 4 | glycosyl rxn00579 (truncated) | **−3.0 vs exp −4.2, err 1.2** | dGP off ~43 | ⚠️ good but UNTRUSTED |
| 4b | reproducibility across seeds | std 12.5, range 37 kJ | — | ❌ NOT reproducible |

**Headline:** on the redox couple, UMA + truncation + thermal + solvation reaches
**~3 kJ MAE from first principles (nothing fit)** — a ~30× improvement over the
incumbents on those reactions, and it partially revises the project's standing
"absolute QM can't work" verdict. The glycosyl transfer also landed near
experiment BUT its 85 kJ conformer spread (flexible sugars) means it may be luck.

## Key caveats (do not over-claim)
- Every ΔG is a small residual of ~1000 kJ terms → **~±10 kJ intrinsic sensitivity**.
- Redox: n=2 sharing GSH/GSSG ≈ one independent couple; proton reference is a
  load-bearing constant.
- **Truncation tames conformer noise only when the REACTIVE CENTER is rigid**
  (nicotinamide, 7 kJ). Flexible sugars (Step 4, 85 kJ) reintroduce it.

## THE open decision (what 4b decides) — scale architecture
Reaction-level truncation is accurate but **per-reaction, not cacheable → won't
scale to 50k reactions**. Two scalable options, and the reproducibility test
(Step 4b) picks between them:
1. **Per-compound cache + ΔG=S·G_cached.** Compute each compound's G_aq ONCE
   (amortized over the hundreds of reactions each cofactor appears in), no
   cancellation. **Viable IFF per-compound absolute G converges with feasible
   sampling** — which is exactly what 4b tests (ΔG stable across conformer seeds?).
3. **Fragment-level references.** Cache energies of shared fragments
   (adenosine-diphosphate, nicotinamide, sugar…) that cancel across ALL reactions.
   Scales, but inherits fragment-additivity error (~7 kJ, the dGP/GNN wall).
- 4b **stable** → option (1), clean scalable QM. 4b **swings** → need (2)/matched
  pairs, and additivity accuracy question returns.

## Scalability rules (MANDATORY for any database-scale run)
- **Always BATCH UMA inference** — score/relax many structures in one GPU forward
  pass (`atomicdata_list_to_batch`), never sequential per-structure ASE BFGS. The
  Step 1–4 scripts use sequential BFGS (fine for a few species) → must be
  rewritten batched before scaling. fairchem's own recipes are NOT batched.
- **Per-compound caching + amortization** of G_gas, ΔGsolv, thermal (compute once,
  reuse; cofactors recur in hundreds of reactions).
- Cheap pre-filter (MMFF/GFN-FF) to rank conformers, UMA-refine only the top few.
- Faster is required but **must not cost accuracy** (user directive).

## Implementation inventory (scripts/)
| script | what it does | status |
|---|---|---|
| `probe_uma_charge.py` | Step 0: UMA on charge ladder (neutral→PO₄³⁻) | ✅ done |
| `step1_isomerization_dE.py` | Step 1: naive gas-phase ΔE, 6 isomerizations | ✅ done (neg) |
| `step2_redox_matched.py` | Step 2: truncated-model redox, gas + CHE proton | ✅ done |
| `step3_redox_solvation.py` | Step 3: + xTB-ALPB solvation | ✅ done |
| `step3b_redox_thermal.py` | Step 3b: + UMA-Hessian thermal + cysteine model | ✅ done (MAE ~3) |
| `step4_glycosyl_transfer.py` | Step 4: glycosyl transfer rxn00579 (min, un-batched) | ✅ done (untrusted) |
| `step4b_reproducibility.py` | Step 4b: min-ΔG reproducibility across seeds | ✅ done (std 12.5) |
| `step4c_boltzmann.py` | Step 4c: Boltzmann vs min, un-batched (superseded by 4d) | ⚠️ superseded |
| `batched_relax.py` | **batched FIRE relax** + `batched_energies` (rank) — scale infra; straggler drop | ✅ built+verified (0.19 kJ) |
| `verify_batched_relax.py` | batched-vs-sequential energy/speed check | ✅ done |
| `step4d_batched_boltzmann.py` | batched Boltzmann, RANDOM conformers | ⚠️ superseded by 4e |
| `step4e_targeted.py` | **Step 4e**: energy-TARGETED (pool→rank→relax lowest keep) Boltzmann + conf cache + keep-k convergence | ▶ RUN NEXT (5 seeds) |

Env note: `batched_relax._predict` MUST pass `r_data_keys=["spin","charge"]` to
`AtomicData.from_ase` or charge is dropped (every structure computed neutral).

## Environment & reproduce
- **`uma` conda env**: fairchem-core 2.21.0, torch 2.8+cu128, ase 3.29, rdkit.
  UMA checkpoint `uma-s-1p2` cached at `~/.cache/fairchem/models--facebook--UMA`.
  Load: `pretrained_mlip.get_predict_unit("uma-s-1p2","cuda")` +
  `FAIRChemCalculator(pu, task_name="omol")`. **charge/spin via `atoms.info`, INT.**
- Solvation: `xtb` 6.7.1 binary via `conda run -n xtb xtb … --alpb water --sp`.
- Scripts: `scripts/step{0..4}*.py`. Each prints its full term ledger.
- **Repo:** this lives in `thermodynamic_calc/` = its own git repo,
  remote `github.com/Ray16/qm_metabolites_reaction_energy` (branch `master`, SSH).
  A daily cron (`daily_commit.sh`, 23:47) now commits AND pushes.

## Next steps
1. Read Step 4b reproducibility verdict → choose scale architecture (per-compound
   vs fragment).
2. Build the chosen architecture **batched-first** (per-compound cache, batched
   UMA relaxation + single points).
3. Validate on the remaining hard-ten classes (nucleotidyl transfers) + independent
   reactions before any claim of generality.
4. Only then: database-scale sweep (with UQ / domain flags).
