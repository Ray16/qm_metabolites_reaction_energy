# TODO — QM (UMA) reaction-ΔG exploration

Living checklist. See `PROGRESS.md` for status/results, `EXPLORATION_LOG.md` for
full reasoning. Check items off as done.

## ▶ TOP PRIORITY (next session) — kill the xtb --ohess bottleneck
GPUs sit at **0% util**: the pipeline is CPU-bound on `xtb --ohess` (thermal+solv
Hessian on 30-40-atom clusters), UMA/GPU idle. `--ohess` bundles geometry re-opt
(NOT needed, UMA already relaxed) + Hessian thermal (expensive) + COSMO solvation
(cheap, needs no Hessian). Fix:
- [ ] **Thermal via batched UMA Hessian on GPU** (reuse `gibbs_corr` from step6), NOT
      xtb. Solvation via single `xtb --sp --cosmo water` (no Hessian, ~0.5s).
- [ ] **Hessian on the SOLUTE only**, not the full water cluster (waters' thermal
      cancels via the water-cluster reference).
- [ ] **Thermal once per species** (slowly varying): find occupancy peak with
      UMA-electronics + xtb --sp solvation only (fast), add ONE thermal at the peak.
- [ ] Wire `cache.py` into step7c/step8 (key = canon_smiles,charge,n_water; method tag
      encodes engine+thermal+solv+scheme). Bump tag when method changes.
- [ ] Parallelize remaining xtb --sp across CPU cores (OMP_NUM_THREADS=1 each).

## NOW — explicit-water solvation (nucleotidyl SOLVED; generalize)
- [x] Nucleotidyl PPi SOLVED: charge-balanced explicit waters (step7b) implicit
      -28..-52 → WPC2 +5.9 → WPC3 +4.0, exp +1.9. mu_water cancels (charge-conserving).
- [x] Cluster-cycle grand potential (step7c): water-cluster reference → occupancy
      self-selects (no mu_water/pinning). Removed obsolete pinned step7.
- [x] Library triage (`library_solvation_triage.py`): 30,498 scoreable compounds;
      40% NEED_EXPLICIT (carboxylate 7741 + phosphate 4580), 20% compact polyanions,
      11% BORDERLINE (soft/delocalized/cation), 49% IMPLICIT_OK. 6501 R-group excluded.
- [~] **Calibration set (step8) RUNNING** (GPU4, nmax=8): 2-3 small reps/category,
      compare grand-potential PEAK ⟨n⟩ to coordination rule (hard O- ×2-3, soft S-
      ×1-2, cation N-H ×1). Validates whether the fixed-count rule transfers.
- [ ] Read step8 result → per-group water-count TABLE → confirm soft/delocalized
      (thiolate/phenolate) peak LOW (implicit-ish) and hard oxyanions in-band.
- [ ] step7c PPi ladder finish: note large-flexible-molecule peak noise (MePPP -3
      peaked at 4 < MeP -2 at 7, backwards — under-seeding; needs more seeds for big mols).

## KEY DISTINCTION (do not lose) — compound-level vs reaction-level
The 40% NEED_EXPLICIT is an UPPER BOUND. Implicit is FINE when the anion's solvation
CANCELS across the reaction (e.g. glycosyl has phosphate but it's a spectator).
Explicit water is only needed when a compact high-charge-density anion is
CREATED/DESTROYED (PPi formed). Real production burden << 40%. Ladder = calibration
instrument (run on ~a dozen reps once); production = triage + fixed count + implicit
where the group cancels.

## SCALE — before any database run (~50k reactions)
- [x] Batched UMA relaxation (`batched_relax.py`, verified 0.19 kJ)
- [ ] xtb bottleneck fix (see TOP PRIORITY)
- [ ] Per-compound cache of (conformer energies, ΔGsolv, thermal); reactions = S·G
- [ ] Cheap pre-filter (MMFF/GFN-FF rank) → UMA-refine top-k only
- [ ] Triage gate: only route compact-anion-changing reactions to explicit water

## ACCURACY / robustness (open)
- [x] ~10 conformers suffice (keep-k flat); energy-targeted Boltzmann
- [x] Solvation: implicit OK for redox/glycosyl; explicit needed for compact anions
- [ ] Proton reference audit (redox): the -1171 constant is load-bearing
- [ ] Thermal: ideal-gas entropy valid only for Δn=0; handle Δn≠0
- [ ] Per-group water-count rule validation beyond phosphate (step8 delivers this)
- [ ] Independent validation reactions (not the hard-ten) before generality claims

## DELIVERABLES
- [ ] Keep PROGRESS/EXPLORATION_LOG/TODO current; commit+push each step (repo:
      qm_metabolites_reaction_energy, master, SSH)
- [ ] Per-group explicit-water-count table (from step8)
- [ ] Per-reaction ΔG table for the hard-ten vs dGP/eQ/experiment
