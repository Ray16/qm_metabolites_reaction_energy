# TODO — QM (UMA) reaction-ΔG exploration

Living checklist. See `PROGRESS.md` for status/results, `EXPLORATION_LOG.md` for
full reasoning. Check items off as done.

## ▶ TOP PRIORITY (next) — validate a NEW reaction category
Three classes solved (redox ~3 implicit, glycosyl ~13 implicit, nucleotidyl ~2-4
explicit). Efficiency + water-count settled. Move to the next mechanism (candidate:
Δn≠0 hydrolysis/decarboxylation, which stresses trans/rot entropy cancellation).
- [ ] Pick the category + 2-3 reactions with trustworthy experimental ΔG.
- [ ] Score with the fast split + `water_count` rule; compare vs experiment + dGP/eQ.
- [ ] If Δn≠0: handle trans/rot entropy (ideal-gas S no longer cancels — see below).

## DONE — efficiency + water-count (this session)
- [x] **Killed xtb --ohess bottleneck**: fast split `thermal_solv.corr_fast` = batched
      UMA-Hessian thermal (GPU) + `xtb --sp --cosmo` solvation. Accuracy-neutral for ΔG
      (bare +4-7 kJ; nucleotidyl −9.5 vs −12.5), ~10× faster. Ladder relax BATCHED;
      backends on separate GPUs; `cache.py` wired (method-tagged).
- [x] **Water count = deterministic coordination rule** (`water_count.py`): 2/hard O,
      1/soft S⁻, 1/cation N-H. First-shell saturation, NOT padding (more ≠ safer:
      extra waters re-explode conformer noise + stop cancelling). Verify per reaction
      with ΔG(n) vs ΔG(n+1) probe (`converged_enough`).
- [x] **ABANDONED + removed occupancy self-selection** (step7/step7c/step8 deleted):
      self-selected peak is noisy (fast pins at cap; ohess bounces 4/6/4 for −2
      phosphate) AND irrelevant to ΔG (insensitive to n; waters cancel). Evidence:
      `test_peak_stability.py` + artifacts/test_peak_stability_{fast,ohess}.json.
- [x] Nucleotidyl PPi SOLVED (step7b): implicit −28..−52 → WPC3 +4.0, exp +1.9.
- [x] Library triage: 40% NEED_EXPLICIT (upper bound), 20% compact polyanions.

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
