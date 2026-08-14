# TODO — QM (UMA) reaction-ΔG exploration

Living checklist of what's next. See `PROGRESS.md` for status/results,
`EXPLORATION_LOG.md` for the full reasoning. Check items off as done.

## NOW — make the batched Boltzmann pipeline work + get the reproducibility answer
- [x] Batched UMA relaxation (`batched_relax.py`) — verified 0.19 kJ; charge bug fixed
- [x] Straggler-robust early-stop + **drop unconverged stragglers** (bad geometries)
- [x] Energy-TARGETED sampling: ETKDG pool → UMA single-point rank → relax lowest keep
      (settled "which/how many conformers": random-48 wasteful; ~20-24 targeted)
- [ ] **RUN: `step4e_targeted.py --seeds 1,2,3,4,5 --pool 128 --keep 24`**
      → Boltzmann ΔG reproducibility (target std ≤ ~5 kJ, mean near exp −4.2)
      + keep-k convergence sweep (how few conformers suffice)
- [ ] Decide the outcome:
  - reproducible → **per-compound caching architecture is viable** (option 2)
  - still swings → CREST sampling, or matched-transfer conformers, or fragment refs
- [ ] Sanity: re-run REDOX (rxn00070/86) through the batched pipeline — confirm the
      MAE ~3 result survives batching (regression guard)

## NEXT — cover more reactions to map where the method works / breaks
Goal: identify the failure modes to prioritize improvements. Run the batched
pipeline on:
- [ ] Nucleotidyl transfers: rxn01675 (Glc-1-P + TTP → PPi + dTDP-glucose),
      rxn01005 (UTP + sugar-P → PPi + UDP-glucuronate) — PPi + polyanions
- [ ] Remaining glycosyl: rxn00605 (UDP-glc + G6P → UDP + trehalose-6-P),
      rxn01713 (UDP-glc + sinapate → UDP + sinapoyl-glucose)
- [ ] rxn01834 (glyoxalase: S-lactoylglutathione → GSH + methylglyoxal)
- [ ] Tabulate per-reaction: ΔG_pred vs exp, conformer spread, reproducibility,
      which term dominates the error → **the "where to improve" map**

## SCALE — required before any database run (~50k reactions)
- [x] Batched UMA relaxation (`batched_relax.py`, verified 0.19 kJ)
- [ ] Batch ALL compounds' conformers together (not per-reaction) — max throughput
- [ ] Per-compound cache of (conformer energies, ΔGsolv, thermal); reactions = S·G
- [ ] Speed up solvation: xTB only on Boltzmann-relevant conformers (within ~20 kJ
      of min), threaded — currently the bottleneck vs batched UMA
- [ ] Cheap pre-filter (MMFF/GFN-FF rank) → UMA-refine top-k only
- [ ] Amortize thermal: compute once per compound (or cheap vibrational estimate)

## ACCURACY / robustness (open questions)
- [ ] Convergence vs conformer count (24 vs 48 vs 100) — how many needed per compound?
- [ ] Proton reference audit (redox): the −1171 constant is load-bearing
- [ ] Solvation model: is xTB-ALPB enough, or need cluster-continuum for polyanions?
- [ ] Thermal: ideal-gas entropy valid only for Δn=0 reactions — handle Δn≠0
- [ ] Independent validation reactions (not the hard-ten) before any generality claim

## DELIVERABLES
- [ ] Keep PROGRESS.md / EXPLORATION_LOG.md / this TODO.md current
- [ ] Commit + push after each meaningful step (repo: qm_metabolites_reaction_energy)
- [ ] Eventually: per-reaction ΔG table for the hard-ten, vs dGP/eQ/experiment
