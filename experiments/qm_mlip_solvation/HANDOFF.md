# HANDOFF — unified QM pipeline, hard-ten validation

## ▶ START HERE (next session)
Goal: validate the ONE unified pipeline (`scripts/unified_pipeline.py`) across the
**hard-ten** ModelSEED reactions, get honest per-category MAEs, diagnose any failures
(solvation / electronics / sampling / speciation), THEN expand to new categories.

**FIRST: check the 3 background jobs from last session (may be done):**
- `logs/u3_glycosyl_hs.log` — glycosyl at SAMPLE_SCALE=2.5 (does more sampling close the +7?).
- `logs/u_00605.log` — rxn00605 (independent glycosyl, exp −9.51).
- `logs/u_01713.log` — rxn01713 (glycosyl + carboxylate→ester + H+ consumed, exp +3.93).
Read final `ΔG =` line in each. (uma env python: `/homes/rzhu/miniforge3/envs/uma/bin/python`)

## Current scorecard (unified pipeline, one scheme, no per-class tuning)
| reaction | exp | unified ΔG | err | status |
|---|---|---|---|---|
| redox rxn00070/86 | 18.0/11.9 | +21.5 | +3.5/+9.6 | ✅ |
| nucleotidyl rxn01675/01005 | +1.9 | −0.4 | **−2.3** | ✅ SOLVED (was −18.4) |
| glycosyl rxn00579 | −4.2 | +7.0 | +11.2 | ◑ residual electronic/structural |

## The hard-ten (from pipeline/reaction_classes.json) — validation targets
- thiolate_redox: rxn00070, rxn00086 ✅ done | rxn32133, rxn34788 = REVERSES (cheap sign-check)
- glycosyl_transfer: rxn00579 ✅ | **rxn00605, rxn01713 (running — independent, dGP-fails)**
- phosphate_transfer: rxn01675, rxn01005 ✅ (share truncated model)
- other: **rxn01834** = S-lactoylglutathione → GSH + methylglyoxal + H+ (Δn≠0 thioester
  elimination; NOT YET ADDED). Truncate GSH→Me: `CSC(=O)C(C)O → C[S-] + CC(=O)C=O + H+`.
  WATCH the GSH thiol/thiolate trap (ModelSEED assigns GSH thiolate −2, but thiol pKa
  ~9 → likely THIOL at pH7; see [[ten-reaction-microspecies-fix]]).

## Plan (execute in order)
1. Read the 3 running results → update scorecard.
2. **glycosyl depth verdict**: if rxn00605/rxn01713 land near exp (err <~5) → glycosyl
   method generalizes; if they miss like rxn00579 (+11), the residual is a systematic
   glycosyl-electronics/truncation issue → run ethyl-vs-methyl cap test (step6 style) on
   rxn00579 to isolate truncation vs UMA electronic limit.
3. **Add rxn01834** (Δn≠0 probe) to REACTIONS. Handle: Δn≠0 (trans/rot entropy no longer
   cancels — IdealGasThermo per species already gives absolute S, so it SHOULD work, but
   verify), thiolate creation (explicit=True? soft S, water_count=1), thiol-vs-thiolate.
4. **Redox/nucleotidyl depth needs NEW substrates** (the classified extras are reverses).
   Mine `pipeline/TECRDB.csv` (4545 rows, KEGG reactions + K_prime + pH) by EC: redox
   EC 1.x non-glutathione couples; nucleotidyl EC 2.7.7.x. Map KEGG→structure→truncate.
5. Report per-category MAE with n≥3 independent reactions before any generality claim.

## KEY METHOD DECISIONS (don't relitigate)
- **corr = fast split** (`thermal_solv.corr_fast`): UMA-Hessian thermal (batched, GPU) +
  `xtb --sp --cosmo` solvation. ~10× faster than `xtb --ohess`, accuracy-neutral for ΔG.
- **EXPLICIT clusters** (compact anion created/destroyed): Boltzmann over cluster ensemble
  (NOT min — min drifts down with seeds), `xtb --opt --cosmo` RELAXED solvation (not --sp,
  which over-solvates ~40 kJ), **bare-solute** thermal (skip floppy water modes). This is
  what took nucleotidyl −18→0. N_EXPLICIT_SEEDS=16, EXPLICIT_KEEP=8 (env-tunable).
- **IMPLICIT** (anion is spectator, e.g. glycosyl phosphates): Boltzmann(E_UMA+ΔGsolv[cosmo])
  + bare-solute thermal. Sampling budget flexibility-scaled (rotatable bonds); SAMPLE_SCALE
  env multiplier. Batched relax makes sampling ~free — sample generously.
- **Water count** = deterministic coordination rule (`water_count.py`): 2/hard O, 1/soft S,
  1/cation NH. First-shell saturation; MORE IS NOT SAFER (over-water re-adds conformer noise).
- **Protonation = ModelSEED ChemAxon** (`microspecies.py protonation()`; compound_*.tsv
  charge+smiles). Dimorphite REJECTED (59% agreement, fails sulfate/polyamine/polyphosphate;
  uninstalled). ATP/TTP/PPi=−3, UDP-glucose/Glc-1-P=−2, fructose=0(furanose).
- **ABANDONED**: occupancy self-selection (step7/7c/8 deleted) — noisy + irrelevant to ΔG.
- Reactions written with H+ term (`n_Hplus`: + released, − consumed, × G_HPLUS≈−1170.7).

## HOW TO RUN
```
UMA=/homes/rzhu/miniforge3/envs/uma/bin/python
# one reaction per GPU, background:
CUDA_VISIBLE_DEVICES=0 setsid nohup $UMA scripts/unified_pipeline.py --only redox > logs/x.log 2>&1 </dev/null &
# reactions: redox glycosyl nucleotidyl glycosyl_00605 glycosyl_01713 (add rxn01834)
# env knobs: SAMPLE_SCALE (pool×seeds), N_EXPLICIT_SEEDS, EXPLICIT_KEEP
```
- Launch jobs in BACKGROUND and keep working (don't sit in sleep loops). ≥8 GPUs, ~30GB
  free each; other users use <3.5GB. Batch relax + run reactions on separate GPUs.
- ModelSEED data: `../../../ModelSEEDDatabase/Biochemistry/{compound,reaction}_*.tsv`.
  Experimental ΔG: `../../pipeline/TECRDB.csv`, per-rxn exp in `pipeline/bench226_scored.json`
  (field `e`; but user says bench226 is just aggregated prior results — use for exp lookup).

## FILES (scripts/)
- `unified_pipeline.py` — THE pipeline (REACTIONS dict; implicit_G / explicit_G / run_reaction).
- `thermal_solv.py` — corr_fast, uma_gibbs_corr (batched Hessian), xtb_dgsolv[_relaxed].
- `water_count.py` — coordination water rule. `microspecies.py` — ModelSEED protonation.
- `batched_relax.py` — batched UMA FIRE relax + energies (r_data_keys=["spin","charge"]!).
- `step7b_charge_balanced_waters.py` — bare_geom(). `step4e_targeted.py` — pool_confs, boltz.
- diagnostics: `diag_fructose.py`, `validate_split_corr.py`, `validate_dimorphite_vs_modelseed.py`.

## Repo: thermodynamic_calc/ (own git, remote qm_metabolites_reaction_energy, master, pushed
through cdbcc72). Memory: [[unified-qm-pipeline]], [[explicit-water-count-calibration]].
