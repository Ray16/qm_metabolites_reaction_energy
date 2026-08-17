# HANDOFF — QM reaction-ΔG pipeline (read PIPELINE_REFERENCE.md + EXPLORATION_LOG.md for depth)

## RUNNING NOW (survives session end)
- **pH-0 full-367 pass** (`tools/ph0_worker.sh` + `launch_ph0.sh`, ~37 GPUs lambda0/1/5/6): AUTO_TRUNCATE
  (v1) + PH0_AUTO on all 367 -> logs/ph0_sweep/. **~101/367 done** when this was written. Verified
  CORRECT (reproduces hand-validation: rxn00695 err +0.3, rxn10427 +4.8, n_H+=0). Completion monitor
  was set at >=360 done. lambda5 T4s OOM on big cofactors + circuit-break (self-heals: big rxns go to
  V100; relaunch idle lambda5 workers with `launch_ph0.sh` if throughput drops).
- Baseline sweep (implicit-anion) already COMPLETE in logs/full367/ (365/367; 2 carnitine failures).

## FIRST THING NEXT SESSION — when pH-0 pass completes
A background poller (`tasks/bhcqgcw06`) fires when >=360 reactions have a final ΔG (or on a ~1h stall).
Steps 2 & 3 below are DONE (2026-08-16); remaining is the final analysis + σ calibration.
1. `python tools/ph0_final_analysis.py`  -> the GATED coherent MAE vs baseline vs TECRDB, per class.
   Partial (121 done, 2026-08-16): baseline 25.7 -> **gated coherent 12.5 kJ** (huge/floppy 45.9->11.9,
   clean 21->14, anion 57->27; will settle as hard rxns finish). |err|<10 at 51%.
2. [DONE] **Isomerase gate WIRED** into unified_pipeline PH0_AUTO block (`is_isomerization` skip).
   BONUS: added an **H-mass-balance guard** to `build_ph0_reaction` — refuses pH-0 (->baseline) when the
   neutralised reaction is not H-balanced at n_H+=0. This is what produced the +-1150 kJ "garbage": all 21
   were net-proton-exchange (NAD(P)/GSH redox, deamination) where forcing n_H+=0 drops one proton (~1170 kJ).
   Guard refuses all 21, keeps the validated phosphate class, and the 4 previously-"good" refused rxns had
   baseline==pH0 (pH-0 was a no-op) -> ZERO accuracy cost. So step 3 (re-run garbage) is now AUTOMATIC.
3. [SUBSUMED by the guard] The 21 pH-0 "garbage" now self-route to baseline. Only genuine BASELINE
   sampling-fails remain (rxn01407, rxn01725 carnitine/NADH_t) — low priority, orthogonal to pH-0.
4. **Calibrate uncertainty.py SIGMA_CLASS** from the final per-class residuals (esp. `anion` 47->~18).

## CODE REORG (2026-08-16, committed)
- `scripts/` now holds ONLY the 12 production modules; 21 exploration one-offs moved to
  `backup/qm_exploration_scripts/` (gitignored, in git history). See `README_MODULES.md` for the map
  and `REFACTOR_PLAN.md` for the deferred hot-path cleanup (split run_reaction, drop dead seeds params,
  rename step4e_targeted->conformers / step7b->explicit_clusters) — DEFERRED until the sweep frees those
  module names; includes a before/after identical-ΔG verification gate.

## WHAT WAS BUILT THIS SESSION (all committed + pushed, master)
- **pH-0 auto-routing fix** (`ph0_auto.py` + PH0_AUTO): neutral-species QM + EXACT Alberty pKa transform
  for the phosphate/NTP anion class. Validated MAE 91.5->18 on 5 phosphate rxns. Key: max-anion
  canonicalization (fixes 61 FRAGILE), full per-group pKa ladders, **n_H+=0** (bug the test caught: the
  charged n_H+ double-counts +1170 kJ). Textbook pKa's, no DB fitting.
- **Isomerase gate** (`ph0_auto.is_isomerization`, formula-bijection, general/no-flags) — validated, in
  code, NOT yet wired (step 2 above). pH-0 helps EVERY class except isomerase.
- **truncate_v2** (`truncate_v2.py` + TRUNC_V2): global-MCS for the 20% (multi-coeff/unequal-side) v1
  refuses. RISKY unguarded (helped 2/3, hurt rxn00065) -> gated by **radius-sensitivity** (validated:
  good cut |ΔΔG|=1.3, bad cut 27.7; `tools/radius_sensitivity.sh`). Strategy = escalate radius until
  ΔG stable. NOT yet integrated as default (measured LOWER priority: after pH-0, full≈trunc MAE ~9.6).
- **uncertainty.py**: calibrated total σ (U_samp + class-σ + motif) for TFA/flux. CHEAP (lookups, no
  extra QM). NEEDS final σ_class calibration. Error is SCATTER not bias -> calibration won't fix it;
  need method (pH-0/truncation) + isodesmic referencing to reduce it.
- **ModelSEED input adapter** (`tools/build_modelseed_reactions.py` -> scripts/reactions_modelseed.json):
  **20,802 runnable reactions** (57x TECRDB). Format + prep verified (truncation/gate/pH-0 all accept it).
  End-to-end GPU run pending free GPUs. exp from GCM deltag (QM-vs-GCM comparison).
- **Loader collision bug FIXED** (build_tecrdb_reactions.py name[:14] collapsed isomer substrate/product
  -> garbage -3.6M kJ; restored rxn01505/rxn03087).
- **Multi-node sweep infra** (lambda-fleet skill + full367_worker/launch_workers): ~37 GPUs, claim-based,
  preflight+circuit-breaker. Node-env gotchas fixed (rdkit lambda5, xtb env lambda5+6). NOT lambda13.

## KEY MEASURED FINDINGS
- **Baseline MAE 29.1 kJ** (excl 4 garbage outliers; median 21.9). By class: Mg/anion ~40-43 (pH-0 target),
  clean 18.7, isomerase 6.6 (UNRESOLVED, ok), huge/floppy 49 (truncation).
- **pH-0 helps every class except isomerase.** Gate = skip isomerase. Gated coherent MAE ~10 (partial).
- **After pH-0, truncated ≈ full-molecule error (~9.6)** -> v2/radius-escalation is LOWER priority than
  thought (pH-0 does the heavy lifting; the huge/floppy conformer noise mostly rides on anions pH-0 fixes).
- **Mg: subsumed by pH-0** (no pMg data; exp_sd absorbs Mg variation; no systematic Mg bias). Mechanism
  detector (phosphoanhydride SMARTS) finds 192 Mg-relevant vs flag's 89 (flag misses 103); 36 precise
  (anhydride created/destroyed). Don't build explicit Mg unless a systematic-bias test demands it.
- **DFT probe: glycosyl floor = REFERENCE ceiling not UMA** (DFT≈UMA +0.7 kJ) -> truncation+DFT DEAD.
- **MCS vs RXNMapper (measured, IN PROGRESS):** neither best alone. RXNMapper unreliable on PHOSPHATE
  (median conf 0.36, n=257) but reliable on non-phosphate (0.83, n=110); handles 43 splits MCS refuses.
  BEST = confidence-gated HYBRID. `truncate_rxnmapper.py` built (works rxn00973; small-frag bug fixed for
  water/hydrolysis). NEXT: finish coverage comparison + GPU ΔG on the differing rxns to set the gate
  EMPIRICALLY (run both — cheap — decide by ΔG accuracy). Maps in artifacts/rxnmapper_maps.json.

## THE COHERENT ROUTER (the goal — mostly built, needs wiring)
Per reaction, auto-route (no manual flags): truncate (MCS+radius-sensitivity, +RXNMapper-hybrid for splits)
-> gated pH-0 (skip isomerase) -> UNRESOLVED flag (|ΔG|<U_samp) -> calibrated σ -> reduced-confidence tag
for glycosyl/thioester (reference ceiling). Wiring = step 2 above + make AUTO_TRUNCATE+PH0_AUTO default.

## NEXT (priority)
1. [pass done] gated analysis + wire gate + re-run garbage + calibrate σ (the 4 first-thing steps).
2. Finish MCS-vs-RXNMapper: coverage + GPU ΔG on differing rxns -> empirical hybrid gate.
3. Isodesmic/reference-reaction referencing = the untapped accuracy multiplier (scatter-dominated error).
4. ModelSEED demo run (coherent pipeline on a sample of the 20,802, QM vs GCM).
Everything pushed to Ray16/qm_metabolites_reaction_energy master.
