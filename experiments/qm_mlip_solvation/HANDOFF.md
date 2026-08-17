# HANDOFF — QM reaction-ΔG pipeline (read PIPELINE_REFERENCE.md + EXPLORATION_LOG.md for depth)

## STATE (2026-08-17)
- **pH-0 full-367 sweep COMPLETE** (364/367 in logs/ph0_sweep/; 3 stragglers won't finish, fine).
  Baseline in logs/full367/. Redox 94-run in logs/ringcofactor/ (89/94).
- **FULL PIPELINE accuracy (pH-0 + COFACTOR_RING + truncation), 361 rxns:**
  baseline 29.1 -> pH-0-gated 20.0 -> **+cofactor-ring 15.3 kJ MAE** (median ~10, **bias ~0 = unbiased,
  scatter-limited**). Redox class 35.5 -> 16.5 via the ring (pH-0 alone does NOTHING for redox).
- **vs dGPredictor (retrained), 354 rxns:** UMA MAE 14 / med 10 vs dGP MAE 5 / med 2. UMA NOT yet
  competitive on TECRDB (expected — dGP is trained on it); UMA's value is frontier coverage. See the
  error histogram `figures/error_histogram_uma_vs_dgp.png` (tools/make_error_histogram.py).

## THE >20 kJ TAIL = the roadmap (26% of rxns, 92/357; tools + worst offenders in EXPLORATION_LOG)
By class in tail: **thioester 55%** (CoA still uncored), **glycosyl 50%** (electronic ceiling),
huge/floppy 26%, clean 24%, isomerase 10%, anion 17%. Mechanistic drivers:
- **CoA thioesters** (acetaldehyde DH etc.) — the full CoA rides along like NAD did.
- **Flavin redox** (dihydroorotate DH +92) — COFACTOR_RING is nicotinamide-only, no FAD yet.
- **Phosphagen kinases** (taurocyamine/lombricine, P-N + Mg).
- **Glycosyl** (orotate PRTase +49) — N-glycosidic electronic ceiling.
- **Reductive-amination DHs** (alanine/alanopine DH) — NAD redox + C=O->C-NH2 double transform.

## FIRST THING NEXT SESSION — the path forward (priority)
1. **Extend canonical cofactor cores to FAD + CoA** (`scripts/cofactor_truncate.py` COUPLES table — one
   row each, same recipe as the validated nicotinamide/cysteine). Biggest lever: attacks the thioester
   (55%) + flavin-redox tail; FAD/CoA are among the most common cofactors (free at scale). Validate like
   the ring-cofactor (build reactions_*.json, GPU run, compare vs logs/ph0_sweep).
2. **Glycosyl electronic ceiling** -> DLPNO-CCSD(T) on the truncated core (affordable now).
3. **Mg-phosphagen kinases** -> the P-N + Mg sub-class.
4. **Calibrate uncertainty.py SIGMA_CLASS** from the final per-class residuals.
5. Re-run sweep with COFACTOR_RING=1 (now default in ph0_worker.sh) for a clean single aggregate.
6. General localizer (localize.py) -> symmetry-robust atom mapping to retire the curated table (GENERALITY.md).

## DONE THIS SESSION (2026-08-16/17, all committed+pushed, master)
- **pH-0 wired + guarded**: isomerase gate + H-mass-balance guard in unified_pipeline/ph0_auto (the guard
  refuses net-proton redox/deamination -> baseline, killing the +-1150 kJ garbage at zero cost).
- **COFACTOR_RING** (`cofactor_truncate.py`, table-driven canonical cores): NAD(P) ring + GSH cysteine-thiol,
  couples COMPOSE (glutathione reductase +34.7->+19.2 auto). Redox class 35.5->16.5. Now default in worker.
- **localize.py** general MCS localizer (works on substrates, fails on symmetric cofactors -> why table needed).
  **GENERALITY.md** = path to a fully general localizer.
- **glycosyl reclassification** (is_glycosyl_transfer): anion class was contaminated by phosphoribosyltransferases;
  pure anion is 5.6 kJ, glycosyl (electronic ceiling) 26.
- **truncate.py naming-scramble bug FIXED** (pair_by_mcs reorder -> mislabeled cores in logs; ΔG unaffected).
- **DECK** `slides/pipeline_talk.pdf` (~15 slides): pipeline story + slide 12 = dGPredictor head-to-head with
  localized reaction schemes (A+B<=>C+D, gold-highlighted reacting bonds via reasoned per-type SMARTS) +
  per-class MAE + before/after scatter. Figures via make_deck_figures.py, plot_comparison_schemes.py,
  update_top10_json.py, make_error_histogram.py. KEEP UPDATED on new results (memory: update-slides-on-new-results).
- Code reorg: 21 one-offs -> backup/qm_exploration_scripts/; README_MODULES.md; REFACTOR_PLAN.md.

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
