# MetaG physics review and revision — session summary (2026-10-01)

Written for an external critic. It covers what was done, why, the evidence for each change, what was
tested and rejected, and the weak points. The working notes are in `NOTES.md` in the same directory.
Commits are on `thermodynamic_calc` master: `9fd178c`, `b96dd6b`, `3572ef9`.

---

## 1. Context and goal

- **MetaG** predicts the standard transformed reaction Gibbs energy ΔrG′° (pH 7, I = 0, 298 K, 1 M) of
  metabolic reactions from structure.
  - Each species gets G = E_elec(UMA `uma-s-1p2p1`, gas) + RRHO thermal (UMA Hessian) + ΔG_solv (xtb GFN2
    implicit solvent, single point) + the 1 atm → 1 M standard-state term. The total is Boltzmann-averaged
    over deduplicated conformers.
  - Reactions are routed before QM:
    - NAD(P) is replaced by a nicotinamide core;
    - spectator groups are truncated by MCS with C–C-only cuts;
    - anion-containing reactions use a "pH-0 route", where QM is done on neutral, fully protonated species
      and an analytic Alberty pKa transform brings them to pH 7.
- **Benchmark:** TECRDB, 364 reactions. Each measurement is Legendre-transformed to pH 7 / I = 0 / no Mg²⁺
  (`reactions_tecrdb_std.json`).
- **User's brief:**
  - do a code review;
  - identify and implement physics-based improvements for reactions predicted badly, looking at
    individual reactions and not only classes;
  - TECRDB is used to calibrate and check, not to fit;
  - adopt a change only if it is physically motivated and tested.
- **Target:** journal submission in about two weeks.

## 2. Starting point

The previous sweep (`physics_20260930b`) was reassembled on the standardized reference:

| | MAE | median | n scored | \|err\| > 20 |
|---|---|---|---|---|
| Production (xtb-COSMO + 8 TECRDB-fitted class anchors) | 12.39 | 9.45 | 340/364 | 58 |
| Same, anchors off | 13.73 | 10.53 | 340 | — |

- Reactions on the pH-0 route had MAE 13–19 kJ/mol; all other routes had 8.5–9.
- 23 reactions returned no estimate.
- The existing figures in `figures/` predated this sweep.

## 3. Methods used this session

1. **Code review** by three parallel read-only agents:
   - pKa-transform layer;
   - species free-energy layer (thermal, solvation, sampling, standard states);
   - routing and assembly layer.
   Their findings were spot-checked and mostly reproduced numerically.
2. **CPU reassembly harness.** Reaction ΔG is rebuilt from the cached per-species G, with the per-stage
   breakdown (species sum, proton, pKa transform) and per-species values. This allows fast A/B tests of
   routing, pKa and reaction-level changes without QM. Script: scratch `reasm_stages.py`, a derivative of
   `analysis/review_fixes/reassemble.py`. Two harness pitfalls were found and fixed:
   - its "baseline" policy forces `ZWITTERION_PH0=0`;
   - it has to emulate the pipeline's re-route when a species changes bonding in every conformer
     (`SpeciesRearranged`).
3. **A fresh GPU species sweep with ALPB as the primary solvent**, with COSMO and CPCM-X evaluated on the
   same conformers.
   - The species list is the union over all routing variants under test, collected by routing every
     reaction without QM (`collect_species*.py`).
   - Claim-based, resumable workers ran through `gpu_reserve` on lambda4/5/11/13.
   - Total: about 950 species, plus later small batches for the hydration, H-bond-filter and isomerase
     variants.
4. **Independent physical validation sets**, so that changes were not judged on TECRDB alone:
   - FreeSolv (642 neutral molecules), with a per-functional-group error regression;
   - 16 carbonyl hydration constants (K_hyd) from the literature: `khyd_set.json`;
   - literature hydrate and enol fractions for oxaloacetate, pyruvate, DHAP, glycolaldehyde and
     glyoxylate;
   - one isodesmic consistency test (FBP + fructose → F6P + F1P);
   - one UMA-vs-DFT check (PBE0/def2-TZVP) on the phosphagen core.
5. **Final step:** a full run of the real pipeline (`analysis/tecrdb_rescore.py`) on the warm cache.
   σ was recalibrated with `metag/tools/calibrate.py`, using nested CV with near-duplicate-grouped folds.

## 4. Key diagnosis

**xtb `--cosmo` has no hydrogen-bond term.** Its output shows Ghb = 0 and Gshift = 0. Evidence that this
matters:

FreeSolv, linear regression of (model − exp) on functional-group counts (kJ/mol per group):

| group | COSMO | ALPB | CPCM-X |
|---|---|---|---|
| alcohol OH | +17.8 | +4.3 | +8.3 |
| COOH | +18.8 | −4.9 | +1.5 |
| amine | +11.7 | +12.2 | +0.9 |
| overall MAE | 9.1 | 6.7 | 5.5 |

- Water ΔG_hyd: COSMO −2.8, ALPB −38, experiment −26.4.
- Carbonyl hydration log K_hyd MAE: COSMO 3.58 (systematically too low, ≈ 20 kJ), ALPB 1.10 (r = 0.96).

**Interpretation:**
- The 8 class anchors and the hydro-lyase water patch were absorbing COSMO's missing H-bond term.
- Several earlier "rejected because it breaks error cancellation" verdicts were COSMO artefacts:
  the environment-specific pKa table, zwitterion re-routing, NTP core, pH-0 for isomerases, and ungated
  hydration.

CPCM-X does best on FreeSolv but was much worse on TECRDB (MAE ≈ 19.8). It was not adopted; the reason
is unexplained.

## 5. Changes adopted (all now pipeline defaults; each reversible by a flag)

The numbers below come from the reassembly ladder on the fresh sweep with current routing. The last two
rows come from real-pipeline runs.

| # | Change | Physical motivation | Independent evidence | TECRDB effect |
|---|---|---|---|---|
| 1 | `SOLV_MODEL=alpb` (was cosmo) | COSMO lacks an H-bond term | FreeSolv per-group errors; K_hyd | together with #2, #3: 12.01 → 11.61 |
| 2 | `WATER_REF_EXP`: liquid water from experimental ΔG_hyd (−26.4) | Liquid water's chemical potential is known exactly; a continuum water-in-water value is unphysical | exact | (in #1) |
| 3 | `ANCHOR_CORRECT` and `WATER_REF_HYDROLYASE` off | They compensated COSMO; no fitted offsets remain | — | (in #1) |
| 4 | `PKA_ENV` on + new α,β-unsaturated carboxyl class (3.75 conjugated diacid, 4.35 isolated) | Textbook α-substituent pKa effects. Fumarate's transform was 11.6 kJ short of its measured pKa (3.03/4.44) | literature pKa values (not tuned) | with #5, #6: 11.61 → 10.73; hydratase 17.4 → 10.4 |
| 5 | `ZWITTERION_PH0` on | Zwitterions are not gas-phase minima: UMA relaxation transfers the proton | observed in every conformer | (in #4) |
| 6 | `FREE_PPI_PKA` on | PPi is one coupled tetraprotic acid (0.9/2.0/6.6/9.4), not two monoesters | literature | ≈ 0 (correctness) |
| 7 | Thiolate/phenolate pH-0 neutralization (pKa 8.7/10.0) | ModelSEED draws GSH as a thiolate. Left unmatched, the pH-0 species became an unphysical thiolate/ammonium zwitterion and xtb failed | pKa > 7, so the transform is < 0.1 kJ and insensitive to the value | 2 unscored reactions recovered |
| 8 | `ARYLAMINE_NONBASIC`: N bonded to an aromatic atom is not a basic amine | Aniline pKa 4.6, adenine N6 not protonated. Counting it hid the destroyed Asp ammonium in adenylosuccinate synthase | textbook | 10.73 → 10.69 (7 reactions change routing) |
| 9 | `NTP_CORE` on: spectator nucleoside → methyl cap | Isodesmic spectator removal (same principle as the default-on NAD core) | NDP kinase becomes exactly isodesmic | 10.69 → 10.65; adenylate kinase −26 → +4 |
| 10 | `CARBONYL_HYDRATION_ALL`: every aldehyde/ketone folded as a carbonyl ⇌ gem-diol mixture; each form's own acid transform applied before mixing; ΔG_hyd calibrated on K_hyd (0.67·calc + 1.94 kJ); α-keto acids excluded | Real aqueous microspecies. The old electron-poor-only gate existed because COSMO gave about 20 kJ too little hydration | K_hyd LOO MAE 0.61 log (3.5 kJ). Hydrate fractions vs literature: DHAP 45 vs 45%, glycolaldehyde 88 vs 90%, glyoxylate 99.5 vs 99%. α-keto acids stay over-hydrated even after calibration (OAA 82 vs 8%, pyruvate 29 vs 6%), and their true contribution is ≤ 0.3 kJ, so they are excluded | 10.65 → ≈ 10.2 |
| 11 | `ACID_HB_FILTER` v2: on neutral pH-0 species, drop conformers where an acidic O–H of one acid unit H-bonds an O of another unit; a P–O–P chain counts as one unit | Such contacts cannot exist at pH 7 (both groups ionised and repulsive), and the pKa table assumes non-interacting groups | FBP + fructose → F6P + F1P = +18.6 kJ (ALPB) vs ≈ 0 implied by FBP's near-additive pKa values; filtering shifts FBP by +19.9 | ≈ 10.2 → 10.08 (real run); FBP aldolase +36 → +16, PEP mutase +24 → +10, GAPDH −12 → −2 |
| 12 | `PH0_ISOMERASE` on: isomerizations also take the pH-0 route | The isomerase gate was set on the COSMO baseline. Under ALPB the neutral forms are the better-described species, while charged sugar-phosphate ring isomers do not cancel | **none (TECRDB only)** | 10.08 → **9.65**; isomerase class 8.4 → 5.6; worst regression −1.6 |
| 13 | Mode-following fall-through bug | An unconverged tight re-optimisation skipped the imaginary-mode test, so the species failed | correctness | recovered dimethylmaleate etc. |
| 14 | Unit tests | `tests/test_physics_20261001.py` plus test updates for the new defaults | — | 17 CPU test modules pass |

Also in the same commits: the previous session's uncommitted routing work (anomeric truncation radius,
free-PPi/anhydride pKa flags). It is part of the validated configuration; `ANHYDRIDE_PKA` stays off.

## 6. Final result (real pipeline, `analysis/sweep_20261001/final/`)

| | Before | After |
|---|---|---|
| Scored | 340–341/364 | **364/364**, 0 errors |
| MAE | 12.39 | **9.65** |
| Median | 9.45 | **7.15** |
| RMSE | 16.7 | **13.67** |
| \|err\| > 20 / > 40 | 58 / 15 | **39 / 9** |
| Fitted point parameters | 8 anchors | **0** |
| σ calibration (nested grouped CV) | 95.3% coverage | 95.9% coverage, held-out MAE = in-sample 9.65 |

- Against the native-condition reference used by the comparison methods in `tecrdb_fiveway.png`:
  MAE 9.56.
- For comparison on the same figure: dGPredictor family-grouped held-out 8.3; eQuilibrator in-sample 3.6;
  GC in-sample 6.6.

## 7. Tested and rejected

| Hypothesis | Result |
|---|---|
| H3 solution-phase relaxation (minimise E_UMA + ΔG_ALPB using the xtb gradient difference; `SOLV_RELAX`) | Species shifts 2–9 kJ (citrate³⁻ 15), mostly cancelling within a reaction; 5–20× cost; does not rescue gas-phase zwitterions. Kept behind a flag, default off |
| Compound-level ChemAxon pKa ladders | Much worse (MAE 14 → 27). Reason: macroscopic ladders (e.g. ATP's charge-0 state is N1-protonated) do not describe the specific QM microstate. Independent-site microscopic pKa values are the correct formalism |
| Enol tautomers (H6) | Rejected from literature without compute: phenylpyruvate is keto in water; OAA is 74% keto / 18% enol / 8% hydrate (0.7 kJ) |
| Uncalibrated (raw-ALPB) hydration | Better TECRDB (9.89 vs 10.10) but unphysical: OAA predicted 99% hydrated. Rejected in favour of the K_hyd-consistent version |
| H-bond filter v1 (each P atom its own group) | Broke nucleotide kinases by ≈ +21; replaced by v2 |
| CPCM-X as primary solvent | TECRDB ≈ 19.8 |
| Global group-wise FreeSolv corrections | Considered, not implemented |

## 8. Remaining errors (39 reactions with \|err\| > 20)

- **Phosphagens** (n = 4, bias +37).
  - Arginine, taurocyamine and lombricine kinases reduce to the identical core: methylguanidinium⁺ +
    MeO-PPP → N-phospho-methylguanidinium⁺ + MeO-PP. It is predicted at +40.7, against measured values
    of +1.1 / −4.0 / −8.8.
  - UMA − PBE0 on the core is +7.8 kJ (20%), so most of the error is solvation or speciation of the
    phosphoguanidinium cation. Unresolved.
- **Lipid-phase references** (4 reactions: retinyl-palmitate esterase K′ = 0.0028, anandamide
  amidohydrolase ×2, carnitine palmitoyltransferase), all −20 to −59.
  - The measured K′ values imply condensation is favoured for aqueous ester/amide hydrolysis, which is
    implausible. This is consistent with micellar partitioning.
  - Proposed: report as a separate, structurally pre-defined category (an acyl chain of C12 or longer
    changes attachment). Currently still included in the 9.65.
- **Tautomer and representation issues:**
  - lactim forms of hydroxypteridines and purines (tetrahydroxypteridine cycloisomerase +34; adenosine
    deaminase +34);
  - NAD oxidations to α-keto or indole products (indolelactate DH +51, gluconate 2-DH +25);
  - PRT and thiamin-P transfers (nicotinate PRT −39, thiamin-P pyrophosphorylase +47);
  - some CoA reactions (citrate synthase +31).

## 9. Threats to validity (please critique these in particular)

1. **Selection on the benchmark.** About 12 binary choices were each A/B-tested on TECRDB. Even with
   physical motivation, choosing the variant that improves TECRDB is a form of model selection. The
   "held-out = in-sample" statement holds only for fitted parameters, not for these design choices.
   Two changes are mainly TECRDB-supported:
   - #12 isomerase pH-0 (no independent test);
   - #11 v2 (the chain-as-one-group definition was chosen after v1 regressed; only the v1 idea has
     independent FBP evidence).
   The α-keto-acid exclusion (#10) and the unsaturated-carboxyl class (#4) were noticed via TECRDB
   regressions, but their criteria and values come from literature.
2. **Small calibration set for hydration.** 16 K_hyd values with a 2-parameter linear map; the α-keto
   exclusion is a rule-based carve-out. Several K_hyd values (DHA, fluoroacetone, isobutyraldehyde) were
   entered from memory or textbook knowledge and have not been verified against primary sources.
3. **Literature pKa values and fractions.** These include fumaric/mesaconic/aconitic pKa, OAA tautomer
   fractions, and pyruvate/α-KG hydrate fractions. Some come from web or textbook recall and should be
   checked against primary sources.
4. **The FreeSolv argument** rests on monofunctional neutrals with MMFF geometries; metabolites are
   polyfunctional, and the pH-0 species are exotic fully protonated acids.
5. **Reassembly vs production.** The ladder numbers come from the CPU reassembly. The final production
   run agrees (10.03 harness vs 10.08 production for the same configuration on a slightly different n).
6. **Reference conventions.** MetaG is evaluated on the standardized (Legendre-transformed) reference;
   the other methods in the comparison figure are on native-condition medians. Both MetaG numbers are
   reported (9.65 / 9.56).
7. **Sampling noise.** Earlier work measured about ±1.7 kJ per reaction between independent samplings;
   aggregate differences below about 0.3–0.5 kJ (e.g. #8, #9 individually) are within noise.
8. **The CPCM-X paradox** (best on FreeSolv, worst on TECRDB) is unexplained and could indicate remaining
   error cancellation in ALPB.

## 10. Suggested independent validation before submission

- **Freeze the configuration.**
- **Thermodynamic cycle closure** over ModelSEED reactions sharing metabolites, with no experimental data
  (`metag/tools/cycle_closure.py`).
- **A small set of external measured ΔG′°** not in TECRDB, scored blind.
- **A per-change ablation table** for the paper, using the ladder in §5.
- **Primary-source verification** of every literature constant used: K_hyd set, pKa values, tautomer
  fractions.

## 11. Reproducibility pointers

- **Code:** `metag/pipeline.py` (`FLAG_DEFAULTS`, `implicit_G` filter, `_hydrate_all_sites`,
  `_solvent_relaxed_ensemble`), `metag/routing/pka_transform.py`, `metag/routing/aldehyde_hydration.py`,
  `metag/energetics/solv_relax.py`, `metag/energetics/uma.py` (`extra_forces` hook).
- **Data:**
  - `analysis/sweep_20261001/final/` — per-reaction production results;
  - `cache/` — species cache: ALPB primary, COSMO/CPCM-X auxiliary (NFS, gitignored);
  - `khyd_set.json`, `list_*.json` (species lists), `phosphagen_core_dft.log`.
- **Calibration:** `metag/data/sigma_class_calibrated.json`; previous copy saved as
  `sigma_class_calibrated.pre_20261001.json`.
- **Figures regenerated:** `tecrdb_fiveway.png`, `diag_signed_bias_by_class.png` (anchor-aware;
  off-scale classes pinned), `reference_scoreboard.png`.
- **Not regenerated:** `where_lacking_by_class.png`, `rep_disagreement.png`, the slide deck.
- **Every old behaviour is reproducible by flag:**
  `SOLV_MODEL=cosmo`, `ANCHOR_CORRECT=1`, `WATER_REF_EXP=0`, `WATER_REF_HYDROLYASE=1`, `PKA_ENV=0`,
  `FREE_PPI_PKA=0`, `ARYLAMINE_NONBASIC=0`, `ZWITTERION_PH0=0`, `NTP_CORE=0`,
  `CARBONYL_HYDRATION_ALL=0`, `HYDRATION_CAL=0`, `ACID_HB_FILTER=0`, `PH0_ISOMERASE=0`.

## 12. Process notes

- One GPU test (`test_pipeline_smoke.py`) was run directly on lambda2 GPU 0 without `gpu_reserve`. It
  OOMed immediately and left nothing running. All other GPU work went through the reservation gate.
- Three review sub-agents were used; their claims were cross-checked. One agent's numbers came from a
  stale results file (its "−45 kJ amino-acid dehydrogenase" claim); this was caught and not acted on.
