# Response to the review of the 2026-10-01 MetaG physics revision

This document answers the review point by point. For each item it gives what was changed, the evidence,
and the files to check. It supersedes the corresponding statements in `SESSION_SUMMARY_20261001.md`.
- Evidence and commands: `provenance/MANIFEST.md`.
- Commits: on `thermodynamic_calc` master, after `98c6dc4`.

## Headline numbers (final configuration)

The final configuration is the real pipeline on 364 standardized TECRDB reactions (pH 7, I = 0, no Mg²⁺),
`final/`. All records share one source hash and one configuration; there are no errors or no-estimates.

| | Value |
|---|---|
| **TECRDB development MAE** | **9.59 kJ/mol** (median 7.10, RMSE 13.41, bias −0.41) |
| \|err\| > 20 / > 40 kJ | 36 / 8 |
| Selection-adjusted MAE (nested policy-selection CV, §1) | 9.80 ± 0.06 |
| Cycle-closure RMS non-closure (49 TECRDB cycles) | 1.47 kJ (experimental values: 3.50) |
| σ: 95% interval coverage, nested grouped CV within TECRDB | 95.6% |

For comparison, the revision as reviewed scored 9.65, and the original COSMO + anchors production 12.39
(341 scored).

---

## 1. "9.65 is a development-set result, not held-out performance" — agreed; relabelled and quantified

- **Relabelled.** Every MAE in the documents is now a **TECRDB development MAE**.
  - In the calibration artifact, `cv_heldout_MAE` now carries `cv_heldout_MAE_note`. The note states it is
    held out only with respect to per-fold refit parameters, which no longer exist. It is therefore equal to
    the development MAE and is not a generalization estimate.
  - In `figures/tecrdb_fiveway.png`, MetaG's row now reads "DEVELOPMENT (no fitted params)" instead of
    "FAIR (never fit)".
- **Nested policy selection** (`run_policy_grid.sh`, `nested_policy_cv.py`, `nested_policy_cv.json`):
  - All 2⁷ combinations of the seven benchmark-selected switches were scored: `PH0_ISOMERASE`, `NTP_CORE`,
    `CARBONYL_HYDRATION_ALL`, `PKA_ENV`, `ZWITTERION_PH0`, `ARYLAMINE_NONBASIC`, `ACID_HB_FILTER`.
  - Inside each training fold the best policy is re-chosen, then scored on the held-out fold.
  - Folds are near-duplicate groups (240 groups); 5 folds × 20 repeats; common set n = 359.
  - Result: **selection-adjusted MAE 9.80 ± 0.06** vs 9.75 for the fixed final policy on the same set, i.e.
    the selection optimism is ≈ 0.05 kJ.
  - The grid uses `THERMAL_ENSEMBLE=0`, because the alternative policies' species exist only in that cache
    generation. The thermal estimator is orthogonal to these routing switches.
- **Per-rule sensitivity** (flip one switch from the final policy; bootstrap 95% CI on ΔMAE; same file):

  | switch | final | ΔMAE if flipped | 95% CI | reactions changed > 1 kJ |
  |---|---|---|---|---|
  | PKA_ENV | on | +0.99 | [0.53, 1.44] | 75 |
  | PH0_ISOMERASE | on | +0.41 | [0.06, 0.77] | 30 |
  | CARBONYL_HYDRATION_ALL | on | +0.36 | [0.06, 0.68] | 54 |
  | NTP_CORE | on | +0.21 | [−0.02, 0.51] | 20 |
  | ARYLAMINE_NONBASIC | on | +0.04 | [−0.05, 0.20] | 4 |
  | ACID_HB_FILTER | **off** | −0.08 if turned on | [−0.27, 0.09] | 11 |
  | ZWITTERION_PH0 | on | 0.00 | — | 0 (kept as an invariant guard) |

  The full-data optimum would set `ACID_HB_FILTER` **on** and `ARYLAMINE_NONBASIC` **off**. The final
  configuration does the opposite on physical grounds (item 6), i.e. it was not chosen to minimise TECRDB.
- **Not done (stated as a limitation): an external, frozen-model dataset.** TECRDB is the only experimental
  ΔrG′° set used. The ground-truth-free complement is cycle closure (Priority 6 below), so far only within
  TECRDB.

## 2. "Documented conformer thermodynamics does not match the implementation" — agreed; tested and replaced

- **The description was wrong.** The old code Boltzmann-averaged E_UMA + ΔG_solv, then added one RRHO
  correction from the lowest-G true minimum.
- **New estimator `THERMAL_ENSEMBLE` (now the default):**
  - RRHO G_corr is computed for every unique minimum within 15 kJ of the lowest (at most 10 per species).
  - The Boltzmann sum is over G_i = E_i + ΔG_solv,i + G_corr,i.
  - Auxiliary solvent models use the same per-minimum G_corr.
  - The cache key carries the mode; `THERMAL_ENSEMBLE=0` reproduces the old estimator.
  - Docstring of `implicit_G` and the flag comment updated.
- **Measured on all 537 species and 364 reactions** (`thermal_ensemble_test.json`):

  | | single minimum → per-minimum ensemble |
  |---|---|
  | species shift | −0.75 ± 1.55 kJ (−9.2 … +2.2); G_corr spread across minima median 3.6 kJ, max 16 kJ |
  | reaction ΔG shift | −0.89 ± 2.27 kJ; 152 reactions > 2 kJ, 17 > 5 kJ, max 8.0 (PRTs, transketolase, polyol DHs) |
  | TECRDB MAE | 9.67 → 9.59 |

  Not negligible for individual floppy reactions, neutral in aggregate. It was adopted as the correct
  estimator, not for accuracy.
- **Remaining approximation, documented in `_thermal_ensemble`.** Imaginary modes above the tolerance are
  floored as soft modes for minima other than the reference; there is no per-minimum saddle search.
- **σ and `U_samp` limitation (stated in §8).** `U_samp` is still the spread of the ensemble G over the last
  seed batches. Thermal variation now enters G itself but not a separate uncertainty term.

## 3. "Explicit-solvation caches can collide across water-reference physics" — agreed; fixed

- `_EXPLICIT_SETTINGS` (a module-level dict reading the raw `WATER_REF_EXP` env string, default `"0"`) is
  replaced by `_explicit_settings()`.
- The new function builds the key at call time from effective values: `bool(_flag("WATER_REF_EXP"))`
  (including its `FLAG_DEFAULTS` default), `SOLV_MODEL` and `float(WATER_DGSOLV_KJ)`.
- **Regression test:** `test_explicit_cache_key_tracks_effective_water_reference`. It asserts that the
  default key ≠ the `WATER_REF_EXP=0` key, and that default = explicit `=1`.

## 4. "Carbonyl hydration is not the exact multi-site fold it claims" — agreed; replaced by exact enumeration

- `hydration_states(smi)` enumerates every non-empty subset of hydratable sites (all 2ⁿ − 1 hydrated
  microstates, n ≤ 3). Each state's energy is computed, and each carries its own acid transform on the pH-0
  route.
- `mixture_G_states` folds **only computed states**:
  G_eff = −RT ln[e^(−G_c/RT) + Σ_S e^(−(G_S − n_S·G_water)/RT)].
- The independent-site product, which implicitly included uncomputed doubly hydrated states, is removed.
- The K_hyd calibration is applied per hydration event.
- Sites are taken in **canonical-rank order** (`Chem.CanonicalRankAtoms`), so the result does not depend on
  input atom order. Molecules with more than 3 sites are truncated to the first 3 canonical sites, and a
  warning is recorded in `routes.warnings`.
- In TECRDB: 4 species have 2 sites, none has ≥ 3. Their double hydrates (e.g. methylglyoxal bis-hydrate)
  were computed.
- **Tests:**
  - `test_hydration_states_exact_enumeration` (methylglyoxal: 3 states incl. the bis-hydrate; one state
    reduces to the two-state formula);
  - `test_hydration_sites_independent_of_atom_order` (a 3-carbonyl molecule written two ways → identical
    7 states).

## 5. "pKa layer mixes microscopic site constants with macroscopic ladders" — partly agreed; scope stated, constants fixed

- **Scope (agreed).** The transform is a pH-7 effective treatment and is documented as such
  (`POLYACID_PKA` comment): MetaG's estimand is ΔrG′° at pH 7, not a titration model.
- **The formalism error is negligible at pH 7.** For nine polyprotic acids with critically evaluated
  macroscopic constants, the independent-site sum and the coupled binding polynomial agree to ≤ 0.01 kJ at
  pH 7. Test: `test_recognized_polyacids_use_compound_constants` (citrate).
- **What was actually wrong were the constant VALUES.**
  - Errors of the old rule table: oxalate −22.7 kJ, malonate −5.6, maleate +3.8, succinate/adipate +1.9,
    tartrate −1.1.
  - The previous fumarate 3.75/3.75 "effective" pair is replaced by compound constants.
- **Fix:**
  - `POLYACID_PKA` holds compound-specific macroscopic ladders from Martell & Smith (*Critical Stability
    Constants*, I = 0, 25 °C) for fumaric, maleic, malic, succinic, citric, tartaric, malonic, oxalic and
    adipic acid. They are used whenever the whole species is one of these acids; the rule table is the
    fallback. Switch: `POLYACID_PKA=0`.
  - The free-PPi ladder remains as before.
- **Microscopic constant where the QM microstate requires it.**
  - On the pH-0 route an α-keto acid is computed as the **keto** microstate.
  - Its correct constant is the keto-form microscopic pKa: Lopalco et al., J. Pharm. Sci. 2016, NMR, I = 0.15.
    Pyruvic acid keto 1.79 vs hydrate 3.23; two homologues 1.60, 1.68; → 1.8 at I = 0.
  - The macroscopic 2.4–2.5 used before mixed in the hydrate.
  - Effect: development MAE 9.73 → 9.61. Lactate dehydrogenases improved by ~4 kJ each.
  - Amino-acid dehydrogenases and transaminases moved by ~4 kJ the other way, exposing a consistency gap on
    the α-amino-acid side; listed in §8.
- **Not done:** compound-specific constants beyond the nine acids. Oxaloacetic, 2-oxoglutaric, isocitric
  and others lack a verified I = 0 source in hand, so the rule table applies to them.

## 6. "The acid H-bond filter removes physical conformers" — agreed; filter switched off

- `ACID_HB_FILTER` default is now **off**. It is kept as a diagnostic flag; cache key v2 is retained for
  reproducibility.
- The FBP isodesmic artefact (FBP + fructose → F6P + F1P = +18.6 kJ, vs ≈ 0 implied by FBP's pKa's) is
  reported as a known limitation of the neutral-reference construction for multi-acid species (§8).
- A proper fix would be a coupled-group interaction term in the transform, not conformer removal.
- Effect of turning it back on: −0.08 kJ, 95% CI [−0.27, 0.09].

## 7. "Archived uncertainty records do not support the deployed-calibration wording" — agreed; regenerated

- **Order of operations is now correct:** production run → σ calibration → production rerun.
  - The archived `final/*.json` were regenerated after calibration (ΔG identical, max change 0.0 kJ).
  - They carry intervals from the matching artifact.
  - The pre-calibration run is kept as `final_precalibration/`.
- **Wording.**
  - New fields: `coverage_calibrated` and `calibration_basis = "TECRDB internal nested grouped CV (not an
    external dataset)"`.
  - `externally_calibrated` is kept only as a deprecated alias.
  - All scope messages now say "coverage-calibrated".
- **Counts.**
  - 243/364 records are coverage-calibrated.
  - The remaining 121 carry OOD flags or classes outside the calibration set; their intervals are labelled
    nominal.
  - Nested-CV coverage of the 95% interval: 95.6%. In-sample coverage of the archived intervals: 96.7%.

## 8. "Validation evidence is not reproducible from a clean checkout" — agreed; provenance bundle committed

- **K_hyd set rebuilt from cited values only** (`khyd_verified.json`).
  - 16 compounds: recommended 298 K values from the review supplement of acp-2021-58, Tables S3/S4, with
    primary references per compound.
  - The earlier memory-based set is retired as `khyd_set_UNVERIFIED_superseded.json`.
  - The 8 compounds I could not source were dropped: chloral, glyceraldehyde, hexafluoroacetone,
    benzaldehyde, dihydroxyacetone, fluoroacetone, chloroacetaldehyde, isobutyraldehyde.
  - New cited compounds were computed: butanal, pivaldehyde, glyoxal, diacetyl, mesoxalic acid.
- **Results on the cited set** (`khyd_validation.py` → `khyd_validation.json`):

  | | log K MAE |
  |---|---|
  | COSMO | 3.02 (bias −2.8) |
  | ALPB raw | 1.70 (bias +1.6) |
  | ALPB calibrated, LOO | 0.64 |

  - The calibration is now fitted on the **application domain**: the 11 aldehydes, ketones and glyoxylic
    acid the pipeline hydrates.
  - Fit: log K_exp = 0.639 log K_calc − 0.411, i.e. ΔG_hyd = 0.639 ΔG_calc + 2.35 kJ per event.
  - LOO MAE 0.46 log units (2.6 kJ) vs 1.25 raw.
  - The α-keto-acid exclusion is now cited: K_hyd at pH 6–7 is 0.08 for pyruvate, 0.06 for the oxaloacetate
    dianion and 0.12 for the 2-oxoglutarate dianion (≤ 0.3 kJ).
- **Committed** (`provenance/`):
  - FreeSolv v0.52 input (hashed) plus the solvation script and its output;
  - the per-group regression script and output, which reproduces COSMO OH +17.8 / COOH +18.8 kJ;
  - the reassembly harness (`reasm_stages.py` + `provenance/reassemble.py`), shown to agree with production
    to ≤ 0.1 kJ per reaction;
  - the phosphagen DFT log;
  - water references, the policy grid, nested CV, cycle closure, the thermal test, eQuilibrator/GC reruns,
    and the common-reference comparison.
  - `MANIFEST.md` maps each claim to its input, command and output, with software versions and input
    SHA-256.
- **Not in git:** the species cache. It is reproducible from the committed species lists; every record
  stores its full settings.

## 9. "The headline comparison mixes estimands and validation regimes" — agreed; common-set table added

- **The mixing was worse than noted.** eQuilibrator and GC had been evaluated at pH 7, I = 0.25 M, pMg 3,
  which is neither reference. Both were recomputed at pH 7 / I = 0 / pMg 14 (`*_std.json`).
- `common_reference_comparison.py` scores every method on one common set (n = 319) against **both**
  references, with the regime in the label (`common_reference_comparison.json`):

  | method (regime) | MAE vs standardized | MAE vs native |
  |---|---|---|
  | MetaG (development) | 9.43 | 8.73 |
  | dGPredictor retrained (held-out, family-grouped CV) | 9.97 | 8.16 |
  | dGPredictor original vocabulary (held-out) | 9.94 | 8.26 |
  | dGPredictor original (in-sample) | 3.60 | 3.01 |
  | eQuilibrator CC (in-sample, I = 0) | 1.26 | 4.48 |
  | eQuilibrator CC (in-sample, I = 0.25, pMg 3) | 3.02 | 2.94 |
  | Group contribution (in-sample, I = 0) | 5.88 | 7.15 |

  - The standardized reference was built with eQuilibrator's own species data, so eQuilibrator against it is
    partly circular, in addition to being in-sample.
  - The only like-for-like comparison is MetaG (development) vs dGPredictor (held-out).
- **Figure.** `tecrdb_fiveway.png` is now titled "native-condition reference", and MetaG's row is labelled
  development.

## Submission-priority items

| # | Item | Status |
|---|---|---|
| 1 | Explicit-cache collision fix + regression test | done (item 3) |
| 2 | Conformer thermodynamics | replaced by the per-minimum estimator, impact quantified (item 2) |
| 3 | Freeze rules, evaluate on unseen data | rules frozen in `FLAG_DEFAULTS`; external dataset **not yet** done |
| 4 | Recast 9.65 as development MAE | done; selection-adjusted 9.80 ± 0.06 (item 1) |
| 5 | Commit provenance | done (item 8) |
| 6 | Cycle closure by route and metabolite | done within TECRDB (`cycle_closure_by_route.json`, below); ModelSEED-scale **not yet** done |
| 7 | Sensitivity intervals for benchmark-selected rules | done (item 1 table) |

**Cycle closure details (item 6):**
- 49 independent cycles, 112 reactions.
- RMS non-closure: MetaG 1.47 kJ vs 3.50 for the experimental values. The previous COSMO + anchors pipeline
  was 6.7.
- Per route, MetaG stays ≤ 1.7 kJ while experiment ranges 1.5–5.6.
- Metabolites on the excess-non-closure reactions are listed in the JSON.

## Known limitations (to be stated in the manuscript)

1. **No external dataset.** Accuracy is TECRDB development (9.59) or selection-adjusted (9.80) MAE.
2. **Neutral pH-0 reference for multi-acid species** (bisphosphates): intramolecular acid–acid contacts are
   over-stabilised. FBP isodesmic +18.6 kJ, not corrected.
3. **Phosphagen core** (n = 4, bias ≈ +37 kJ): UMA − PBE0 is only +7.8, so the cause is the solvation or
   speciation of the phosphoguanidinium cation. Unresolved.
4. **Lipid-phase references** (4 reactions with ≥ C12 acyl transfer/hydrolysis): K′ values imply micellar
   partitioning. Reported as a category; included in all MAEs.
5. **pKa coverage.**
   - Compound constants for 9 acids only; rule-table fallback elsewhere.
   - Single constants for non-α amines.
   - α-amino-acid carboxyl / amine constants are macroscopic-style while the α-keto side is now microscopic,
     which is the source of the ~4 kJ shift on amino-acid dehydrogenases.
   - Unmatched anions (sulfite, nitrite, enolates, ring N⁻) stay charged.
6. **Thermal ensemble.** Soft-mode flooring of non-reference minima; `U_samp` excludes thermal variation.
7. **Hydration.** Calibrated on 11 simple carbonyls; application to large sugars is an extrapolation.
8. **Cost.** The per-minimum thermal ensemble raises per-species cost by about 2–5×.
