# Response to the second-round review (2026-10-01)

This round answers the eight findings of the second review. It supersedes the corresponding statements in
`REVIEW_RESPONSE_20261001.md` (round 1).

**Status at the time of writing:**
- All code and documentation fixes are committed, with tests passing.
- A full species recompute is running, required by finding 5. It covers all TECRDB species plus a new
  unseen-chemistry set.
- The **final numbers will be regenerated when it completes.** Numbers below marked *(pending)* will change.

## Finding 1 (High) — calibration guard did not see every estimator change → fixed

- `effective_config()` (`metag/pipeline.py`) now additionally fingerprints:
  - the **full species-cache settings** (`implicit_settings`): thermal ensemble mode, solvent relaxation,
    dedup, sampling, model;
  - `thermal_ensemble`, `solv_relax`, `acid_hb_filter` (with their parameters), `polyacid_pka`,
    `arylamine_nonbasic`;
  - **content hashes of every constant table**: `pka_constants` (all P / carboxyl / sulfate / thiol /
    phenol ladders, `CARBOXYL_PKA_ALPHA`, `POLYACID_PKA`) and `hydration_constants` (`HYDRATION_CAL`,
    `MAX_HYDRATION_SITES`).
  - An edited constant now changes the fingerprint even without a version bump.
  - `PKA_TABLE_VERSION` is bumped to `2026-10-01b`.
- **Tests:**
  - `test_effective_config_tracks_every_estimator_switch` toggles POLYACID_PKA, ARYLAMINE_NONBASIC,
    PKA_ENV, CARBONYL_HYDRATION_ALL, HYDRATION_CAL, PH0_ISOMERASE, NTP_CORE, THERMAL_ENSEMBLE,
    ACID_HB_FILTER and SOLV_RELAX; each must change the fingerprint.
  - `test_constant_edit_changes_fingerprint` edits one pKa constant.
- **Consequence:** the calibration artifact from round 1 no longer matches the runtime fingerprint, so
  intervals are reported **not coverage-calibrated** until recalibration on the pending production run.
  This is the guard working as intended.

## Finding 2 (High) — hydration calibration not reproducible from the stated command → fixed

- `khyd_validation.py` now computes the **deployed** application-domain fit itself (`calibration_domain_fit`).
  - The domain is selected with the pipeline's own SMARTS (`aldehyde_hydration._KETO_ACID`).
  - The 5 α-keto acids are excluded; 11 members are listed in the output.
- Running `python khyd_validation.py` reproduces the deployed fit: a = 0.639, b = 2.347 kJ per event,
  LOO 0.46 log units. Deployed values: `HYDRATION_CAL = (0.639, 2.35)`. The output also records
  `deployed_HYDRATION_CAL`.
- **Test:** `test_deployed_hydration_calibration_matches_reproducible_fit` fails if the deployed constants
  drift from the committed fit.
- The all-16 fit (a = 0.509) is still reported, labelled as such; it is not the deployed calibration.

## Finding 3 (Medium) — scope of the optimism estimate → stated precisely

The selection-adjusted 9.80 ± 0.06 (round 1) quantifies **only** the re-selection of seven binary routing
switches. The policy grid used `THERMAL_ENSEMBLE=0`, the legacy single-minimum thermal estimator, on the
359 reactions common to all 128 combinations.

It does **not** account for TECRDB-informed choices outside those switches:
- the pKa constants and rule classes;
- the hydration domain, calibration and α-keto exclusion;
- the solvent model and water reference;
- the truncation rules;
- the thermal estimator and other development decisions.

The total optimism of the final development MAE is therefore **not** quantified. A frozen-model external
evaluation is the only way to estimate it. The figure 9.80 is reported only as "optimism from re-selecting
seven routing switches ≈ 0.05 kJ", never as a generalization estimate.

## Finding 4 (Medium) — cycle-closure description → corrected

- **The round-1 claim "every route ≤ 1.7 kJ" was wrong.** From `cycle_closure_by_route.json`:

  | route | n | MetaG | experiment |
  |---|---|---|---|
  | truncated + pH-0 | 33 | 1.89 | 3.85 |
  | NTP core + truncated + pH-0 | 15 | 1.65 | 5.57 |
  | cofactor ring + truncated | 15 | 0.18 | 2.81 |
  | pH-0 | 13 | 1.13 | 2.80 |
  | cofactor ring + truncated + pH-0 | 13 | 0.87 | 2.78 |
  | truncated | 6 | 0.80 | 1.51 |
  | **full** | **2** | **4.90** | 0.35 |
  | others | ≤ 4 each | ≤ 0.87 | — |

- **What the numbers are.** One global weighted projection of all 364 ΔG onto the cycle space. The
  reaction-level residuals are then **attributed** to routes, so this is not an independent closure test
  per route.
- 1.47 kJ is the RMS projection residual over the 112 cycle-supported reactions.
- Individual basis cycles still close with errors up to 15.1 and −16.1 kJ (median |error| 1.9 kJ over the
  15 listed).
- This is evidence of internal state-function consistency of the estimator, **not predictive validation**.
  The numbers will be recomputed on the pending production run.

## Finding 5 (Medium) — thermal correction not in convergence / U_samp → fixed (recompute running)

- `_ThermalTrack` (`metag/pipeline.py`) computes per-minimum RRHO corrections **during** sampling.
  - After every seed batch, minima inside the 15 kJ window get a UMA-Hessian correction. It is computed once
    per representative geometry, and recomputed if a lower-G duplicate replaces it.
  - The adaptive convergence test and the `U_samp` trajectory now use G = E + ΔG_solv + G_corr.
- The reference minimum is still validated by the true-minimum test (mode following), and its value
  replaces the tracked one.
- The cache key changed (`thermal_ensemble: v2conv-…`), so all species are being recomputed. This run is
  in progress *(pending)*.

## Finding 6 (Medium) — hydration "exact" wording → qualified; omissions now recorded

- The docstrings now state the scope: **exact over the successfully computed hydration states of at most
  three canonically selected sites.**
- Omitted states are no longer silent; each is recorded in `routes.warnings`:
  - states that are not gas-phase minima (`SpeciesRearranged`);
  - states whose QM fails;
  - states that cannot be constructed;
  - molecules truncated to three sites.

## Finding 7 (Low) — reproducibility / documentation defects → fixed

- `provenance/input_sha256.txt` uses repository-root-relative paths throughout. Verification:
  `cd <repo root> && sha256sum -c MetaG/analysis/sweep_20261001/provenance/input_sha256.txt` (all OK).
- `reasm_stages.py` and `run_policy_grid.sh` no longer contain user-specific absolute paths; they resolve
  paths relative to the script and use `$PY` (default `python`).
- Stale comments corrected:
  - the hydration source comment now cites the verified set and its numbers;
  - water reference, NTP core, zwitterion guard, anchors, the hydro-lyase patch and SMD now state their
    current defaults;
  - the module docstring of `pipeline.py` describes the current default pipeline.

## Finding 8 (Low) — "like-for-like" → corrected

The MetaG and held-out dGPredictor rows of `common_reference_comparison.json` use **the same reactions and
the same reference**, but **not the same validation regime**:
- MetaG is a TECRDB development result;
- dGPredictor is held-out, refit per family-grouped fold.

The comparison is reported with that qualification only.

## Additional change requested by the PI — generic rules only

- `POLYACID_PKA` (compound-specific macroscopic ladders for 9 named acids) is now **off by default**.
  - A lookup of named compounds does not transfer to unseen chemistry, and its TECRDB effect is ≈ 0.02 kJ.
  - The pKa layer therefore uses only generic structural rules.
- **Known weak spots of the rule table** (reported as limitations; errors are for the parent acids at pH 7):
  - adjacent carboxyls of the oxalate type (−22.7 kJ) and malonate type (−5.6 kJ);
  - cis-unsaturated diacids of the maleate type (+3.8 kJ);
  - long saturated diacids (+1.9 kJ).
- **New unseen-chemistry generality test** *(pending)*. `build_generality_set.py` draws 300 random, balanced
  ModelSEED reactions not in TECRDB, 50 per EC class.
  - They are scored with the final configuration to measure failure rate, out-of-distribution flag rate and
    how often each rule fires.
  - No experimental values exist for them; this tests robustness and applicability, not accuracy.

## Pending (to be filled when the running recompute finishes)

1. Production run on TECRDB with the final defaults → development MAE, recalibrated σ, production rerun.
2. Cycle closure (finding 4 table) and common-reference comparison recomputed.
3. Generality report for the 300 unseen reactions.
4. Figures regenerated from the final results: the three-way figure is rewritten to use one standardized
   reference, with MetaG labelled "development"; 18-pt fonts throughout.
