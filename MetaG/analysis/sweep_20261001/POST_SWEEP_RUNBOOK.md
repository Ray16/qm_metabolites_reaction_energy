# Post-sweep runbook

> **Benchmark reference = openTECR (since 2026-10-02).** Every reported metric is scored against
> `experiments/qm_mlip_solvation/scripts/reactions_opentecr_std.json` (built by `analysis/build_opentecr_standard.py`
> from the pinned snapshot `analysis/opentecr_source/`). The current production directories are
> `final_20261002_opentecr` (pre-calibration) and `final_20261002_opentecr_calibrated`; the TECRDB-scored
> `final_20261001c*` directories are kept for the SI comparison only. See section 8.

This is the restart-safe record and rerun checklist for the completed `2026-10-01c` production analysis.
Run commands from the MetaG repository root. The frozen calibration artifact and production records
share source hash `ab4c5caa9c25990047e07580bd0706807cec91e8fe757fa2fab7abbc4c2eb2b6`.

## 0. Confirm that species computation is complete [complete]

The production list is the exact union of:

- `sp_tec_v2.json`: 539 routed species for 364 TECRDB reactions;
- `sp_gen.json`: 714 routed species for 300 non-TECRDB ModelSEED reactions;
- overlap: 131 species;
- `list_final_big.json` / `list_final_small.json`: 1,122 unique species total.

Check progress:

```bash
find analysis/sweep_20261001/done_final -maxdepth 1 -type f | wc -l
find analysis/sweep_20261001/claims_final -mindepth 1 -maxdepth 1 | wc -l
ps -eo pid,etime,cmd | rg 'species_worker.py .*list_final_(big|small)\.json'
```

Completion gate:

- 1,122 readable records exist in `done_final/`;
- every record belongs to `list_final_big.json`;
- all errors are reviewed (the expected unsupported `[Hg+2] q2` record is explicit);
- no live species workers remain.

If a worker was interrupted, a claim directory can remain without a completion record. Confirm that no
worker owns it, then remove only that stale claim and relaunch a species worker. Never delete
`cache/`: species records are content-addressed by canonical SMILES, charge and effective numerical
settings, and are written atomically.

## 1. Assemble and score the TECRDB production run [complete]

Use a new output and claim directory; do not overwrite the pre-review runs.

```bash
mkdir -p analysis/sweep_20261001/final_20261001c
mkdir -p analysis/sweep_20261001/final_20261001c_claims
bash analysis/sweep_20261001/launch_final.sh \
  analysis/sweep_20261001/final_20261001c \
  analysis/sweep_20261001/final_20261001c_claims \
  HOST:GPU [HOST:GPU ...]
```

Completion gate:

- exactly 364 readable reaction JSON files;
- one `source_hash`, one `config`, and physics `2026-10-01c` across all records;
- all records use `reactions_tecrdb_std.json`;
- every error or `dG = null` is reviewed.

Archive this first pass as the pre-calibration result set.

## 2. Recalibrate uncertainty [complete]

```bash
python -m metag.tools.calibrate \
  analysis/sweep_20261001/final_20261001c \
  --reactions ../experiments/qm_mlip_solvation/scripts/reactions_tecrdb_std.json
```

This rewrites `metag/data/sigma_class_calibrated.json`. Confirm that:

- its `config` matches the production records;
- its physics is `2026-10-01c`;
- grouped near-duplicate folds were used;
- `cv_coverage_interval95`, `interval_sigma_mult`, `cv_heldout_MAE`, and source provenance exist.

Frozen result: 347/364 coverage (95.3%; exact binomial 95% CI 92.6–97.3%), `m = 2.25`, using
245 grouped-CV clusters.

## 3. Run the interval-matched production pass [complete]

Run the same 364 reactions into a second new directory after calibration:

```bash
mkdir -p analysis/sweep_20261001/final_20261001c_calibrated
mkdir -p analysis/sweep_20261001/final_20261001c_calibrated_claims
bash analysis/sweep_20261001/launch_final.sh \
  analysis/sweep_20261001/final_20261001c_calibrated \
  analysis/sweep_20261001/final_20261001c_calibrated_claims \
  HOST:GPU [HOST:GPU ...]
```

Require the same completion checks as step 1. Point estimates must match the pre-calibration pass;
interval fields, `coverage_calibrated`, and `calibration_basis` must match the new artifact.

## 4. Recompute cycle consistency [complete]

```bash
python analysis/sweep_20261001/cycle_closure_by_route.py \
  analysis/sweep_20261001/final_20261001c_calibrated
```

Output: `analysis/sweep_20261001/cycle_closure_by_route.json`.

Report the global projection residual and route attribution as internal state-function consistency, not
as predictive validation.

## 5. Recompute common-reference comparisons [complete]

Regenerate the standardized-condition baselines if their inputs or environments changed:

```bash
conda run -n eqapi python analysis/sweep_20261001/run_equilibrator_std.py
conda run -n eqapi python analysis/sweep_20261001/run_group_contribution_std.py
python analysis/sweep_20261001/common_reference_comparison.py \
  analysis/sweep_20261001/final_20261001c_calibrated
```

Output: `analysis/sweep_20261001/common_reference_comparison.json`.

Require one common reaction set and preserve the regime labels: MetaG is a development result;
dGPredictor is held out; eQuilibrator and group contribution are in-sample.

## 6. Score and summarize the ModelSEED generality panel [complete]

The fixed input is `analysis/sweep_20261001/generality_inputs.json` (300 balanced reactions absent from
TECRDB, 50 per top-level EC class). Score it with the same source and effective configuration as the
TECRDB calibrated pass, setting:

```bash
TECRDB_INPUTS="$PWD/analysis/sweep_20261001/generality_inputs.json"
```

`launch_final.sh` forwards `TECRDB_INPUTS` to workers. Summarize with
`python analysis/sweep_20261001/summarize_generality.py`. The report includes:

- successful estimates and explicit failures;
- OOD and interval-calibration status;
- routing-rule frequencies;
- runtime distribution;
- GC/eQuilibrator availability and disagreement where present.

Do not report MAE or call this external validation: the panel has no experimental reaction energies.

## 7. Regenerate figures and synchronize the manuscript [complete]

Regenerate the paper-facing result figures from the calibrated production directory. Figure labels
preserve each method's validation regime:

```bash
conda run -n base python analysis/sweep_20261001/common_reference_scatter.py
conda run -n base python analysis/sweep_20261001/make_results_figure.py
```

Canonical analysis copies are written to `analysis/sweep_20261001/figures/`; synchronized manuscript
copies are written to `manuscript/figures/`.

Then update generated manuscript numbers:

```bash
cd manuscript
make numbers SWEEP=../../MetaG/analysis/sweep_20261001/final_20261002_opentecr_calibrated
make clean
make
git diff --check
```

Manually fill only macros not owned by `tools/sync_numbers.py`: common-set MetaG error, cycle residuals,
and ModelSEED generality counts. Verify that no red result placeholders remain except author,
acknowledgement and repository/DOI fields.

Finally commit and push the nested manuscript repository:

```bash
git add main.tex numbers.tex refs.bib sections figures
git commit -m "Finalize MetaG production results"
git push overleaf HEAD:main
```

## Final release gate

- production and calibrated-pass record counts are complete;
- configuration fingerprints and source hashes are uniform;
- calibration matches the deployed estimator;
- cycle and comparison artifacts were regenerated from the calibrated pass;
- ModelSEED is described as an applicability stress test;
- manuscript builds without undefined references or stale result placeholders;
- final commit hashes and artifact hashes are added to `provenance/MANIFEST.md`.

## 8. Benchmark reference switched to openTECR (2026-10-02) [complete]

The point estimates do not depend on the reference; only `exp`/`err`, the calibration and the intervals do.
Same source hash (`ab4c5caa…`) and configuration as the 2026-10-01c production records.

```bash
# reference (eqapi env; CPU)
~/miniforge3/envs/eqapi/bin/python analysis/build_opentecr_standard.py
# re-assemble from the warm cache (launch_final.sh now defaults to the openTECR reference)
bash analysis/sweep_20261001/launch_final.sh $PWD/analysis/sweep_20261001/final_20261002_opentecr \
  $PWD/analysis/sweep_20261001/final_20261002_opentecr_claims HOST:GPU [HOST:GPU ...]
PYTHONPATH=$PWD python -m metag.tools.calibrate analysis/sweep_20261001/final_20261002_opentecr \
  --reactions ../experiments/qm_mlip_solvation/scripts/reactions_opentecr_std.json
cp metag/data/sigma_class_calibrated.json ../MetaG_new/src/metag/data/sigma_class_calibrated.json
bash analysis/sweep_20261001/launch_final.sh $PWD/analysis/sweep_20261001/final_20261002_opentecr_calibrated \
  $PWD/analysis/sweep_20261001/final_20261002_opentecr_calibrated_claims HOST:GPU [HOST:GPU ...]
python analysis/sweep_20261001/cycle_closure_by_route.py analysis/sweep_20261001/final_20261002_opentecr_calibrated
python analysis/sweep_20261001/common_reference_comparison.py analysis/sweep_20261001/final_20261002_opentecr_calibrated
python analysis/sweep_20261001/nested_policy_cv.py          # rescored from stored dG vs BENCH_REF (default openTECR)
python analysis/sweep_20261001/common_reference_scatter.py; python analysis/sweep_20261001/make_results_figure.py
```

Checks done: 364/364 records in both passes, 0 errors; dG, dG_raw, U_samp, class and routes identical to
`final_20261001c_calibrated` for all 364; every `exp` equals the openTECR reference; 44 intervals changed after
recalibration. `BENCH_REF=…/reactions_tecrdb_std.json` reproduces the old nested-CV and cycle numbers exactly.
The old TECRDB-calibrated artifact is kept as `sigma_class_calibrated.tecrdb_20261001c.json`.

| metric | TECRDB (SI) | openTECR (reported) |
|---|---:|---:|
| MAE / median / RMSE, all 364 | 9.58 / 7.24 / 13.39 | 9.54 / 7.15 / 13.32 |
| 95% interval coverage (nested CV) | 347/364 = 95.3% | 346/364 = 95.1% (CI 92.3–97.0%) |
| multiplier m / held-out MAE | 2.25 / 9.58 | 2.25 / 9.54 |
| common set (319): MetaG / dGP held-out / eQ / GC | 9.42 / 9.97 / 1.26 / 5.88 | 9.38 / 9.91 / 1.34 / 5.82 |
| nested-selected / final-policy MAE (359) | 9.80 / 9.75 | 9.75 / 9.70 |
| cycle RMS, experimental values | 3.50 | 3.47 |
