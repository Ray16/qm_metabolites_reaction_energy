# Post-sweep runbook

This is the restart-safe checklist for finishing the `2026-10-01c` production analysis after the
current species sweep. Run commands from the MetaG repository root. Do not substitute values from an
older calibration artifact: `metag/data/sigma_class_calibrated.json` currently describes an earlier
physics/configuration fingerprint and is intentionally rejected by `manuscript/tools/sync_numbers.py`.

## 0. Confirm that species computation is complete

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

## 1. Assemble and score the TECRDB production run

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

## 2. Recalibrate uncertainty

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

The previous values (95.6% coverage and `m = 2.25`) are historical only and must not be copied into the
paper unless reproduced by this calibration.

## 3. Run the interval-matched production pass

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

## 4. Recompute cycle consistency

```bash
python analysis/sweep_20261001/cycle_closure_by_route.py \
  analysis/sweep_20261001/final_20261001c_calibrated
```

Output: `analysis/sweep_20261001/cycle_closure_by_route.json`.

Report the global projection residual and route attribution as internal state-function consistency, not
as predictive validation.

## 5. Recompute common-reference comparisons

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

## 6. Score and summarize the ModelSEED generality panel

The fixed input is `analysis/sweep_20261001/generality_inputs.json` (300 balanced reactions absent from
TECRDB, 50 per top-level EC class). Score it with the same source and effective configuration as the
TECRDB calibrated pass, setting:

```bash
TECRDB_INPUTS="$PWD/analysis/sweep_20261001/generality_inputs.json"
```

`analysis/tecrdb_rescore.py` can score this input, but `launch_final.sh` does not currently forward
`TECRDB_INPUTS`; either extend that launcher or pass the variable explicitly to each worker. Use fresh
output and claim directories.

A dedicated summary script is still required. It must report:

- successful estimates and explicit failures;
- OOD and interval-calibration status;
- routing-rule frequencies;
- runtime distribution;
- GC/eQuilibrator availability and disagreement where present.

Do not report MAE or call this external validation: the panel has no experimental reaction energies.

## 7. Regenerate figures and synchronize the manuscript

Regenerate all result figures from the calibrated production directory, including the three-way
comparison and any class/error/coverage panels retained for the paper. Figure labels must preserve each
method's validation regime.

Then update generated manuscript numbers:

```bash
cd manuscript
make numbers SWEEP=../analysis/sweep_20261001/final_20261001c_calibrated
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
