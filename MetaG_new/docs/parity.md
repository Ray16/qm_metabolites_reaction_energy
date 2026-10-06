# Frozen Parity Contract

The initial refactor targets the MetaG `2026-10-01c` scientific behavior. The
source copied into `src/metag` is initially byte-identical to the frozen
runtime, except for packaging-only additions.

## Reference Release

- Production source hash:
  `ab4c5caa9c25990047e07580bd0706807cec91e8fe757fa2fab7abbc4c2eb2b6`
- Species sweep: 1,122 attempted, 1,115 successful, 7 reviewed failures
- TECRDB: 364 estimates from 364 reactions
- TECRDB error: MAE 9.58, median absolute error 7.24, RMSE 13.39 kJ/mol
- Uncertainty coverage: 347/364 (95.3%; exact 95% CI 92.6-97.3%)
- Cycle closure: 49 independent cycles, 1.52 kJ/mol RMS residual
- ModelSEED panel: 300 reactions, 297 estimates, 3 explicit failures, 14 OOD

The canonical machine-readable values and artifact hashes are in
`tests/regression/fixtures/frozen_2026-10-01c.json`.

The original calibration artifact is retained at
`tests/regression/fixtures/sigma_class_calibrated.json`; its bytes still match
the frozen hash. The live calibration's configuration metadata was revalidated
after the October 6 bookkeeping fixes, with all numerical calibration statistics
unchanged. See [accuracy review](accuracy-review.md) for the paired comparison
and the distinction between historical output parity and current correctness.

## Required Gates

Before `MetaG_new` replaces the frozen implementation:

1. Every TECRDB point estimate, route, stage, and failure status matches.
2. Every ModelSEED panel estimate, route, OOD flag, and failure status matches.
3. Effective configuration and uncertainty fields match.
4. Cache keys either match exactly or reject old entries with an explicit
   version boundary.
5. Interrupted work resumes without losing valid records.
6. Writes and claims remain atomic under multiple workers.
7. Warm-cache reaction assembly and cold species throughput do not regress.
8. Largest-first scheduling and multi-GPU scaling remain available.

Changes to physical policy are deferred until these gates pass. Such changes
will receive a new physics version and fresh validation rather than silently
altering `2026-10-01c`.

## Reproduction Commands

Compare routing in isolated old/new interpreters:

```bash
python scripts/verify_frozen_parity.py \
  --old-package ../MetaG \
  --inputs ../experiments/qm_mlip_solvation/scripts/reactions_tecrdb_std.json
```

> Note: `verify_frozen_parity.py` is a TRANSITIONAL refactor-validation tool. It deliberately
> compares the package against the old `../MetaG` tree (and its cache / water-reference), so the
> old-tree paths below are expected and are NOT part of the self-contained runtime. The package
> itself (CLI, numbers, figures) no longer reaches outside `MetaG_new/` — see `artifacts/README.md`.

Compare full assembly while forbidding all fresh QM work:

```bash
python scripts/verify_frozen_parity.py \
  --mode warm-cache \
  --old-package ../MetaG \
  --inputs ../experiments/qm_mlip_solvation/scripts/reactions_tecrdb_std.json \
  --cache ../MetaG/analysis/sweep_20261001/cache \
  --water-reference ../MetaG/analysis/sweep_20261001/water_ref_G_expt.json
```

The current shared cache reproduces old and new implementations exactly, but
differs from 134 archived TECRDB records at one or more exact fields. Observed
stage shifts are commonly 0.01-0.03 kJ/mol and can cross a one-decimal rounding
boundary. This is evidence that the shared cache changed after the archived
production run. Full archived-output parity therefore requires an immutable
cache manifest, not only the source hash.
