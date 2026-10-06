# Refactor Status

Last updated: 2026-10-02

## Complete

- Created an installable `src/` layout under `MetaG_new`.
- Copied every frozen runtime module and package-data file.
- Excluded caches, bytecode, sweep outputs, figures, and manuscript files.
- Added a single package CLI entry point.
- Separated fast tests from the opt-in GPU integration test.
- Removed test dependencies on the old `analysis/` directory.
- Promoted reusable result, routing, pKa, precision, calibration, and
  provenance checks into `metag.validation`.
- Archived the frozen release metrics, representative records, and artifact
  hashes as regression fixtures.
- Added tests for canonical cache keys, settings sensitivity, atomic writes,
  resumable reads, and disabled-cache behavior.
- Added immutable reaction/species input models and moved validation out of
  the orchestration module while preserving its public compatibility alias.
- Added shared atomic JSON publication and exclusive task-claim primitives,
  including concurrent-contention and failure-cleanup tests.
- Built an installable wheel and verified its contents.
- Verified exact routing/configuration parity on all 364 TECRDB and 300
  ModelSEED panel reactions.
- Verified exact old/new warm-cache result parity on all 364 TECRDB reactions,
  with backend calls forbidden.
- Added validation at the public router boundary and an AST-based policy audit
  that rejects reaction-ID or annotation-dependent point-routing conditions
  and freezes the known legacy-anchor dependency inventory.
- Audited the anchor-derived truncation veto on both frozen panels: 30 TECRDB
  and 20 ModelSEED reactions match an anchor subclass, but the veto changes
  zero routes because none pass the independent full-molecule gate.
- Added `docs/anchor-audit.md`, which evaluates every legacy anchor by physical
  purpose and current ALPB behavior. Most COSMO-era corrections should be
  retired; phosphagen remains a real diagnosed limitation but not yet a
  quantitatively justified correction.
- Added immutable `RoutingPolicy` resolution so environment-driven routing
  choices are captured once at the public boundary and can be tested without
  changing frozen defaults.
- Moved family detectors from `routing/anchor.py` into
  `routing/reaction_families.py` (no offsets); `anchor` re-exports them and
  `uncertainty` imports detection directly.
- `prefer_full()` consults family detection only when `ANCHOR_CORRECT` is on.
  Route-neutral on both frozen panels; on unseen chemistry this removes the
  last benchmark-derived input to point routing.

## Verification Record

- Fast suite: 182 passed.
- GPU integration suite without explicit opt-in: 1 skipped.
- Source compilation: passed.
- Wheel build and isolated installed-package import: passed.
- Required package JSON data present in the wheel.
- Frozen versus refactored routing: 0 differences across 364 TECRDB and 300
  ModelSEED reactions.
- Frozen versus refactored warm-cache output: 0 exact record differences
  across all 364 TECRDB reactions.
- Representative old/new species cache paths: exact matches.

## Frozen During Parity

- All numerical algorithms in `pipeline.py`, `energetics/`, and `routing/`
- Environment-variable names and defaults
- Species cache version, key payload, and record schema
- Reaction routing decisions on the frozen panels (the `prefer_full` anchor
  veto now applies only with `ANCHOR_CORRECT` on; 0 panel routes changed)
- Uncertainty model and calibrated data

## Provenance Finding

The current shared cache gives identical old/new results but differs from 134
archived TECRDB records in at least one exact field, usually by 0.01-0.03
kJ/mol in an unrounded stage. The archived production source hash alone did
not freeze cache contents. Before declaring full release reproduction, add a
per-record or aggregate immutable cache manifest and verify against it.

## Next Extractions

1. Introduce immutable typed configuration and an environment compatibility
   adapter.
2. Move reaction validation and input normalization from `pipeline.py` into
   explicit domain models.
3. Extract route planning from `pipeline.py` without changing route output.
4. Extract species-energy computation and reaction assembly behind narrow
   interfaces.
5. Add content-addressed execution tasks and atomic claims for local, SSH, and
   Slurm workers.
6. Run full TECRDB and ModelSEED output parity and performance benchmarks.
7. Move maintained analysis/figure generation and then the manuscript.

The subsequent scientific review is recorded in
`docs/scientific-policy-audit.md`. It uses a narrow definition of hardcoding:
dataset- or reaction-specific knowledge that changes a production point
prediction. It separately tracks benchmark-selected structural policies,
physical and literature chemistry, numerical convergence, and
dataset-calibrated uncertainty.

The manuscript now lives in `manuscript/` (moved 2026-10-02; still its own git repo synced with Overleaf and ignored by the parent repo). `tools/sync_numbers.py` reads `src/metag` and the frozen `../MetaG/analysis` artifacts; the two figure scripts in `../MetaG/analysis/sweep_20261001` write here. Scientific
policy changes begin only after this parity milestone and require a new physics
version.
