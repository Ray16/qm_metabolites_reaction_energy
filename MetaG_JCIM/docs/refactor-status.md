# Refactor Status

Current checkpoint: **0.2.0.dev23**, with the frozen numerical source used for the
fresh biochemical evaluation. The maintained interface has one lazy `metag COMMAND`
entry point for 13 supported commands; their arguments and artifact contracts are
documented in [runtime-reference.md](runtime-reference.md) and
[advanced-workflows.md](advanced-workflows.md). Dated study drivers remain immutable
scientific evidence rather than additional public workflow entry points.

The current integration adds explicit reducing-sugar solution families, complete
anomer/ring/open-chain partitions, owned protonation factors and exact cancellation
of shared physical pools. It preserves original structures, charges and stoichiometry.
The frozen core passes **1,122 tests**. The wheel's CPU-only command/preflight parity
and unchanged numerical source are recorded in
[the release verification](../analysis/remaining_review_20261022_01/release_candidate_01/installed_verification.json).

The completed cold evaluation returns **327 estimates from 364 requested inputs**:
331 pass structural admission, four molecular calculations return no estimate, and
33 inputs remain structurally excluded. Explicit pools recover **72 of 74** newly
admitted cases. This improves supported representation and availability, not the
common-target accuracy metric. The current-label, historical-label and identical-target
comparisons are separated in [the metrics](../analysis/remaining_review_20261022_01/current_panel_metrics.json).
The [current dGbyG comparison](../analysis/remaining_review_20261022_01/bridge_candidate/current_matched_comparison.json)
retains its author-OOF regime and bridge-support limits.

The finite policy-selection audit remains in progress. Its production source,
candidate grid, groupings and external-source evaluation are frozen; alternatives
selected in that audit are diagnostics and are not promoted to the point workflow.
See [the current decision log](../analysis/remaining_review_20261022_01/selection_current_01/DECISION_LOG_CURRENT.md).

The preceding checkpoints below retain their original source, test and result identities.

## Earlier coordinate-integrity checkpoint (0.2.0.dev22)

RMSD checks now align a copy because RDKit's `GetBestRMS` mutates its probe.
This preserves the actual coordinates at which retained electronic and solvent
terms were evaluated. Six regression cases fail on the frozen prior source and
pass with exact coordinate preservation. The duplicate criterion and lower-G
representative policy remain unchanged; numerical point parity is not claimed.
The optional CPCM-X path also requires a successful return code and finite
parsed energy. Supplied-Hessian and external-solvation CPU imports still work.
All 1,081 tests pass. Earlier source/wheels and results remain immutable.

## Review cleanup candidate

Removed five unused conformer/demo/legacy-solvation functions and made MMFF
preoptimization failures visible without changing the coordinate fallback.
Centralized seven protonation flags with identical defaults and checked 77
before/after truth-table cases. Nine active numerical conformer helpers retain
AST parity apart from documentation and that explicit warning.

The maintained state-comparison module and command now report a neutral
fixed-solvent phase-transfer range from all validated recorded minima. The
module refuses ions, incomplete state sets and nonfinite references; the command
also verifies hashes and refuses overwrites. No point-model parameter or
residual correction was added. All 1,071 tests pass. The bundled reference-data
correction preserves every nonexperimental field of the 364 inputs.

The evaluated dev20 snapshot remains immutable. The new candidate has its own
source/wheel identity and CPU verification; it is not relabelled as the producer
of the earlier fresh quantum runs. See
[the review response](../analysis/reviewer_revision_20261022_01/RESPONSE.md).

## Previous consolidation (dev20)

## Current consolidation

The maintained tools now share the lazy `metag COMMAND` interface, while old
module invocations and scoring arguments remain compatible. Uncertainty
reporting is separate from point assembly, and fitting/inference share primitive
reaction normalization. One calibration snapshot supplies class widths,
provenance and applicability. A dedicated calibration extra supplies fitting
dependencies. The active scorer no longer contains unreachable benchmark-offset
or legacy solute-only SMD/water correction branches; production policy continues
to reject those switches before QM.

The physical default retains exact source-backed acid models. The full neutral
ring-state candidate failed its whole-panel gate and remains disabled. No
favorable-case exception was introduced. The refactored source passes 1,065 tests;
fresh numerical integration, installed-package equality and whole-panel replay
are recorded in `analysis/workflow_release_20261022_01/`. Frozen scientific runs
remain in their original paths with their original sources and failures.

This checkpoint does not establish independent accuracy or calibrated coverage.
One frozen job per worker process remains the supported configuration boundary.

## Latest phase: generalization and physical error diagnosis

The user subsequently authorized systematic generalization fixes and parallel
large-error experiments. See the follow-up section in
[the comprehensive review](comprehensive-review-2026-10-06.md) for tests,
rejected hypotheses, coverage reductions and unresolved scientific work.
The first refactor's parity results below remain historical evidence, not a
claim that later domain safeguards preserve every old prediction.

## Earlier priority: refactor before accuracy changes

The user requested that architectural cleanup precede further work on large
prediction errors. The numerical baseline is the code after the pKa stage
below; no additional scientific changes are part of the refactor batch.
The other ring-chain effort was abandoned with code present. Its module is
retained and explicitly documented as experimental, with its default off.

Completed in the first extraction batch:

- Moved shared runtime switches to `config.py` and route planning to
  `routing/planner.py`; split planning into core, truncation, protonation and
  neutral-reference stages. The existing pipeline routing API is preserved.
- Captured the ring-chain switch in the routing policy rather than reading
  the environment inside an explicitly configured route.
- Separated the reaction JSON loader, CLI parsing, batch checkpointing and
  display from the QM orchestrator. CLI help works without QM dependencies.
- Extracted pure reaction assembly, structural sampling budgets, acid-contact
  diagnostics, thermal-ensemble bookkeeping, and aqueous-state mixture
  evaluation. Backend callbacks and numerical settings are supplied explicitly
  at these new boundaries; existing pipeline entry points remain available.
- Preserved arithmetic, stage order, pKa assignments, defaults, cache keys
  and result schemas. Corrected stale documentation describing the abandoned
  ring-chain estimator as validated/default-on.

Validation: **287 passed, 1 GPU test skipped**; exact routing parity for
**364/364** openTECR inputs; **zero changes** in paired replay energy/pKa-stage/
calibration-flag fields across **364/364** reactions; configuration unchanged.
Wheel build and isolated installed-package routing/CLI import checks passed
with Torch, ASE and fairchem imports prohibited. Evidence:
`../analysis/code_review/refactor_validation.json` and `refactor_replay.json`.
The replay uses cached finite-difference energies, not fresh GPU calculations.

Remaining engineering gates, before a new accuracy-development phase:

1. Extract the species-energy backend behind a service contract with immutable
   per-job settings; preserve thermal, solvation, cache and failure behavior.
2. Separate final uncertainty/result reporting from scoring orchestration,
   and finish the minimal CPU test/dependency boundary.
3. Validate optional execution paths and the external generality panel as well
   as the current openTECR baseline; retain explicit failure cases.
4. Freeze source/configuration/cache manifests for the cleaned implementation.

The whole codebase is not yet declared clean or scientifically validated.

## Scientific improvements after the parity milestone

The October 6 review is in `comprehensive-review-2026-10-06.md`. Work now
proceeds beyond historical parity, keeping UMA as the electronic-energy core
and excluding openTECR reaction energies from point-estimator fitting.
Ring-chain development was subsequently reported abandoned. The pKa stage
did not enable it or claim it was validated.

Completed pKa stage:

- Extracted inference from `tools/fit_site_pka.py` into
  `routing/site_pka.py`; production routing no longer imports a fitting tool.
- Reject unknown model names, missing/corrupt parameter artifacts, and
  nonfinite or incomplete parameters. Removed the silent fallback to the
  table when a requested site model fails to load.
- Record the selected site-model artifact's SHA-256 and inference version.
  `PKA_SITE_MODEL_FILE` provides an explicit artifact-selection interface;
  changing a file during a long-lived process refreshes its parameters.
  A fully immutable per-job configuration is still outstanding.
- Reject thiolate, phenolate, carbonate, phosphoramidate and acyl-phosphate
  groups outside the optional site's parameterization, rather than assigning
  phosphate/formate/monoester parameters to them. The default table retains
  its existing explicit rules. Rejection identifies the unsupported domain;
  it does not supply a validated prediction for that chemistry.
- Consolidated the independent-site protonation calculation into
  `energetics/protonation.py`, with stable logarithms and validation of input
  sides, site kinds, pKas and stoichiometric multiplicities before QM.
  Removed the misleading claim that the product is an exact coupled
  macroscopic transform.
- Extended reference-state canonicalization to the already-recognized
  thiol and phenol classes. Neutral and ionized drawings now yield the same
  acid reference and pKa terms. No new empirical pKa values were introduced.
  The new `pka_representation` configuration field distinguishes this behavior
  from the historical uncertainty calibration; stored calibration artifacts
  were not rewritten to claim coverage.

Validation: **284 passed, 1 GPU test skipped** (51 dependency warnings).
Tests cover unsupported domains, artifact replacement, invalid parameters,
protonation representation invariance, extreme pKas, reaction reversal,
fractional scaling and spectator cancellation.
The read-only paired replay in
`../analysis/code_review/pka_improvements.json` scored **364/364** reactions
using the historical finite-difference species cache. Twelve rounded values
changed, each by 0.1 kJ/mol. MAE was 9.543626 before and 9.544725 after.
This is a representation correction, not a demonstrated accuracy gain or
an external validation. The optional site model has unsupported groups in
48 of the input reactions. The replay script accepts explicit input/cache/
water paths, blocks cache writes and fresh calculations, and records hashes
of all consumed species-cache records.

Deferred scientific stages and their acceptance criteria (after refactoring):

1. **Protonation model:** distinguish independent microscopic sites from
   sequential macroscopic ladders; represent coupled protonation states and
   their reference microstates explicitly. Validate against independent pKa
   and population data, including uncertainty and applicability. Audit the
   approximate class values and the disabled named-polyacid lookup; moving
   constants into a file alone does not make their physics generalizable.
2. **Large residuals:** use the review's ranked error inventory to separate
   reference-condition errors, protonation, solvation, missing microstates
   and sampling. Require independent evidence for a mechanism and a class-
   wide validation panel before adopting a change; do not add per-reaction
   offsets or tune pKas to those residuals.
3. **Execution/configuration:** freeze one configuration per job, make cache
   provenance cover the actual backend/checkpoint, and require all included
   conformers to satisfy the minimum/stability policy. Preserve the separate
   experimental ring-chain code and validate any eventual integration jointly with
   hydration/protonation populations.
4. **Release evidence:** rerun the chosen production backend, evaluate
   reaction-family/scaffold-held-out and external labeled panels, quantify
   cycle closure and failures, then recalibrate uncertainty for that exact
   estimator. The archived benchmark alone cannot establish improved
   generalization.

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

The manuscript now lives in `manuscript/` (moved 2026-10-02; still its own git repo synced with Overleaf and ignored by the parent repo). As of 2026-10-06 the package is SELF-CONTAINED: `manuscript/tools/sync_numbers.py` reads `src/metag` and the pinned in-package `artifacts/` (no longer `../MetaG/analysis`), and the two figure scripts now live in `analysis/figures/` and read `artifacts/` (provenance of every pinned file: `artifacts/README.md`). Nothing under `MetaG_new/` reaches into `../MetaG`, `../experiments`, or `../results` at runtime, except the optional ~36 MB per-species QM cache used only to regenerate UQ/generality features (override `METAG_SPECIES_CACHE`; absence is non-fatal). Scientific
policy changes begin only after this parity milestone and require a new physics
version.
