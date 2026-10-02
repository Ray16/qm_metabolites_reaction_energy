# Scientific Policy Audit

Status: audited against frozen `2026-10-01c`; no scientific behavior changed.

## Meaning Of Hardcoded

For this audit, **hardcoded** means a dataset- or reaction-specific exception
that imports benchmark knowledge into a production point prediction and is
unlikely to generalize outside that dataset.

A fixed value is not hardcoded in this sense merely because it is numerical.
Physical constants, cited chemical data, structure-based chemistry rules, and
global numerical convergence controls are evaluated separately. They still
need sources, domains, or convergence evidence, but they are not automatically
evidence of benchmark leakage.

## Point-Estimate Conclusion

There is no direct runtime branch on a TECRDB or ModelSEED reaction identifier
in the point-estimate routing path. Reaction identifiers found in routing files
are comments, historical examples, tests, or calibration metadata.

The one indirect benchmark dependency found in the frozen release has been
removed from production routing:

| Location | Frozen behavior | Current behavior |
|---|---|---|
| `routing/truncation_gate.py::prefer_full()` | Called `routing.anchor.subclass()` and forced any matching family away from full-molecule scoring | The veto is consulted only when `ANCHOR_CORRECT` is on, which is its sole stated purpose (an applied offset must see the route it was calibrated on). With anchors off, family recognition cannot change a route |

On the frozen panels, 30 of 364 TECRDB reactions and 20 of 300 ModelSEED
reactions match a legacy family, but none also pass the independent physical
conditions for full-molecule routing. The change is therefore exactly
route-neutral on both panels (verified with the frozen-parity tool); it only
differs from `2026-10-01c` on unseen reactions that match a family *and* pass
the physics gate, where the old behavior was benchmark leakage.

Family detectors now live in `routing/reaction_families.py`, which carries no
offsets. `routing/anchor.py` re-exports them for compatibility and holds only
the disabled numerical corrections.

## Dataset-Specific Components

### Active In Point Routing

- None. The former `prefer_full() -> anchor.subclass()` coupling is inactive
  unless `ANCHOR_CORRECT` is enabled (see above).

### Present But Disabled For Point Estimates

- `routing/anchor.py` contains TECRDB reaction IDs, fitted offsets, subclass
  splits, and fitted subclass dispersions.
- `ANCHOR_CORRECT=False` in the frozen production configuration, and frozen
  records contain no applied anchor corrections.
- The anchor module therefore does not add a fitted numerical correction to
  current point estimates. Its family detectors (now in
  `routing/reaction_families.py`) feed the structure-first uncertainty class,
  not point routing.
- `POLYACID_PKA` contains exact compound lookups for nine polyprotic acids, but
  `POLYACID_PKA` is disabled by default and does not affect frozen production
  point estimates.

### Uncertainty Only

- `uncertainty.py` uses TECRDB-calibrated class widths, residual quantiles, and
  an interval multiplier.
- Its fallback class assignment also uses enzyme-name and EC-like note
  keywords when structural classification is unavailable.
- These choices affect reported uncertainty intervals, not point `dG`.
- The intervals must continue to be described as internally calibrated on
  TECRDB-like chemistry, not externally validated uncertainty.

## Benchmark-Selected, Structurally General Policies

The following are not reaction-ID exceptions. They operate from molecular
structure and can apply to unseen reactions, but their adoption or default
setting was influenced partly by TECRDB results:

- `PH0_ISOMERASE=True`
- full-molecule versus balanced-truncation routing
- radius 3 around anomeric reaction centers
- selected conservative exclusions in the pKa transformation
- exclusion of alpha-keto acids from carbonyl hydration
- curated cofactor and NTP core transformations

These policies may be scientifically valid and generalizable. Claims of
benchmark independence should nevertheless distinguish a general structural
rule from a rule whose selection used benchmark performance. Independent
validation is preferable to removal.

## General Scientific And Numerical Policies

These are not considered dataset-specific hardcoding:

- Thermodynamic conditions such as `T = 298.15 K`, pH 7, ionic strength 0 M,
  and the 1 M solute standard-state conversion.
- Sourced pKa values, hydration constants, and the experimental liquid-water
  reference, provided their source, convention, and applicability domain stay
  attached.
- SMARTS or topology rules that represent identifiable chemistry and are
  applied uniformly to any matching molecule.
- Balanced reaction-center truncation. It is a physical isodesmic
  approximation intended to reduce non-cancelling conformational fluctuations,
  not a dataset-specific correction.
- Conformer budgets, energy windows, optimizer tolerances, truncation radii,
  and retry thresholds. These are numerical heuristics that require
  convergence and sensitivity tests.
- Worker counts, batching, paths, task claims, and other execution policy,
  provided they cannot alter scientific results silently.

## Evidence Still Needed

| Policy | Evidence |
|---|---|
| Truncation radii | Radius convergence stratified by reaction size, center type, and molecular flexibility |
| Conformer limits and energy windows | Repeat-seed and budget convergence on rigid and floppy species |
| pKa structural rules | Coverage report for unmatched ionizable groups and comparison with independent constants |
| Cofactor and NTP cores | Atom-mapping, cancellation, reversibility, and radius-sensitivity tests |
| Carbonyl hydration | Preserve independent hydration-constant validation and report chemical-domain limits |
| Uncertainty model | External calibration data and a structural classifier that does not depend on annotation text |

The legacy numerical corrections and their original physical rationale are
reviewed individually in `docs/anchor-audit.md`.

## Rules For Changes

- Do not alter frozen `2026-10-01c` behavior in place.
- Give every scientific policy change a new physics version and cache
  namespace.
- Keep point routing, point corrections, and uncertainty calibration separate
  in code and reporting.
- Compare a proposed policy with the frozen release on all 364 TECRDB routes,
  all 300 ModelSEED routes, warm-cache point estimates, cycle closure,
  independent physical evidence, and compute cost.
- Do not adopt a policy solely because it improves TECRDB MAE.
- Record rejected hypotheses and regressions as well as successful changes.
