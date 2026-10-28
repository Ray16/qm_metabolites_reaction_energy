# Scientific Policy Audit

Status: the original audit below describes the frozen `2026-10-01c` comparison.
The following production safeguards were added on 2026-10-06.

## Generalization safeguards (2026-10-06)

- Production routing now rejects `ANCHOR_CORRECT`, `POLYACID_PKA`, and
  `WATER_REF_HYDROLYASE` when enabled. Historical helper functions and fingerprint
  fields remain for archive inspection; these switches cannot enable corrections
  through `route_reaction` / `score_reaction`. No research bypass is provided.
- Both pKa providers reject unsupported P coordination/substitution, including
  phosphonates and phosphites, instead of borrowing phosphate parameters. Neutral
  P-OH drawings and ionized drawings both undergo the check.
- The P-N ladder is now quarantined: its first entry lacks a defensible phosphate
  assignment. A class constant with wrong-site provenance cannot be replaced by
  fitting a reaction residual; independent state-specific parameters are needed.
- Table ladders cannot be padded with invented/repeated pKas or silently drop
  excess sites. The site provider also rejects excess sites instead of clamping
  their parameter indices. Supported numerical constants were not changed.
- The `pka_domain` fingerprint records this changed applicability boundary;
  historical uncertainty calibration is not automatically claimed for it.
- `python -m metag.tools.preflight INPUT.json -o REPORT.json` audits original
  structures and routed cores without QM. It records effective routing/pKa
  switches, source hashes, input hash, references, pKas, errors, and known gaps.
  It checks the same JSON reaction schema as scoring, with arbitrary identifiers.

The 300-reaction / 588-structure ModelSEED panel produces **265 review, 34 no
known gap detected, and 1 blocked**. The blocked phosphonate reaction previously
received phosphate-class pKas. Among unique input structures, 117 contain aromatic
nitrogen needing speciation review, 238 have multiple acid sites treated with the
independent-site approximation, and 9 retain anions outside the recognition rules.
Counts overlap. The 26 references retaining net charge include permanent ions;
net charge alone does not establish a missing ionization state. This is a panel
preflight, not full-ModelSEED validation. The full report is
[`modelseed_preflight.json`](../analysis/code_review/modelseed_preflight.json).

### Remaining scientific work

| Priority | Limitation | Required replacement/validation |
| --- | --- | --- |
| 1 | Aromatic N, guanidines/amidines, and other unmodeled ionizations | Consistent protonation-state enumeration with a provider that covers the actual reference microstate; validate alternate input protonation drawings |
| 1 | Phosphonates/phosphites now explicitly unsupported | Independently sourced or predicted microstate pKas and domain tests; do not copy phosphate constants |
| 1 | P-N ladder quarantined | Obtain verified microstate transitions, including phosphate/guanidine coupling, before restoring automated coverage |
| 2 | Scalar class pKas transferred to diverse substituted compounds | Independent pKa validation and uncertainty; changing a constant into a configuration file does not solve transferability |
| 2 | Macroscopic ladders flattened into independent-site terms | Binding-polynomial calculation over consistent protonation microstates, including coupled sites |
| 2 | Core, truncation, and hydration rules chosen during benchmark development | External chemistry splits and full-molecule/radius comparisons; absence of reaction IDs is insufficient evidence of transferability |
| 3 | Metals, binding and unmodeled tautomers | Explicit domain definition and corresponding thermodynamic state models |

The preflight is deliberately conservative and incomplete: `no_known_gap_detected`
is not a certificate of complete speciation, accuracy, or UMA applicability. It
reports original-molecule gaps even if a core transformation removes that group.
Table pKas and the optional site model remain approximations; no reaction ΔG
residuals were fitted in this cleanup. Ring-chain work remains experimental/off.

## Original frozen-baseline audit

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

### Experimental reference rebuild (2026-10-06)

The active openTECR panel retains all 364 reaction structures. Under the new
reference policy, 54 reactions retain labels and 310 have `exp=[]` with recorded
exclusion reasons. These counts describe reference availability, not prediction
coverage or an accuracy improvement. Of the 54 retained labels, 44 changed after
measurement selection. Historical benchmark scores and uncertainty calibration
artifacts have not been recomputed on this reference set.

Unknown free Mg activity is no longer replaced with an Mg-free assumption.
Total Mg in a recipe is insufficient to determine free Mg activity. Unresolved
curation flags, invalid equilibrium constants, and species-level constants are
excluded by generic rules, independent of reaction identity or prediction error.
No disputed equilibrium constant is corrected by choosing the value that best
matches the model. Recovering excluded labels requires primary-source evidence
or sufficient solution composition and binding data to determine free activity.

Condition corrections use only differences in the eQuilibrator dissociation
transform; fitted reaction formation energies do not enter. Remaining limitations
are explicit: missing ionic strength defaults to 0.25 M, and transformations
retain each measurement's temperature. Labels are therefore not necessarily
references at the pipeline's 298.15 K. Use `--require-reported-I` to exclude
unknown ionic strength as well. Temperature matching remains necessary for a
strict comparison at 298.15 K.

Reproduce from the MetaG_new directory in an environment with eQuilibrator:

```bash
PYTHONPATH=src python -m metag.tools.build_reference \
  --source artifacts/benchmark/opentecr_source \
  --modelseed ../../ModelSEEDDatabase \
  --reactions artifacts/benchmark/curated/legacy_reactions_opentecr_std.json \
  --output artifacts/benchmark/curated/reactions_opentecr_std.json \
  --report artifacts/benchmark/curated/reference_report.json
```

The report records source rows, inclusion/exclusion decisions, condition
corrections, mapping candidates, and source/code/database hashes. The legacy
input is archived alongside it. Pipeline outputs carry reference provenance
separately from the point prediction.

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
