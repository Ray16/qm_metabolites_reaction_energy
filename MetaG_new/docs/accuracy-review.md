# Accuracy review, October 6, 2026

This review fixes implementation and diagnostic errors in place. No numerical
pKa values were replaced, no reaction-energy anchors were enabled, and no
parameters were fitted to benchmark errors. It does not demonstrate a reduction
of the large-error tail.

## Fixes

- **Solvent relaxation was overwritten.** `implicit_G` constructed an aqueous
  relaxed ensemble and then replaced it during gas thermal processing. It now
  validates gas minima first and applies solvent relaxation afterward. Only
  converged, finite, connectivity-preserving aqueous structures contribute.
  A completely failed relaxation fails the species; partial ensembles are
  flagged and not cached. The changed optional estimator has a distinct cache
  fingerprint, so old solvent-relaxed entries cannot silently be reused.
- **Neutral-base bookkeeping depended on the input drawing.** Within the
  neutral-microspecies routine, neutral NH3 and saturated aliphatic amines now
  retain the same existing protonation terms as their conjugate-acid drawings.
  Neutral tertiary amino acids are recognized consistently too. Other nitrogen
  chemistries were not assigned new generic constants.
- **Base constants were missing from the calibration fingerprint.** Ammonia,
  amino-acid amine, generic amine, imidazole and guanidinium constants now
  participate, preventing a pKa edit from silently retaining calibration status.
- **The two-state hydration partition could overflow.** Its log-sum-exp now
  subtracts the maximum exponent. Ordinary inputs are mathematically unchanged.
- **Duplicate microstate names could corrupt populations.** The partition
  routine now rejects duplicate labels instead of counting both energies while
  overwriting one population.
- **The tautomer pilot could lose its reference state.** The input tautomer is
  always retained before limiting alternatives. A missing or failed input is
  reported as failure, never replaced by the lowest alternative. Scoring now
  uses the production estimator, including conformer deduplication, solvation
  and thermal terms. Partial selected sets are labeled explicitly.

`SOLV_RELAX` remains an optional diagnostic, off in production. Its revised
thermal treatment carries RRHO corrections from the starting gas basins;
it does not compute aqueous Hessians or search conformations that cannot be
reached from gas minima. Its sampling uncertainty describes gas-ensemble
convergence only. These limitations preclude calling it a complete solution to
hydratase or zwitterion sampling errors.

## Validation and preservation of results

The full test suite passes: **219 tests**. New tests cover the bugs above with
mocked backends, input-representation equivalence, fractional stoichiometry,
and configuration fingerprints. No GPU calculations were launched by this review.

`analysis/accuracy_review/audit.py` replays all **364** bundled openTECR inputs
using an explicitly supplied existing species cache, forbids new QM, and
disables cache writes. It compares corrected base bookkeeping with the
pre-fix rules using exactly the same cache. The report also checks stages,
routes, scored species, sampling uncertainties, uncertainty metadata, and
failure flags; compares cycle closure; and compares routing on the **300**
bundled ModelSEED inputs. See `analysis/accuracy_review/report.json` for the
results and a ranked list of all benchmark residuals.

There were zero replay failures, zero changes in the compared non-point fields,
zero calibration-configuration mismatches, and zero ModelSEED routing changes.
The 49-cycle closure report is identical before and after the fix; its uniformly
weighted RMS residual remains 1.52 kJ/mol.

All 364 replayed point estimates are unchanged by the base fix. MAE remains
**9.54 kJ/mol**. Relative to archived production, 25 rounded estimates differ
by 0.1 kJ/mol: archived MAE 9.54170, replay MAE 9.54363. Those differences also
occur with the pre-fix rules, so they are not improvements or regressions from
this fix. Exact reproduction of the archived run still requires its immutable
cache, not merely a matching code configuration.

The live uncertainty artifact retains all calibration statistics, widths and
multipliers. Only configuration metadata and a revalidation note changed.
The original artifact is preserved byte-for-byte at
`tests/regression/fixtures/sigma_class_calibrated.json`, where its frozen hash
is still tested. This preserves historical evidence without weakening the live
configuration check. Paired consistency is not an independent validation of
coverage outside TECRDB. The 300-input check validates routing, not a new
ModelSEED energy sweep.

Example replay, from the package root:

```bash
PYTHONPATH=src python analysis/accuracy_review/audit.py \
  --cache /path/to/species/cache \
  --water-reference /path/to/water_ref_G_expt.json
```

The cache and water reference are explicitly supplied validation inputs; the
script does not make the installed package depend on a sibling source tree.

## What remains scientifically unresolved

The [pKa provenance audit](pka-provenance.md) distinguishes measured compound
constants from class approximations and unverified citations. In particular,
the P–N ladder's first value is not source-validated here. The prior phosphagen
half-reaction comparison uses approximate reference values and truncated
species; it does not uniquely eliminate protonation, Mg speciation or
truncation as contributors. Nonconvergent small-cluster calculations establish
that the tested protocol failed, not that all physical treatments must fail.

The old `analysis/glycosyl_tautomers/pilot_results.json` predates the repaired
pilot and has not been regenerated. It cannot conclusively rule out tautomer
effects: its guanine reference was not scored correctly, and its estimator
differed from production. A corrected pilot must also test the reaction's
other species; lowering one product alone does not prove a reaction correction.
Sugar ring forms and anomers remain untested by this pilot.

Class-average residual signs alone cannot identify a physical mechanism.
Transaminases exchange amino groups between substrates; a universal amino-acid
offset can cancel. Both sides' species terms and protonation transitions must
be checked before assigning the residual to zwitterions. Reaction-dependent
routing also remains a source of possible thermodynamic inconsistency.

An external pKa lookup should first run in a comparison report, with the data
snapshot pinned. Each adopted entry needs a matching structure, protonation
transition, micro/macroscopic meaning, solvent, temperature and ionic-strength
convention. Missing matches must be explicit. Before adoption, rerun the full
benchmark, large-error cases, representation/cycle checks, ModelSEED routing
and affected uncertainty calibration. If predictions change, regenerate their
derived figures and tables. Agreement with existing reaction errors must not
be used to select the pKa itself.
