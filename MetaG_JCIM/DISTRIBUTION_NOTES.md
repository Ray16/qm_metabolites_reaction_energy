# Distribution inventory — 10 October 2026

**Historical inventory.** The dated release descriptions below are not the
current estimator or submission status. Consult the [official-profile work
status](analysis/official_default_20261024_01/WORK_STATUS_CURRENT.json) and
[review work record](analysis/remaining_review_20261022_01/WORK_STATUS.txt)
for the current evaluation and outstanding gates. Author and licensing
decisions remain with the project owners.

The dev20 reviewer release includes a fifth packaged JSON asset,
`source_protonation_models.json`, containing exact-reference acid ladders with
source locators, artifact hashes, conditions, scale conversions and uncertainty
limitations. Include it in the final byte inventory and attribution review.
The current release is indexed under `analysis/workflow_release_20261022_01/`:
1,065 source tests pass, all 108 installed source/data files match, and fresh
integration uses the same frozen source. `REVIEW.md` separates software checks
from unproven independent accuracy and interval coverage. The physical studies
remain under `analysis/state_workflow_20261021_01/`. Earlier wheels below retain
their original verification scope and are not relabelled as the current release.

21 October update: dev18 adds CPU-only controlled component and declared-state
ensemble comparisons. Its local reviewer wheel matches all 102 packaged source/
data files in an isolated target on the existing minimal CPU environment, with
passing dependencies and no optional QM packages. The source suite reports
996 passed and one opt-in GPU skip. Production molecular-energy source is
unchanged from dev17. See `analysis/publication_readiness_20261021_01/release/`.
The completed dev17 benchmark has 251/364 estimates and historical-reference
MAE9.251kJ/mol; this is development evidence. The state diagnostic and software
release do not establish independent experimental accuracy. Existing author,
copyright, project-license and asset-attribution decisions remain required.

18 October update: the current dev17 candidate preserves source stereochemistry
through cysteinyl cores and rejects ambiguous generic cuts. Its reviewer wheel,
967-test result and two-host clean CPU installation checks are recorded under
`analysis/publication_sprint_20261018_01/routing_recovery_01/release/`. The fresh
dev16 cold full-panel evaluation and separate dev17 numerical follow-up retain
their own immutable source identities, costs and unresolved scientific gates.


Current software checkpoint, 17 October: dev16 adds source-backed protonation
admission and shared-state consistency checks, with 956 passing tests and one
opt-in GPU skip. Its release verification is recorded under
`analysis/pipeline_enhancement_20261017_01/release/`. The version-specific
installation records below remain historical. The complete biochemical
benchmark still belongs to dev14, and the citrate source-model execution check
belongs to dev15; neither is relabelled as a dev16 numerical evaluation.

The current package contains MetaG source and five packaged JSON artifacts. It does not include UMA weights, xTB binaries, or the manuscript's full reference snapshots. Installing the QM extra does not grant model access or install xTB.

UMA weights use the [FAIR Chemistry License](https://huggingface.co/facebook/UMA/blob/main/LICENSE), with a linked acceptable-use policy and publication acknowledgment requirements. The project must retain that dependency's own terms and acquisition instructions; MetaG's eventual code license cannot replace them. This inventory does not certify compliance for every redistribution scenario.

No project LICENSE file or license field is currently declared for this development package. Before public distribution, the authors need to confirm ownership, choose the project license, and record attribution. Review the provenance and redistribution terms of the bundled reaction and calibration JSON artifacts separately. The source snapshots and model weights used for this review remain local and have not been published.

Dev13 is an earlier verified development wheel; its installed verification is recorded in the release manifest. Both installed environments match its maintained source and pass dependency checks. It adds accepted-conformer geometry/hash exports and a CPUXYZexport utility, retains explicit gas enthalpy/entropy decomposition and temperature/pressure inputs for supplied-Hessian checks, retains an optional thorough search preset (no accuracy guarantee), withholds inapplicable confidence intervals, adds analytic protonation sensitivities, and retains the optional owned-compound reference mode; the reference model and its class pKas remain conditional. The full primary ferulate paper is retained locally for reproducibility, is not bundled in the wheel, and has not been redistributed. See `analysis/publication_hardening_20261009/release_manifest_dev13.json` for current evidence; the dated installation history below retains its original versions.

The earlier dev6 wheel was also installed and source-verified in both environments, with passing dependency checks. Its full-molecule routing preset and CPU panel gate are tested without importing optional QM libraries. `analysis/publication_hardening_20261009/release_manifest_dev6.json` records this interface release; the numerical runs below retain their original versions.

Clean installation verification now covers the CPU thermochemistry extra and a separate full QM environment with constrained NumPy 2.4.6, Torch 2.8.0, fairchem-core 2.21.0, ASE 3.29.0 and RDKit 2026.3.5. Both installed dev5 packages match the maintained source fingerprints and pass dependency checks. The full environment completed real CUDA water energy/force/Hessian/RRHO/xTB checks and public AMP scoring under dev4/physics 10d; dev5 changes lazy optional imports and dependency packaging, with an unchanged correction on the same real DFT Hessian. The minimal CPU extra reproduces that correction without Torch or fairchem installed. See `analysis/publication_hardening_20261009/release_manifest_dev5.json` for version-specific evidence. These are verified local installations, not a guarantee across operating systems or GPU architectures. Actual imported module versions are recorded separately from distribution metadata.


The current [asset inventory](analysis/publication_hardening_20261009/distribution_inventory_dev13.json)
records all four packaged JSON hashes and their provenance. The upstream
[openTECR data repository](https://raw.githubusercontent.com/opentecr/opentecr-data/main/LICENSE)
declares CC0-1.0, consistent with the pinned snapshot documentation. The
[ModelSEED data license](https://github.com/ModelSEED/ModelSEEDDatabase/blob/master/LICENSE)
declares CC BY 4.0 except where otherwise noted and retains component-source
conditions. The derived reaction file combines measurement data, structures,
mappings and transformed references; keep those source identities and terms
separate from MetaG's project code license. The code copyright holder and
preferred code license have been requested from the author; no license or
public distribution has been inferred.

Dev12 adds strict complete sampled-ensemble SMD reweighting and a CPU file tool,
with installed reproduction of both hexenol species. It also honors disabled
neutral-reference fallback after molecular-identity failure and retains failure
provenance. The current suite passes 971 tests with one opt-in GPU test skipped.
No solvent default or class pKa is promoted. Proposed notices and concrete author
decisions are staged in `release/`; code ownership/license and final authorship
remain unanswered, and no public distribution has occurred.

Dev13 prevents automatic PH0 retries from overriding disabled isomerase/base
gates and prevents a zwitterion guard from authorizing a pure anion reference
change. Seven route-policy regression cases accompany the refinement tests;
the full suite passes 974 tests and one opt-in GPU skip. Both installed
environments pass source equality and dependency checks. A fresh installed
dev13 CPU process reproduces all 77 prior conformer partitions. Scientific
records preserve their dev11/dev12 producer versions.
