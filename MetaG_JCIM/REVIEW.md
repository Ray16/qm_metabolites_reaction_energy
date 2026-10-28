# Workflow review: current revision

Current source is the **0.2.0.dev22 coordinate-integrity candidate**. It fixes
RMSD probe mutation and optional CPCM-X partial-result admission, with 1,081 tests
passing. The complete higher-level neutral/monoanion control has all24 reference and8 UMA
points, both diffuse reference bases, convergence logs and T1 diagnostics; its
limited electronic scope does not validate biochemical accuracy. Fresh component/reference checks are in
`analysis/highlevel_components_20261022_01/`. The dev21 release and manuscript
revision described below remain frozen; no new whole-panel accuracy or interval
claim is attached to these fixes.

The colleague's review has produced a corrected manuscript and a separately
versioned **0.2.0.dev21** cleanup candidate. The current
[point-by-point response](analysis/highlevel_components_20261022_01/RESPONSE_CURRENT.md)
distinguishes completed corrections from experiments still required.

The default reference builder now retains **39 labels**, of which **29** have
available evaluated-estimator predictions. Exactly 298.15 K and reported ionic
strength leave **seven labels / four predictions**. These give descriptive MAEs
7.47 and 5.01 kJ/mol on different cohorts; they are not an estimator improvement
or independent validation. Historical **9.17/251** is separately labelled. On
those same targets the original estimator gives **9.19**; its excluded 113 give
**10.32**. All 364 outcomes remain visible, with 69.0% estimate availability.

The complete neutral component panel gives **14/14 estimates and 306 recorded
minima**. Every reported reference center is outside the range reachable by
changing only electronic/thermal weights on the fixed ALPB basis. The new
`compare-states` diagnostic is maintained code, not a fitted point correction.

| Check | Evidence |
| --- | --- |
| Full candidate tests | [1,071 passed](analysis/reviewer_revision_20261022_01/tests_revision_full.log); no fresh GPU inference |
| Active-helper and flag parity | [Code cleanup report](analysis/reviewer_revision_20261022_01/code_cleanup_parity.json) and 77-case before/after flag audit |
| Default reference-only asset change | [Promotion proof](analysis/reviewer_revision_20261022_01/reference_asset_promotion.json); all 364 nonexperimental inputs identical |
| Installed CPU candidate | [Candidate release](analysis/reviewer_revision_20261022_01/release/); distinct from the evaluated snapshot |
| Manuscript evidence | [Eight checkers](analysis/reviewer_revision_20261022_01/manuscript_checks.json), current reference intersections and historical records checked separately |
| Original evaluated workflow | [Immutable release](analysis/workflow_release_20261022_01/release/manifest.json); 1,065 tests, four-arm replay and two fresh cold inputs retain that source identity |

Independent biochemical validation, broader basis-converged metabolite controls,
source-defined anomer pools and calibrated predictive intervals remain open. The preliminary policy inventory is not a complete reconstructed
decision history. The current paper makes a conditional physical-diagnosis
claim and does not claim validated flux-constraint deployment.

Authors, affiliations, correspondence, public repository/archival identifiers
and owner-approved licensing remain pending at the author's request. The
manuscript lives in its separate Overleaf-synchronized repository. The
[previous review](analysis/reviewer_revision_20261022_01/before_code/REVIEW.md)
is preserved as history, not the status of the current checkout.
