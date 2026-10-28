# Reproduce the development examples

These inputs demonstrate actual exercised scoring paths. They are development cases, not an independent benchmark or a promise of a particular output on another source/runtime.

From the package root, preflight without QM dependencies:

```bash
python -m metag.tools.preflight examples/amp_hydrolysis.json -o amp_preflight.json
python -m metag.tools.preflight examples/citrate_isocitrate.json -o citrate_preflight.json
```

Run inside an allocated GPU environment with the documented UMA weights and xTB executable:

```bash
RXN_FILE="$PWD/examples/amp_hydrolysis.json" METAG_OUT="$PWD/amp_results" metag --only amp_hydrolysis
```

For replicate searches, run separate processes with `CONFORMER_SEED_OFFSET=0` and `CONFORMER_SEED_OFFSET=1000`, separate output directories, and the same frozen source/configuration. The implicit search window defaults to eight consecutive seeds. Explicit-cluster sampling does not become independent merely by changing this offset. Use the [replicate audit contract](../analysis/physical_improvement_20261010/SAMPLING_AUDIT_USAGE.md), retain every failed outcome, and declare the expected replicas in advance.

The frozen 10c AMP public-API calculation returned −11.2 kJ/mol after recovering one minimum using LBFGS; total gated wall time was 423.4 seconds. The separate source-derived conditional reference is −14.2787 kJ/mol. This is one development example and its error is not generalization evidence. [Prediction and comparison](../analysis/minimum_recovery_20261010/full_amp_10c/comparison.json).

The frozen 10d citrate/D-threo-isocitrate runs returned 5.9 and 4.3 kJ/mol. Both preserve the declared stereochemistry, with observed range 1.6 kJ/mol; the twofold equilibrium-ratio tolerance at 298.15 K is 1.7183 kJ/mol. This finite-repeat check is not proof of complete sampling or experimental accuracy. [Run summary](../analysis/chemical_identity_20261010/curated_run_01/summary.json).

The isocitrate structure is [ChEBI:15562](https://www.ebi.ac.uk/chebi/CHEBI:15562), consistent with the [IUBMB EC 4.2.1.3 definition](https://iubmb.qmul.ac.uk/enzyme/EC4/2/1/3.html). Earlier archived inputs omitted its two stereocenters; those numerical results are retained separately. The old screened experimental label is deliberately absent: source inspection found a total-versus-free magnesium ambiguity requiring reconstruction before accuracy scoring.


## Stereo-sensitive controls

`stereo_controls.json` supplies balanced, explicitly assigned E-to-Z reactions
for 2-butene and 1,2-dichloroethene. There are no experimental labels in the input.
These calculations concern microscopic aqueous species. Gas Gibbs and gas
enthalpy references in the component report are different quantities.

Check structures and a shared compound-reference representation on CPU:

```bash
METAG_ROUTING_MODE=compound_reference python -m metag.tools.preflight examples/stereo_controls.json -o stereo_preflight.json
METAG_ROUTING_MODE=compound_reference python -m metag.tools.audit_routes examples/stereo_controls.json -o stereo_routes.json
```

After installing the documented QM environment, provisioning UMA access and
setting `XTB_BIN`, run within an allocated GPU environment:

```bash
METAG_ROUTING_MODE=compound_reference RXN_FILE="$PWD/examples/stereo_controls.json" METAG_OUT="$PWD/stereo_seed0" METAG_CACHE="$PWD/stereo_cache_seed0" CONFORMER_SEED_OFFSET=0 metag --sampling-profile thorough --require-consistent-panel
METAG_ROUTING_MODE=compound_reference RXN_FILE="$PWD/examples/stereo_controls.json" METAG_OUT="$PWD/stereo_seed1000" METAG_CACHE="$PWD/stereo_cache_seed1000" CONFORMER_SEED_OFFSET=1000 metag --sampling-profile thorough --require-consistent-panel
```

Use separate cold output/cache directories when reproducing sampling costs;
shared raw-state caches change the cost experiment. Inspect both results and
failures. Two agreeing seeds are a repeatability check, not an error bar or proof
of complete state coverage. The package's development license/data decisions
remain pending; this example does not grant access to external models.
