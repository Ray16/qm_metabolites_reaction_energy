# Runtime configuration and result contract

Set configuration before importing the scoring pipeline. Defaults and flag
parsing are centralized in [config.py](../src/metag/config.py); the effective
configuration is retained with each result. Scientific ablations need a separate
evaluation, rather than selecting settings from reaction residuals.

| Variable | Default or role |
|---|---|
| `RXN_FILE` | Packaged reaction inputs, or a caller-supplied reaction JSON file |
| `METAG_OUT` | `./metag_out`; CLI results |
| `METAG_CACHE` | `./.metag_cache/species_cache`; species records |
| `XTB_BIN` | External xTB executable; set its absolute path |
| `UMA_MODEL` | `uma-s-1p2p1`; must match the loaded checkpoint |
| `UMA_HESSIAN` | `autograd`; `fd` selects the historical finite-difference implementation |
| `SOLV_MODEL` | `alpb` |
| `PKA_MODEL` | `table`; the optional `site` model has a restricted domain |
| `PKA_SITE_MODEL_FILE` | Optional site-model artifact, identified by its content hash |
| `CONFORMER_SEED_OFFSET` | `0`; an independent search must use a nonoverlapping seed window |
| `CONV_TOL`, `CONV_HITS`, `CONV_MAX` | `2.5` kJ/mol, `2`, `8`; provisional plateau rule |
| `THERMAL_VALIDATION_MAX` | `64`; unresolved or excess candidates withhold the species estimate |
| `SPECIES_CACHE` | `1`; `0` disables persistent species reads and writes |
| `METAG_ROUTING_MODE` | `reaction`, `full_molecule` or `compound_reference` |

The sampler chooses budgets from molecular flexibility. Legacy `--seeds`,
`--keep` and `--pool` arguments do not define independent replicas. The opt-in
`--sampling-profile thorough` increases the search budget and tightens the
plateau rule; it has no established accuracy advantage.

## Routing and molecular state ownership

`reaction` applies the configured structural cores and spectator cuts.
`full_molecule` disables those structural reductions while retaining the
protonation, hydration and solvation policy. `compound_reference` additionally
builds references from each compound with owned acid/base terms and explicit
proton bookkeeping. Permanent ions may retain charge. Unsupported references
and conflicting caller models fail explicitly.

The stereo gate covers original and routed structures. Unspecified
nonanomeric configuration, double-bond configuration and unsupported relative
stereoisomer groups remain unresolved. Resonance normalization prevents
equivalent nonbridging phosphate oxygens from creating false stereo problems.
Source-declared [reducing-sugar pools](reducing-sugar-pools.md) cover their
supported free reducing centre while preserving every other assigned centre.

The [source acid registry](source-backed-protonation.md) applies only to exact
full-reference matches, with temperature, activity scale, stereo and ownership
checks. Macrostate ladders are not transferred to arbitrary truncated cores.
Acid-only routes can retain protonated amines: the reference is not universally
neutral. Its actual charge and owned transform enter reaction assembly once.

## Results and uncertainty

The CLI checkpoints all outcomes after each reaction in
`METAG_OUT/unified_pipeline_<key-or-all>.json`.

| Field | Meaning |
|---|---|
| `dG`, `dG_raw` | Reported and unanchored routed reaction energies, kJ/mol |
| `conditions`, `config` | Target conditions and complete effective estimator identity |
| `original_species`, `species_scored` | Caller chemistry and evaluated representation |
| `stages`, `routes` | Energy decomposition, reference/state models and decisions |
| `suspect` | Reason an unsupported or failed estimate is withheld |
| `U_samp` | Conditional conformer-search diagnostic propagated through the sampled partitions |
| `sigma_pred`, `ci95` | Withheld when no matching calibration applies |
| `ci_info.coverage_calibrated` | Calibration/applicability decision, not a universal validity certificate |
| `uncertainty_diagnostic` | Historical widths retained as diagnostics |
| `sign_status`, `unresolved` | Sign-applicability status; unavailable uncertainty is not a confident sign |

`dG: null` is unavailable, never zero. All attempted and structurally withheld
targets belong in coverage reports. `U_samp` comes from dependent provisional
search estimates; it is not a calibrated standard error, a missing-state bound
or physical model uncertainty. The current release supplies no validated
reaction prediction intervals or probabilistic flux constraints.

## Cache and producer identity

Persistent species keys include canonical structure, charge, spin when needed,
sampling and solvent settings, thermal policy and raw-physics source identity.
Producer records identify software, imported module versions, xTB executable
bytes, checkpoint, elemental references and inference settings. The complete
reaction-estimator source has a separate identity. Older entries lacking the
required producer identity do not enter live scoring.

The native loader supplies model provenance. A custom predictor without the
complete identity can be scored, but persistent caching and water-reference
memoization are disabled. Restart after changing code, installed dependencies,
weights or inference settings. Keep original raw records and their producer
identities during diagnostic replays; assigning a new source label to an old
record is not a compatibility proof.

Model changes need configuration-matched evaluation and new uncertainty
calibration. Source dates, numerical completion, chemical identity and
experimental reference admission are separate evidence gates.
