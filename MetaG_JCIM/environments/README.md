# GPU runtime profiles

The **0.2.0.dev24 default** is the official Nitro-preferred calculator. Supported
single-system optimization calls use Nitro; batched calls, partition/ranking
energies, endpoint convergence checks and autograd Hessians use official general
FP32. Approximate optimization cannot certify a minimum: any required FP32
refinement stays inside the original remaining step budget. Results record the
requested profile and actual per-invocation mode/fallback counts separately.

The official SDK is pinned to commit
`645af7d03e91138ef07e51de387316c9e2e152dc`, with Torch 2.13.0. Model IDs and
composition averaging are not overridden. The new default has passed the full
unit suite (1129 passes and one optional skip) and real V100/H100 E/F, batch,
convergence, Hessian and RRHO contracts. Both complete reaction inventories and
the updated manuscript are available. The separate current-profile finite-policy
selection audit remains unfinished; runtime validation does not establish
experimental accuracy or calibrated uncertainty.

| Profile | Runtime | Verification |
|---|---|---|
| New default, V100 32 GB | Python3.12, CUDA12.6; official general FP32 fallback | Runtime contract and all24 external-source reaction predictions completed |
| New default, H100 | Python3.11, CUDA13; Nitro preferred with FP32 finals | Runtime contract and full364-input biochemical inventory completed |
| Legacy reproduction | FairChem2.21/Torch2.8; standard UMA | Version-bound historical cold workflow and numerical evidence |

## Install the matching official runtime

Use Python 3.12 for the recorded V100 runtime. Install its CUDA wheel first:

```bash
python -m pip install 'torch==2.13.0' --index-url https://download.pytorch.org/whl/cu126
python -m pip install -c environments/qm-official-cu126.txt -e '.[qm]'
export XTB_BIN=/absolute/path/to/xtb
```

For compatible H100 drivers, use the CUDA13 wheel index and
[qm-official-cu130.txt](qm-official-cu130.txt). The CUDA13 Torch build lacks
`sm_70` kernels, so switching Nitro off in that environment does not provide V100
support. V100 uses [qm-official-cu126.txt](qm-official-cu126.txt). Accessible model
weights, xTB and the recorded process library paths are also required. Existing
pinned shared environments are reused without installing this package into them;
validation workers import their frozen MetaG source explicitly.

`UMA_RUNTIME_PROFILE=nitro` is the default. `official_fp32` explicitly disables
Nitro while retaining current official model semantics. The source/profile,
checkpoint, reference, SDK-code and numerical-policy identities enter provenance;
old values are not relabelled or silently reused under a new profile.

See the [new default validation study](../analysis/official_default_20261024_01/)
for frozen source, executable manifests, tests and actual runtime outcomes.

## Frozen legacy-profile reproduction

The earlier FairChem2.21/Torch2.8 profile is retained for version-bound
reproduction:

```bash
python -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c environments/qm-linux-cuda128.txt -e '.[qm-legacy]'
export UMA_RUNTIME_PROFILE=legacy
export XTB_BIN=/absolute/path/to/xtb
```

The legacy profile requires its legacy SDK. It is not emulated by spoofing a model
ID or forcing the older composition rule into the newer SDK.

The frozen legacy V100 device checks use the same checkpoint, elemental references, inference
settings and scientific source as their matching legacy H100 control. These historical checks
retain their original source identities; they are not results of the newer official SDK. Electronic,
solvation and context-matched thermal components pass the original tolerances
on four literal geometries with charges 0, +1, +2 and -1. The original strict-only
thermal mismatch and the subsequent unchanged imaginary-mode-policy diagnosis
are both retained in the
[device qualification](../analysis/remaining_review_20261022_01/selection_current_01/available_backend_context_validation_v2.json).
Fresh molecular-state jobs also execute on V100 in this profile.

The full reaction check is opt-in and must run inside a GPU allocation with a
fresh private species-cache directory:

```bash
export UMA_RUNTIME_PROFILE=legacy
export METAG_RUN_GPU_TESTS=1
export METAG_CACHE=/absolute/path/to/a/new/empty/species-cache
export METAG_SMOKE_REPORT=/absolute/path/to/a/new/smoke-report.json
python tests/integration/test_pipeline_smoke.py
```

The test runs three declared reactions through the real production scorer and
checks that the fourth, stereochemically unresolved input is withheld. It checks
autograd selection, actual model provenance and the current unavailable-interval
contract, without using experimental labels. The direct invocation needs no
pytest installation. The cold V100 run passed: three reaction estimates and the
expected stereochemical refusal, with nine fresh species records and 104 retained
minima carrying electronic, solvation and autograd thermal components. All three
reaction values match the H100 values at their reported precision. See
[the completion verification](../analysis/v100_runtime_20261024_01/completion_verification.json)
for the exact configuration, executed inputs, caches and limits of this check.

Tests so far use a **32 GB V100**. A 16 GB V100 needs a separate memory check.
`UMA_ENERGY_CHUNK`, `UMA_FIRE_CHUNK` and `UMA_HESS_ROWS` control memory use without
removing candidate conformers or changing the requested Hessian method. Record
the effective configuration and verify any changed execution profile.

## Migration evidence and limits

The shared experimental environment uses FairChem commit
`645af7d03e91138ef07e51de387316c9e2e152dc` and Torch 2.13.0+cu130. Its dependency
lock and runtime library path are recorded in the
[setup evidence](../analysis/uma_nitro_20261023_01/shared_setup_verification.json).
Six single-system calculations executed in actual Nitro mode on H100 with the
same registered checkpoint and reference hashes.

That initial execution check alone did not qualify a production replacement. Nitro energy
and force differences from the current FP32 baseline exceed the prespecified
reproduction criteria. Batch and
requested-Hessian tests explicitly record their non-Nitro fallback modes. See
[the first H100 receipt](../analysis/uma_nitro_20261023_01/first_h100_runtime_receipt.json).
Existing energies retain their original producer identities and are not
relabelled as Nitro results. The later integration retains the V100 profile
and records the backend actually used for each stage.

An explicitly non-Nitro configuration of the newer API also executed four
electronic points and a requested water Hessian on V100 using the preserved
CUDA12.6 trial wheel. Turning merging and compilation off did not eliminate
the legacy-energy drift. These
[diagnostic results](../analysis/uma_nitro_20261023_01/v100_standard_candidate_01/completion_verification.json)
support API execution, rather than production numerical qualification.

An independent six-case charge/spin diagnostic subsequently isolated the
legacy/current FP32 discrepancy to composition-averaging semantics. With the
genuine model ID unchanged, temporarily changing only that averaging decision
reproduced legacy energies and forces within the original tolerances; restoring
the official method restored its predictions. See
[the causal comparison](../analysis/uma_nitro_20261023_01/composition_semantics_metag_01/completion_verification.json).
This diagnostic is not a production override or a migration qualification.
The genuine newer official model requires validation as its own profile;
spoofing its model ID or forcing legacy behavior is not the production strategy.
Earlier publication results retain their recorded SDK semantics. The separate
Nitro-versus-current-FP32 precision comparison remains applicable.

The six recorded unmodified official-current FP32 calculations also agree across
H100 and V100 within the original energy/force tolerances, using their compatible
CUDA13 and CUDA12.6 runtimes. See
[the cross-device check](../analysis/uma_nitro_20261023_01/official_FP32_cross_device_01.json).
This establishes fixed-geometry E/F portability. The subsequent workflow and
thermochemistry checks are described above; they do not by themselves establish
independent experimental accuracy.
