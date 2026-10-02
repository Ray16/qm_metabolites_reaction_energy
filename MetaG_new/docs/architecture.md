# Architecture

## Package Map

The import package is `metag`; the repository is named `MetaG_new`.

| Path | Responsibility |
|---|---|
| `src/metag/chem.py` | Reaction balance and chemical representation helpers |
| `src/metag/reactions.py` | Validated reaction and species input models |
| `src/metag/routing/` | Structure-based route selection and transformations |
| `src/metag/energetics/` | Conformers, UMA, thermal terms, solvation, and species cache |
| `src/metag/uncertainty.py` | Calibrated prediction uncertainty |
| `src/metag/validation/` | Reusable parity, provenance, and result audits |
| `src/metag/execution/` | Atomic records and exclusive distributed task claims |
| `src/metag/tools/` | Maintained calibration and cycle-closure utilities |
| `src/metag/pipeline.py` | Frozen orchestration facade and compatibility CLI |
| `configs/` | Versioned scientific-policy records |
| `tests/unit/` | Fast pure-logic and mocked-backend tests |
| `tests/integration/` | Full-stack tests requiring CUDA, UMA, or xtb |
| `tests/regression/` | Frozen release fixtures and numerical parity tests |
| `tests/performance/` | Reproducible throughput and scaling benchmarks |

Generated caches, sweep outputs, figures, and manuscripts are not package
source. They must be written to caller-selected external directories.

## Dependency Direction

The intended dependency flow is:

```text
chemistry/domain models
        |
        +--> routing/speciation
        |
        +--> conformers/energetics/solvation
                       |
                       +--> cache
        |
        +--> reaction assembly --> uncertainty
                       |
                       +--> execution and CLI
```

Scientific decisions belong in versioned configuration or narrowly named
policy functions. Execution concerns such as worker count, GPU assignment,
claim directories, and output paths must not alter cache identity or numerical
results.

## Refactor Sequence

1. Preserve the frozen implementation and establish release fixtures.
2. Introduce typed configuration while retaining environment compatibility.
3. Extract reaction validation and domain models.
4. Extract routing orchestration, species-energy service, and assembly.
5. Add local and distributed execution around the same content-addressed task.
6. Prove full TECRDB and ModelSEED parity.
7. Move paper-generation tooling and the manuscript only after parity.

Each extraction should be reviewable on its own and should leave the frozen
regression suite green.
