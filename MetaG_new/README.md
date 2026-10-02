# MetaG

MetaG estimates standard transformed reaction Gibbs energies from molecular
structures using UMA electronic energies, conformer-resolved thermal
corrections, implicit solvation, structure-based reaction routing, and
calibrated uncertainty.

This directory is the maintained successor to `../MetaG`. Its first milestone
is strict numerical and routing parity with the frozen `2026-10-01c` release.
Scientific policy is intentionally unchanged during that work.

## Quick Start

Install the library and test dependencies:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The full QM path additionally requires the project UMA environment, an
available CUDA GPU, and xtb:

```bash
python -m pip install -e ".[qm,test]"
METAG_RUN_GPU_TESTS=1 python -m pytest -m gpu tests/integration
```

Run GPU tests only inside a reserved device allocation. CUDA visibility alone
does not opt in, because visible devices may be occupied by production jobs.

Use the Python API for reaction scoring:

```python
from metag.pipeline import score_reaction
```

The compatibility CLI remains available as either `metag` or
`python -m metag.pipeline`. See [docs/architecture.md](docs/architecture.md)
for the package map and [docs/parity.md](docs/parity.md) for the frozen
reference and migration gates. Current progress is recorded in
[docs/refactor-status.md](docs/refactor-status.md).

## Status

The runtime currently remains a behavior-preserving transplant. Its modules
are being separated behind stable interfaces before internal code is moved.
Do not treat the new directory as the paper-producing implementation until all
acceptance gates in `docs/parity.md` pass.
