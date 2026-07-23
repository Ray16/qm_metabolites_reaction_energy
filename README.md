# QM Thermodynamics Pipeline (`qm_thermo`)

Compute aqueous standard Gibbs energies for ModelSEED metabolite **structures**
and transformed **reaction** Gibbs energies (Δ_rG′°) at the **quantum level**
(ORCA DFT, with GFN2-xTB conformer screening) — a more accurate, extrapolatable
alternative to the group/component-contribution methods ModelSEED currently uses
(ModelSEED GCM, eQuilibrator, dGPredictor), all of which fit experimental data
rather than solving electronic structure.

First milestone: the **83 central metabolites** in
`central_metabolites_in_opentecr.json`, benchmarked against experimental openTECR
values and the existing methods.

## Scientific protocol (per molecule)

1. **Microspecies** — the ModelSEED SMILES are already the dominant pH-7
   microspecies (explicit charges); we trust the protonation state.
2. **Conformer ensemble** — RDKit ETKDG → MMFF prune → **GFN2-xTB** optimise in
   ALPB water → keep the lowest conformers within an energy window.
3. **DFT (ORCA)** — `r2SCAN-3c` geometry optimisation + frequencies with
   **SMD(water)** implicit solvation → aqueous G per conformer.
4. **Ensemble G(aq)** — Boltzmann-average conformers, add the 1 atm→1 M
   standard-state correction.
5. **Reaction Δ_rG′°** — sum ν·G (elemental references cancel), then the Alberty
   Legendre transform to pH 7 + extended Debye–Hückel ionic-strength correction.

Implicit solvent is **mandatory**: most metabolites here are charged
(phosphates, carboxylates), and aqueous ion solvation dominates the energetics.

## Layout

| module | role |
|--------|------|
| `config.py`     | paths, conditions (pH/I/T), QM level, parallel layout |
| `structures.py` | load/validate the 83 pH-7 microspecies (RDKit) |
| `geometry.py`   | 3D geometry container + XYZ I/O |
| `conformers.py` | ETKDG ensemble + GFN2-xTB screening |
| `qm_backend.py` | ORCA DFT backend (opt+freq+SMD) behind a pluggable interface |
| `thermo.py`     | Boltzmann-averaged ensemble G(aq), standard-state correction |
| `compute.py`    | per-compound driver + parallel batch (JSON-cached, resumable) |
| `reactions.py`  | Δ_rG and the Alberty/Debye–Hückel transform to Δ_rG′° |
| `references.py` | load ModelSEED reactions + existing-method Δ_rG values |
| `benchmark.py`  | QM vs openTECR + GCM/eQuilibrator, stats + parity plot |
| `cli.py`        | command-line entry point |

## Setup

ORCA 6.1.1 and a matching **OpenMPI 4.1.8** (built from source) live under
`/nfs/lambda_stor_01/homes/rzhu/`. The ORCA build requires exactly 4.1.8 — the
system 4.1.2 and conda's 4.1.6 will not drive its MPI binaries.

```bash
source thermodynamic_calc/env.sh        # PATH/LD_LIBRARY_PATH for ORCA + OpenMPI
conda activate palm                     # provides rdkit/ase; xtb is called by path
```

Each ORCA job uses **16 cores** (`config.ParallelSettings.orca_nprocs`); the batch
runs up to 4 such jobs concurrently on the 80-core node. QM scratch goes to local
`/tmp` (NFS home is near-full); final results cache under `results/`.

## Usage

```bash
cd thermodynamic_calc
python -m qm_thermo.cli check --id cpd00001        # tool check + 1-molecule run
python -m qm_thermo.cli compute --ids cpd00020 cpd00029
python -m qm_thermo.cli compute --all              # all 83 (cached/resumable)
python -m qm_thermo.cli benchmark                  # openTECR comparison + plot
```

## Scaling beyond the benchmark

`config.py` exposes the knobs to trade accuracy for throughput at whole-DB scale:
fewer conformers (`ConformerSettings`), single-conformer mode (`max_qm_confs=1`),
or a cheaper DFT level (`QMLevel.opt_keywords`).
