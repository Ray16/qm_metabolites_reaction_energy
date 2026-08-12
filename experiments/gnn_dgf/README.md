# GNN component-contribution model for reaction ΔG

A message-passing GNN that maps each compound's graph to a scalar formation
energy `f`; a reaction's ΔG is the stoichiometric difference `ΔG = S·f`. Trained
on reaction ΔG only, with quantum-chemistry injected as features (xtb Mulliken
charges, HOMO/LUMO, ALPB + CPCM-X solvation, RDKit descriptors). "dGPredictor
done right" — a learned representation replacing the linear group table.

**Headline (held-out CV, 367 TECRDB reactions):** 6.8 random / 8.6 compound-disjoint
MAE — statistically tied with a regularized linear group-contribution model and
with the retrained dGPredictor judged fairly. The model class is **not** the
bottleneck; the data (n=367) is. See `FINDINGS.md` for the full verdict.

## Layout
```
gnn/                 importable package
  paths.py           centralized paths (no hard-coded relatives)
  features.py        compound featurization + CompoundGraphs   (needs rdkit)
  model.py           MPNN + Graph device wrapper               (pure torch)
  training.py        train/CV/ridge/delta utilities            (pure torch)
  data.py            TECRDB json loaders
scripts/             thin entry points (see below)
artifacts/           data.pt, distill_data.pt, checkpoint.pt, qm_features.json, *.json
logs/                run logs
FINDINGS.md          the writeup
```

## Setup — one conda env (`gnndgf`)

Everything (rdkit featurization **and** CUDA GPU training) runs in a single env.
Create it either from the file or by hand:

```bash
# from the environment file (recommended)
mamba env create -f environment.yml          # or: conda env create -f environment.yml
conda activate gnndgf

# --- or manually ---
mamba create -n gnndgf python=3.11 -y
conda activate gnndgf
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install rdkit numpy matplotlib
```

Verify (should print `cuda avail True` and an rdkit version):
```bash
python -c "import torch, rdkit; print('cuda', torch.cuda.is_available(), '| rdkit', rdkit.__version__)"
```

**Dependencies:** python 3.11 · torch 2.6.0 **+cu124** · rdkit 2026.3.5 · numpy · matplotlib.
The cu124 build matches this cluster's CUDA-12.4 driver on the V100s. (The old
base-env torch was cu130 and could not see that driver — hence a single pinned
env; do not use the base env.) `xtb` (for `extract_xtb.py`) is a separate binary,
available in the `xtb` conda env.

## Pipeline — run everything with the `gnndgf` env

```bash
conda activate gnndgf
CUDA_VISIBLE_DEVICES=1 python scripts/run_cv.py     # (set a free GPU)

scripts/extract_xtb.py     geometries -> artifacts/qm_features.json   (needs xtb; conda run -n xtb)
scripts/prepare_data.py    -> artifacts/data.pt
scripts/run_cv.py          held-out CV: linear vs GNN vs GNN-delta    (GPU)
scripts/run_baselines.py   fair dGP-linear + QC charge-gated test
scripts/save_model.py      --mode scratch|delta -> artifacts/checkpoint.pt (GPU)
scripts/predict.py         held-out OOF preds -> artifacts/predictions.json (GPU)
scripts/plot_comparison.py GNN vs eQ vs dGP scatter -> figures/
scripts/plot_reactions.py  10-reaction disagreement figure -> figures/
scripts/prepare_distill.py -> artifacts/distill_data.pt (eQ 8.7k + 367)
scripts/run_datascale.py   --mode distill|finetune (both negative)    (GPU)
scripts/diagnose_training.py  prints device + convergence curve       (GPU)
```
Pick a free GPU with `CUDA_VISIBLE_DEVICES` (check `nvidia-smi`).

## The saved model (`artifacts/checkpoint.pt`)
From-scratch GNN[rich] (no dGPredictor anchor — coverage-appropriate), 4-seed
ensemble. Load with `torch.load(..., map_location='cpu')`. Predict: `ΔG = S·f`,
`f` = ensemble mean of `MPNN(graph)`. Reproduce: `scripts/save_model.py`.
