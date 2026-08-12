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

## Environments
Featurization needs **rdkit** (base env); GPU training needs **CUDA torch**
(`uma` env — the base torch cu130 can't see the CUDA-12.4 driver). The package
is split so training modules never import rdkit.
- base env:  `python scripts/prepare_data.py`, `run_baselines.py`, `plot_*.py`
- xtb env:   `conda run -n xtb python scripts/extract_xtb.py`
- uma env:   `CUDA_VISIBLE_DEVICES=1 <uma>/python scripts/run_cv.py`

## Pipeline
```
scripts/extract_xtb.py     geometries -> artifacts/qm_features.json      (xtb env)
scripts/prepare_data.py    -> artifacts/data.pt                          (base)
scripts/run_cv.py          held-out CV: linear vs GNN vs GNN-delta       (uma/GPU)
scripts/run_baselines.py   fair dGP-linear + QC charge-gated test        (base)
scripts/save_model.py      --mode scratch|delta -> artifacts/checkpoint.pt (uma)
scripts/predict.py         held-out OOF preds -> artifacts/predictions.json (uma)
scripts/plot_comparison.py GNN vs eQ vs dGP scatter -> figures/          (base)
scripts/plot_reactions.py  10-reaction disagreement figure -> figures/   (base)
scripts/prepare_distill.py -> artifacts/distill_data.pt (eQ 8.7k + 367)  (base)
scripts/run_datascale.py   --mode distill|finetune (both negative)       (uma/GPU)
```

## The saved model (`artifacts/checkpoint.pt`)
From-scratch GNN[rich] (no dGPredictor anchor — coverage-appropriate), 4-seed
ensemble. Load with `torch.load(..., map_location='cpu')`. Predict: `ΔG = S·f`,
`f` = ensemble mean of `MPNN(graph)`. Reproduce: `scripts/save_model.py`.
