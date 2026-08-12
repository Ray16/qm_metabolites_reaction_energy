# GNN for per-compound formation energy (dGPredictor-done-right) — findings

**Question:** can a message-passing GNN that predicts per-compound formation
energy (dG = S·f, trained on reaction dG only) + QM/physical features beat
linear group contribution and extend coverage?

## Setup
- Data: 367 TECRDB reactions in ModelSEED ids, 453 compounds with SMILES.
  Target `dG_kJ = -RT ln K'` (median over measurements; pH-mixed → noisy).
- Model: scatter-MPNN over atoms, **sum readout** (extensive), reaction head
  `pred = S @ f`. QM injected as node feature (xtb GFN2 Mulliken charge) +
  graph features (dGsolv, HOMO/LUMO/gap, RDKit descriptors, CPCM-X solvation).
- Delta-learning: `pred = linear_group_prior + S @ f_residual`.
- Eval: held-out CV, two schemes — RANDOM (interpolation) and COMPOUND-DISJOINT
  (extrapolation to unseen compounds; the coverage regime). Early stopping on
  inner val, dropout+wd+LayerNorm, seed-ensembled. No hyperparams tuned on test.

## Held-out results (kJ/mol MAE)
| model | RANDOM | CPD-DISJOINT |
|---|--:|--:|
| predict-zero | 9.7 | 10.4 |
| linear group-CC | 6.6–7.0 | 8.7 |
| GNN (graph only) | 7.0 | 8.9 |
| GNN + xtb (full) | 6.7–7.5 | 8.7–8.9 |
| GNN-delta (full) | 6.69 | 8.66 |
| GNN + rich (+CPCM-X) | 6.80 | **8.59** |
| GNN-delta (rich) | 6.75 | 8.61 |
| eQuilibrator / dGP **in-sample** | 3.0 | 3.0 |
| QM absolute (prior work) | — | 35.8 |

## Conclusions
1. **Model class is not the bottleneck — the data is.** Linear CC, GNN,
   GNN+QM, and GNN-delta all land within ~0.3 kJ/mol of each other. Adding a
   nonlinear learned representation on top of group-additivity captures
   essentially no residual signal at n=367.
2. **Random held-out ~6.6 is near the experimental noise floor** (TECRDB sd
   ~6 kJ/mol). The incumbents' in-sample 3.0 is fitting below that floor
   (see lam sweep: in-sample reaches 1.6 while held-out worsens — visible
   overfitting). So on random CV there is little room for ANY model.
3. **QM/physical features help only where it matters — compound-disjoint
   extrapolation — and only a little.** CPCM-X ("rich") is the single best
   config on CPD-DISJOINT (8.59), physically sensible (better solvation aids
   novel-compound transfer), but the margin over linear (8.69) is within noise.
4. **The GNN's real value is coverage, not TECRDB accuracy.** On the 367
   reactions the incumbents already cover, they win in-sample; the GNN cannot
   beat a regularized linear model held-out. The only defensible advantage —
   graceful extrapolation to compounds eQ/dGP can't decompose — cannot be
   measured on this labeled set (those reactions have no experimental dG).

## Data-scale lever — TESTED, does not help
Trained on 8,765 confident eQ-labeled ModelSEED reactions (unc<15 kJ), 3,415
compounds, 84% test-compound coverage:
- **Naive distillation** (train eQ, test experimental): MAE ~18 — WORSE than
  367-only. Distribution mismatch: eQ ModelSEED reactions span sd 150 (big
  oxidations) vs TECRDB enzyme reactions sd 12.5; model fits coarse absolutes,
  not the fine differences the test needs.
- **Transfer learning** (pretrain eQ -> fine-tune 367, held-out CV):
  from-scratch 7.14 vs pretrained **7.23** — no benefit. Pretraining representations
  optimized for large-molecule absolutes don't transfer to fine near-thermoneutral
  differences (pretrain eQ-val only 19.6).

## Final verdict
No architecture or data-augmentation lever beats ~6.7-7 held-out MAE on TECRDB.
Random-CV ceiling is the experimental noise floor (~6 kJ). eQ/dGP in-sample 3.0
is the practical best where they have coverage. The new method's only unrealized
value is COVERAGE on the ~42% they can't score -- which cannot be validated on
TECRDB by construction. That requires a different evaluation (unlabeled ModelSEED
+ external checks), not a different model.

Repro: `prepare_data.py` (base env) → `train_v2.py` (uma env, GPU).
Features: `extract_xtb.py` → `qm_features.json`.
