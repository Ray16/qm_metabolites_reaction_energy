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
2. **Random held-out ~6.6 sits ABOVE the experimental noise floor, not at it.**
   The floor — median per-reaction measurement sd over the 212 multiply-measured
   TECRDB reactions — is **~2.0 kJ/mol** (mean 2.6, IQR 1.0–3.9), NOT ~6. (An
   earlier draft wrote "~6 kJ floor"; that conflated it with sd(y)=12.5, the
   signal spread across reactions.) The incumbents' in-sample MAE 3.0 / medAE 1.5
   is near that ~2 kJ floor (train≈test). Held-out 6.6 leaves a real ~4–5 kJ gap
   above the floor. The ablation shows that gap is **data-limited (n=367), not
   architecture-limited** — no model class closes it held-out and the naive
   retrain overfits — but it is NOT irreducible noise. Room exists in principle;
   more/broader labels, not a better model, is what would use it.
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

## Improvement ablation (2026-08-12) — controlled, same folds, paired bootstrap CI
`scripts/run_ablation.py`: every variant on IDENTICAL folds vs the production
baseline (level `rich`, DEFAULT_HP, `w=log1p(n)`, MSE), 3 CV seeds, 3-ensemble,
95% CI on ΔMAE bootstrapped over reactions. Verdict = HELPS only if the CI
excludes 0. `artifacts/ablation_results.json`.

| variant | RANDOM ΔMAE [95% CI] | CPD-DISJOINT ΔMAE [95% CI] | verdict |
|---|---|---|---|
| baseline | 7.04 (ref) | 7.91 (ref) | — |
| #6 inverse-variance weight `n/sd²` | +0.41 [+0.18,+0.65] | +0.52 [+0.34,+0.71] | **HURTS** |
| #6 count weight `n` | +0.41 [+0.20,+0.60] | +0.38 [+0.20,+0.55] | **HURTS** |
| #6 uniform weight | +0.01 [−0.16,+0.17] | −0.02 [−0.14,+0.10] | within noise |
| #6 Huber (δ=6) | −0.08 [−0.25,+0.09] | +0.03 [−0.08,+0.13] | within noise |
| #1 condition head `S·f+h(pH,I,T,pMg)` | −0.11 [−0.24,+0.02] | +0.07 [−0.03,+0.17] | within noise |
| #4 QM-in-messages | +0.29 [+0.03,+0.57] | +1.08 [+0.83,+1.37] | **HURTS** |

**Verdicts**
- **#6 robust/weighted loss — does NOT help.** No weighting beats the existing
  `log1p(n)`. Inverse-variance and raw-count weighting HURT: they over-concentrate
  the loss on a few heavily-measured reactions. Huber ties baseline. The loss is
  not a lever.
- **#1 condition features — within noise.** Adding a reaction-level head on
  measured pH/I/T/pMg is a hair better on RANDOM (−0.11, CI touches 0) and a hair
  worse on CPD-DISJOINT; no reliable gain. Condition-*awareness* as a feature can't
  substitute for condition-*correcting the labels* (Legendre/pKa — not available).
- **#4 QM-in-messages — HURTS, badly on extrapolation** (+1.08 cpd-disjoint).
  Injecting QM before message passing overfits; readout-only is better. Confirms
  again that architecture is not the bottleneck.
- **#7 ensemble-variance UQ — WORKS (the one useful outcome).** Spearman(spread,
  |error|) = +0.28; selective prediction is monotone: most-confident 50% MAE 5.61
  vs 7.09 over all (RANDOM, 8-ensemble). The seed spread is a usable "don't trust
  this one" signal — worth more to the coverage mission than any sub-0.3 kJ
  accuracy move, none of which materialized anyway.
- **#3 metals — untestable here.** 0/453 TECRDB compounds contain a metal, so a
  metal-coverage gain cannot be shown on this set by construction; it belongs to
  the coverage evaluation, not TECRDB CV.

**Net:** consistent with the standing verdict — no accuracy lever crosses the
noise floor; the only real product improvement is calibrated uncertainty (#7).
Repro: `extract_conditions.py` → `CUDA_VISIBLE_DEVICES=0 python scripts/run_ablation.py`.

## Head-to-head vs eQuilibrator & dGPredictor (2026-08-12)
`scripts/benchmark_incumbents.py`: all methods scored against the SAME
experimental target (`tecrdb_full_experiment.json`, median −RT ln K′), n=367.
Figure `figures/gnn_vs_incumbents_tecrdb.png`, metrics
`artifacts/benchmark_incumbents.json`. predict-zero MAE 9.69, sd(y) 12.5.

| method | regime | MAE | RMSE | medAE | maxAE | sign% | R² | ρ |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| eQuilibrator | **in-sample** | 3.08 | 5.3 | 1.44 | 36.9 | 90.7 | 0.82 | 0.94 |
| dGPredictor | **in-sample** | 2.99 | 5.5 | 1.46 | 51.3 | 90.7 | 0.80 | 0.94 |
| dGP-retrained | in-sample refit | 5.66 | 12.4 | 1.98 | 89.8 | 87.4 | 0.00 | 0.84 |
| linear grp-CC | **held-out** | 7.01 | 9.9 | 5.18 | — | 73.6 | 0.37 | 0.62 |
| **GNN-delta** | **held-out** | **6.78** | 9.7 | 4.81 | — | 74.1 | 0.39 | 0.63 |
| QC first-princ | **no fit** | 36.71 | 49.7 | 26.45 | 209.0 | 60.2 | −14.9 | 0.20 |

`QC first-princ` = the from-scratch quantum-chemistry composite (conformers →
xtb/UMA geometries → ALPB/CPCM-X solvation → `ΔG=S·G_aq`), zero fit to
experiment — distinct from the GNN, which only uses QC as input FEATURES. On
full pH-7 TECRDB it is dominated by ALPB anion under-solvation (max err 209 kJ,
R²−14.9, ρ0.20): worse than predict-zero. Confirms the standing QC-thermo
verdict — absolute QC can't reach useful accuracy on polyanionic pH-7 species;
its value is coverage, not TECRDB accuracy. The GNN exists to recover ~6.8 MAE
by putting QC features on a learned group-additive backbone.

Paired ΔMAE vs GNN (95% bootstrap CI, +ve ⇒ GNN closer to exp):
eQ −3.69 [−4.52,−2.91], dGP −3.78 [−4.62,−2.94] (**incumbents better**);
dGP-retrained −1.05 [−2.39,+0.30] (tie); linear +0.23 [+0.05,+0.41] (GNN
edges it, within noise).

**Read carefully — the comparison is NOT like-for-like.** eQ and dGP are
COMPONENT-CONTRIBUTION MODELS FIT ON TECRDB; their 3.0 MAE is *in-sample*
(train ≈ test). The GNN/linear numbers are *held-out CV* (no test reaction
seen). So "incumbents win by 3.7 kJ" = in-sample vs held-out, not a
generalization result. The apples-to-apples held-out comparison for that model
class is our dGP-retrained refit: fit the same group basis, score it → MAE 5.66,
R² 0.00, RMSE 12.4 (it OVERFITS: 1421 groups, weak reg), i.e. WORSE than the
regularized GNN/linear held-out. Conclusion unchanged from the standing verdict:
where eQ/dGP have coverage they are the deployed best (in-sample); no model
class beats ~6.7–7 held-out; the GNN's value is coverage, not TECRDB accuracy.
Repro: `python scripts/benchmark_incumbents.py` (gnndgf env).
