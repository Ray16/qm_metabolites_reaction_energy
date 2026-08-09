# Can an empirical linear regression correct QC ΔrG′° into usefulness?

Tested on the full 367-reaction TECRDB↔ModelSEED set (`results/benchmark/tecrdb_full_scored.json`).
Motivation: QC does not capture all reactions (raw MAE 36.1 kJ/mol), which is
exactly why component-contribution / eQuilibrator fit an empirical residual layer.
Question: does that rescue *our* QC composite? Answer: **no — it is dominated by
fitting ΔG empirically without QC at all.**

## The reference points
- raw QC composite error: **MAE 36.1**, systematically biased (+16…+32 signed on phosphates)
- predict-zero baseline: **9.7**
- eQuilibrator (data-driven): **~5.5** (README)

## Where QC error lives (stratification)
| class | n | QC MAE | predict-zero | QC signed |
|---|--:|--:|--:|--:|
| phosphoanhydride (P-O-P) | 192 | 45.0 | 11.1 | +32.4 |
| phosphate (other) | 65 | 40.4 | 10.0 | +16.4 |
| no P / no CoA / no NAD | 109 | 17.6 | 6.8 | +4.9 |

The error is concentrated in phosphate/polyanion chemistry and is a *signed bias*,
not scatter — i.e. exactly the continuum ion-solvation error diagnosed in
`FINDINGS.md`. Even the phosphate-free subset (17.6) loses to predict-zero (6.8).

## Corrections, under honest cross-validation
LRO = leave-reaction-out (new reactions, known compounds).
LCO = leave-compound-out (reactions with a novel compound — QC's intended niche).

| correction basis | in-sample | LRO-CV | LCO-CV |
|---|--:|--:|--:|
| group-change (13 SMARTS groups) | 23.5 | 24.7 | 28.7 |
| per-reactant (component-contribution residual) | 18.7 | 23.7 | 33.0 |
| reactant + group | 16.6 | **22.2** | 29.2 |
| charge / Born Δ(Σz²), 1 param | 31.9 | 32.0 | **32.4** |

- The best correction reaches **~22 for known compounds** but only **~29 for novel
  compounds** — and never approaches predict-zero (9.7).
- Charge/Born features are the *only* ones that transfer perfectly to novel
  compounds (in-sample ≈ LRO ≈ LCO), because they need only formal charge — but
  they remove just ~4 kJ/mol. The transferable part of the error is small; the
  large part is high-dimensional per-species solvation scatter (consistent with
  the compound-grouped-CV R²=0.177 in `FINDINGS.md`).

## The decisive test (leave-reaction-out, vs experiment)
| predictor | MAE |
|---|--:|
| predict-zero | 9.7 |
| QC alone | 36.1 |
| QC + empirical reactant correction | 23.8 |
| **empirical reactant model ALONE (no QC), same basis** | **7.8** |
| empirical model + QC as an extra feature | 7.6 |

**Fitting the empirical model to predict ΔG directly gives 7.8; correcting QC gives
23.8.** Handing QC to the empirical model as a feature moves 7.8 → 7.6 — QC adds
~0.2 kJ/mol. QC as a *base* is counterproductive: it injects ~36 kJ/mol of
per-species solvation noise that the linear layer only partially removes, and it
carries almost no information orthogonal to the empirical basis.

## Conclusion for the pipeline
1. For accuracy, do **not** use QC + empirical correction. Regress ΔG directly
   (component-contribution / eQuilibrator): 5.5–7.8 kJ/mol.
2. QC's only defensible role stays what `FINDINGS.md` says: independent
   corroboration (redox/thiol), and the novel-compound regime with *no* empirical
   data — but LCO shows a correction cannot make QC accurate there either
   (~29 kJ/mol). Use it for sign/magnitude sanity, not as a ΔG source.
3. The only route that raises QC's ceiling is fixing the physics, not regressing
   it away: explicit-solvent MLIP + alchemical FEP for polyanion solvation
   (`results/bulk_fep/`).

Reproduce: `results/benchmark/tecrdb_full_QCerror_ranked.csv` (367 rows ranked by
|QC error| with chemistry flags) and the scripts under `pipeline/`.
