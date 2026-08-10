# Experimentally-anchored phosphate/charged manifold — does the factorization work?

Tested the "pin phosphate free energies to experiment, let QC keep the covalent
chemistry" factorization on the full 367-reaction TECRDB set.
Script: `pipeline/score_anchored_manifold.py`. Env: `boltz-2` (run single-threaded
BLAS — the LOO loop oversubscribes cores otherwise).

## Construction (linear, honest CV)
Scoring is linear (`drG' = Σ ν·dfG'`), so "pin species c to data" = an additive
correction `δ_c` to its transformed formation energy:
`pred(r) = QC_baseline(r) + Σ_{c∈H} ν_rc·δ_c`, H = anchored set.
`δ_H` is fit (ridge) to experimental reaction dG and evaluated **strictly
out-of-sample** (leave-one-reaction-out). Using the training reactions as the
experimental anchor pool is the deployment scenario and the fair analogue to how
eQuilibrator is fit to TECRDB. Baseline reproduces 36.13 exactly.

## Reference points
| predictor | MAE | RMSE |
|---|--:|--:|
| QC baseline | 36.13 | 48.59 |
| predict-zero | 9.69 | — |
| eQuilibrator (data-driven) | 3.08 | 5.27 |

## Anchoring result (leave-one-reaction-out)
| anchored set H | \|H\| | LOO MAE (best λ) |
|---|--:|--:|
| recurring cofactors only (16) | 16 | 23.6 |
| all P-bearing / \|z\|≥2 (194) | 194 | **19.7** |
| all charged (299) | 299 | 19–23 |

**36.1 → ~20 out-of-sample.** Matches the prior per-reactant residual (LRO 23.7,
`FINDINGS_correction_layer.md`); restricting δ to the charged manifold + light
ridge is slightly better-conditioned (19.7).

## Where it works — stratified by max|z| (base → anchored)
| max \|z\| | n | base MAE | anchored MAE |
|---|--:|--:|--:|
| 0 (neutral) | 29 | 11.63 | 11.63 (untouched) |
| 1 | 40 | 16.63 | 17.11 |
| 2 | 160 | 32.82 | **18.27** |
| ≥3 | 138 | 50.78 | **23.80** |

Anchoring cuts the multiply-charged phosphate error roughly in half — **exactly the
channel the physics diagnosis predicted**. The neutral floor (11.6) is untouched
(nothing to anchor) and is the non-solvation residual (conformer/RRHO/tautomer).

## The boundary (why it plateaus at ~20, and what it means for the niche)
- Cofactor-only anchoring (23.6) captures most of the gain; adding substrate
  phosphates buys only ~3.5 more out-of-sample (20.3), and that part is mostly
  in-sample overfit (in-sample cofactor 24.0 → all-charged 20.0, a 4-point gain
  that shrinks to ~1–3 out-of-sample).
- **The cofactor pool is what transfers** — pinning ATP/ADP/AMP/Pi/PPi/NAD(P)(H)/
  CoA/GSH to a few external gold-standard reactions is robust, parameter-light,
  and generalizes to new reactions.
- **The substrate phosphates do not** — surviving top residuals are sugar-P
  rearrangements (transaldolase +105, aldolase +105), acyl-CoA thioester redox
  (−130), each a rare, scaffold-specific center with no shared anchor. This is
  the leave-COMPOUND-out regime (FINDINGS_correction_layer LCO ~29–33) — i.e. the
  novel-metabolite niche that justifies the physics pipeline. Anchoring cannot
  reach it because a never-seen substrate has no experimental anchor.

## Would a FIXED EXTERNAL anchor table (zero TECRDB fitting) work? — quantified
Simulated it: estimate per-species offsets on one random half, apply blind to the
disjoint half (the honest analogue of literature anchors → new reactions).

- **Coverage ceiling:** only 199/367 reactions touch a recurring cofactor. A fixed
  cofactor table structurally CANNOT touch the other 168 (base MAE 21.9).
- **Covered-reaction result:** base 48.2 → **28.5 ± 2.4** blind. Removes ~20 kJ/mol
  of the covered error but plateaus at ~28 (and this OVER-estimates a real 5-anchor
  table, which estimates each primitive from 1 reaction, not ~180). Aggregate ≈ 26.
- **Which anchors transfer (offset std/|mean| over 600 half-fits):** STABLE →
  Phosphate −44.5 (0.13), ATP +55.2 (0.24), NADP +31.8 (0.26), Acetyl-CoA +52.4
  (0.26), PPi −27.6 (0.36), NADPH −17.4 (0.46). UNSTABLE (anchoring adds noise) →
  AMP (9.6), NADH (2.2), GSSG (1.7), CoA (0.76), NAD (0.67), ADP (0.65).
  Design implication: a defensible table includes only the stable phosphate/redox
  anchors (Pi, PPi, ATP, NADP/NADPH couple, acetyl-CoA); the near-zero/noisy ones
  hurt. Split-half δ-vector correlation median r = 0.73 (moderate).

Confidence: a fixed external table reliably fires on the big systematic
phosphoanhydride/redox biases (those offsets are transferable), lands ~26 aggregate
(36→26, clean/no-leakage), but is capped by 54% coverage + ~±13 phosphoanhydride
scatter and cannot approach predict-zero (9.7) or eQ (3.1).

## Verdict
The factorization is **real and worth deploying in its robust form** (pin the
recurring charged cofactor pool to external experimental anchors: 36 → ~24, the
part that generalizes). But it **does not dissolve the phosphate wall**: residual
|z|≥3 error stays ~24, still 2× predict-zero (9.7) and 6× eQuilibrator (3.1),
because the surviving error is scaffold-specific substrate solvation + non-additive
per-reaction noise, not a transferable constant. For TECRDB accuracy it cannot
compete with data methods; for the novel-substrate niche it does not help, because
that is exactly the unanchorable part.
