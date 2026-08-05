# Full TECRDB run — results against the pre-registered predictions

367 reactions (80% of all 457 distinct TECRDB reactions with a usable K,
carrying 87% of usable measurements), 453 compounds, 3,263 conformers.
MACE-POLAR-1 + xtb-GFN2/ALPB composite, pH-7 fixed microspecies.

Predictions were fixed in `score_tecrdb_full.py` before the numbers existed
(commit 6f7f94e).

## P1 — CONFIRMED (and uninformative, as expected)

| | MAE | signs |
|---|---:|---:|
| QM composite | **36.1** | 231/367 (63%) |
| predict zero | **9.7** | — |

## P2 — CONFIRMED, precisely

Δ(P–O–P) = −1, now n = 35 (was 13): mean signed error **+63.8 kJ/mol** against
a pre-registered +64 ± 15. The per-phosphoanhydride bias replicates at 2.7× the
original sample. This is the most solid quantitative result in the project.

## P3 — FAILED. The residual structure is multi-group, not one term

Ridge (α = 1, condition number 14, so not collinear), coefficients per +1 unit
of the group, fold-stable unless marked:

| group | kJ/unit | | group | kJ/unit |
|---|---:|---|---|---:|
| phosphoanhydride | **−95.0** | | ketone | +23.1 |
| thioester | −33.1 | | thiol | +20.1 |
| aldehyde | +30.9 | | aromatic N | −15.1 |
| carboxylate | +23.4 | | carboxylic ester | +14.8 |

(unstable across folds, discard: amide, acetal/anomeric, C=C)

Sign convention: coefficients are per **+1** unit. Cleaving one P–O–P bond is
Δ = −1, giving −1 × −95.0 = **+95 kJ/mol** of error — consistent with P2.

## P4 — FAILED decisively. The reference-network strategy is dead

Best-reference fingerprint cosine across the 346 scorable reactions:

| threshold | coverage |
|---|---:|
| ≥ 0.95 | 27% |
| **≥ 0.90** | **30%**  (pre-registered: >50%) |
| ≥ 0.80 | 48% |

30% against 29% on the 130-reaction set — going 130 → 367 bought **one
percentage point**. The reason is structural: TECRDB holds only 457 distinct
reactions and they are chemically diverse, so adding reactions does not
densify the network. There is no larger set to try; 367 is 80% of everything
that exists. Per the pre-registration, this strategy is abandoned, not tuned.

## Does the group correction transfer? Yes — and it is still not useful

| cross-validation | MAE before → after |
|---|---|
| random 5-fold (optimistic) | 36.1 → 24.8 |
| leave-one-EC-subclass-out | 36.1 → 27.8 |
| compound-grouped 5-fold | 36.1 → 28.5 |
| leave-one-EC-class-out (strictest chemistry holdout) | 36.1 → **29.1** |
| *predict zero* | *9.7* |

Unlike the per-species additive scheme (in-sample R² 0.769, grouped 0.177), this
one **survives every grouped holdout** — it is a real, transferable description
of where the composite errs. But it lands at 3× predict-zero, so it is a
diagnostic, not a predictor.

## What this settles

Both correction strategies now have measured ceilings far above the trivial
baseline: reference-network coverage is capped at 30% by the diversity of the
experimental record, and group-residual correction plateaus near 28–29 kJ/mol
under honest validation. Together with the earlier negatives (electronic method,
MLIP choice, solvation model, geometry method, conformer sampling, Mg,
speciation), the absolute-composite route is closed for this domain.

The phosphoanhydride term is worth keeping as physics: a replicated +95 kJ/mol
per cleaved P–O–P bond is a specific, quantified statement about where a
continuum composite fails, and it points at the solvation of charge separation
rather than at anything electronic.
