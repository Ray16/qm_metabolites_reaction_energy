# Phase 1 — ab-initio pipeline vs Du 2018 curated ΔfG

Goal: assess current pipeline (MLIP E_elec + xtb-ALPB solvation + qRRHO) against Du's
per-species formation data, as an independent cross-check on the internal error budget.
Script: `assess_formation.py`. Residuals: `assess_dGf_residuals.csv`.

## Method
Ab-initio gives *absolute* aqueous G; Du gives ΔfG (element-referenced, per charge state,
J/mol). Charge-matched 125 species, then fit atom-equivalents
`G_aq(ai) = ΔfG(exp) + Σ_e n_e c_e + c_z·z + c0` (element counts from computed geometry).
Regression residual = per-species error with the systematic element reference removed.

## Result

| | MAE | RMSE | median | (kJ/mol) |
|---|--:|--:|--:|---|
| in-sample | 20.8 | 31.6 | 15.8 | |
| **LOO-CV** | **24.1** | 43.3 | **16.7** | |

By |charge| (LOO MAE): |z|0 **23.7** · |z|1 27.1 · |z|2 21.7 · |z|3 23.7 · |z|4 45.0 (n=3).

## Findings
1. **Independent corroboration of the internal error budget.** Median per-species error
   16.7 kJ/mol ≈ the repo's reaction-derived per-species oracle floor (15.9). Two
   independent routes (TECRDB reactions vs Du formation data) agree the per-species floor
   is ~16 kJ/mol. First external validation of that number.
2. **Per-species ΔfG error is ~flat with charge** (~24, ignoring the n=3 |z|=4 bin) —
   unlike the *reaction* error which scales steeply (11.6→50.8). Implication: the reaction
   charge-scaling is largely an **accumulation** effect (more/larger charged species per
   high-charge reaction, adding ~quadratically), not a bigger per-species error on each
   anion. Worth confirming — it changes where to spend effort.
3. **The atom-equivalent reference is itself a confound.** Mean (24) >> median (17) because
   of reference-model artifacts, not ab-initio error:
   - S poorly referenced (few S compounds) → APS −250, L-Met +250 are S-reference
     artifacts, not real errors.
   - unusual bonding not captured by atoms: cAMP (cyclic phosphate) −159.
   - genuine phosphate-wall signal survives: DHAP +86, GAP +79, Ru5P +60, FBP +60,
     PRPP +53, ATP −60 — sugar-/poly-phosphates, mixed sign.
   → **median is the trustworthy statistic; ~17 is an upper bound** (includes reference
   confound + RRHO + solvation + conformer + exp error).

## Consequence for how to use Du's data
Absolute-vs-formation with atom-equivalents is confounded by the reference model. The
**clean** uses of Du's curated data are:
- **Reaction differences** (Σν ΔfG): element refs cancel *exactly*, no atom-equivalent.
  Build reactions from the 233 ModelSEED-matched dG_f compounds → an expanded, independent
  ab-initio-vs-experiment reaction benchmark beyond TECRDB. (→ Phase 3.)
- **Entropy (ΔfS, 87 in-set / 242 in ModelSEED):** cleanest per-species channel, but needs
  the ab-initio **S** split out of G_RRHO (currently only combined G is stored) — a
  frequency/thermo re-run on stored geometries. (→ Phase 2 improvement.)
- **ΔfH (85 in-set):** with ab-initio H, enables the H vs S error decomposition of the 36.1.

## Next
- Phase 2: split H/S in the pipeline (re-run xtb thermo on stored geometries) → decompose
  error into enthalpic (electronic+solvation) vs entropic (RRHO), and validate ΔfS vs Du —
  especially the sugar/isomer cases Du's own regression fails.
- Phase 3: Du-ΔfG reaction-difference benchmark + TECRDB reaction ΔG demonstration.
