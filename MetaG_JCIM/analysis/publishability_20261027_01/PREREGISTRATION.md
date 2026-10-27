# Preregistered reference-admission policy (2026-10-27, written before any MetaG residual on it was computed)

Problem: the 10-10 rebuild excludes every measurement whose free Mg2+ activity is not reported
(1,053 "unspecified" + 569 "Mg present" + 413 "pMg = total salt"), leaving 39/364 labels.
Excluding a measurement is only warranted when the unknown quantity could change the label.

Policy "Mg-bounded" (all else identical to the 10-10 rebuild: same sources, mapping guard,
curation exclusions, I imputation 0.25 M, per-measurement temperature, median per reaction):
1. Bound the unknown free Mg2+ activity physically (never invent a value):
   - pMg missing, no Mg in recipe:           pMg_free in [4, 14]  (<=100 uM contaminant)
   - pMg missing, Mg in recipe, c_tot parsed: pMg_free in [-log10 c_tot, 14]
   - pMg missing, Mg in recipe, no c_tot:     pMg_free in [2, 14]  (<=10 mM total)
   - pMg equals -log10(total salt):           pMg_free in [reported, 14] (free <= total)
   - pMg reported as free activity:           point value (unchanged)
2. Evaluate the eQuilibrator dissociation transform on an 11-point grid over that interval.
   Admit the measurement iff max-min <= 2.0 kJ/mol (Mg ambiguity immaterial); label = mid-range.
3. Mapping: a multi-hit mapping is admitted iff every hit has identical full stoichiometry
   (duplicate ModelSEED IDs of the same chemistry); otherwise excluded as before.
Sensitivity (reported, not selected on): thresholds 1.0 and 4.0 kJ/mol.
Primary metric and all baselines are scored on the identical admitted intersection.
