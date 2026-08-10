# Phase 2 — H/S decomposition & entropy validation vs Du ΔfS

Split the combined `G_RRHO` into H and S (xtb `--ohess` re-run, `split_HS.py`,
reproduces stored G_RRHO to 0.02 kJ/mol), then validated ab-initio entropy against Du's
ΔfS. Scripts: `assess_entropy.py`; residuals `assess_dSf_residuals.csv`; H/S per species
`mlip/HS_split_tecrdb_full.json`.

## Result (86 charge-matched species)
- **ab-initio ΔfS(gas) vs Du ΔfS(aq): Pearson r = 0.983.** The RRHO vibrational entropy is
  fundamentally correct. Mean signed offset +340 J/K/mol = the gas→aqueous solvation-
  entropy gap (expected; ab-initio S is gas-phase, Du is aqueous).
- **Per-species RRHO-entropy accuracy (element-referenced, LOO): median 33.7 J/K/mol**
  → entropic part of the ΔG error **T·MAE 12.6 kJ/mol (median 10.0)**.
- So of the ~17 kJ/mol per-species ΔfG error (Phase 1), **~10 kJ/mol is entropic**; the rest
  enthalpic (electronic + solvation → the phosphate wall, Channel 1).

## Physics: where the entropy error lives
Worst residuals are all **floppy** species — fructose-1,6-bisP (+89 kJ), PEP (+70),
oxaloacetate (−48), 1-octanol/octanal (−37/−29), sucrose (−26). The current pipeline
shares ONE Hessian (lowest conformer) and the conformer search (16 ETKDG seeds) collapses
floppy molecules to n_eff≈1 (ATP 1.00, CoA 1.01), so **conformational entropy is missing**
for exactly these species. Harmonic S_vib is fine (r=0.983); the gap is conformational,
not vibrational.

## Improvement launched
Deep conformer re-run on the **128 flexible under-sampled species** (rotatable bonds ≥4,
n_conf<10): `FAST_EMBED=300 FAST_NSTART=30 FAST_MIN_CONF=10` → new files
`ensemble_tecrdb_min10.json` / `geometries_tecrdb_min10/` (baseline left intact for a
controlled before/after). Next: MLIP E_elec on the new conformers (free GPUs 1–3), re-split
H/S, then re-score TECRDB reactions + re-run the Du per-species assessment to see if it helps.

Note: the repo previously found conformer depth shifts *reaction* MAE ~0.1 kJ/mol (they
largely cancel); this test targets the *per-species / absolute ΔfG* and *entropy* channels,
where they should not cancel.
