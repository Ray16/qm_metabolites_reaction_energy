# Route 1 test: does QC have a redox niche (potential-space / cofactor-referencing)?

Tested 2026-08-12 on the 93 NAD(P)-linked reactions in the 367-set, using the
already-computed `G_aq_tecrdb_full.json`. No new compute.

## Idea
A biochemical redox reaction is `S_red + NAD+ -> S_ox + NADH`. The cofactor
half `G(NADH)-G(NAD+)` is a constant across all NAD reactions and is a
multiply-charged, badly-solvated species. Hypothesis: reference it out (to an
experimental E'0, or by differencing two same-cofactor reactions) and the
residual substrate half-reaction is small/neutral -> QC accurate. This is the
literature redox recipe (Jinich 2018: 2-4 kJ/mol).

## Result: the hypothesis fails on biological substrates

Cofactor-cancelling pairwise differences (cofactor cancels exactly):

| set | QC MAE |
|---|--:|
| raw per-reaction | 40.5 |
| cofactor cancelled (NAD) | 33.6 |
| cofactor cancelled (NADP) | 34.4 |

Cancelling the cofactor barely helps -> **the error is in the substrate couple,
not the cofactor** (confirms the earlier EXPERIMENTS_LOG cofactor-cancellation
note). Stratifying the substrate-exchange error by substrate charge:

| substrate character | pairs | QC MAE | median |
|---|--:|--:|--:|
| neutral substrates | 320 | **13.0** | 9.1 |
| max \|z\|=1 | 281 | 20.9 | 16.3 |
| max \|z\|>=2 | 148 | **62.6** | 48.3 |
| phosphate in substrate | 34 | 66.9 | 65.8 |

## Conclusion
Redox is **not a special QC niche** in this dataset. The redox error is the same
phosphate/polyanion solvation wall, one level down in the substrate. QC does OK
on neutral-substrate redox (median 9) -- but that is just the general neutral
floor (11.6), no redox-specific advantage. Of the 93 NAD(P) reactions only 38
have neutral substrates.

The literature redox accuracy (Jinich) was on neutral organic carbonyl/alcohol
couples; the charged biological substrates (malate, isocitrate, phospho-sugars)
carry the same wall. Route 1 is closed as an escape from the wall.

## What this leaves
QC's usable envelope is reactions with **no species above |z|=1** (MAE ~11-15).
Everything multiply-charged is the wall. The two routes that remain live both
AVOID computing charged-species energies:
- **Route 2 (speciation)**: QC for relative microspecies energies (pKa,
  tautomer, anomer, stereo) feeding a data-driven dG. A single deprotonation
  cancels most of the solvation error; QC never scores an absolute polyanion.
- **Route 4 (solvation physics)**: cluster-continuum or ML ion-solvation on the
  ~15 recurring anions, cached. The only route that attacks the wall directly.
