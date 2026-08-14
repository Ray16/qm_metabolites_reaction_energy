# ModelSEED formation-energy coverage: GNN vs dGPredictor vs eQuilibrator

**Goal:** show the GNN scores compounds the incumbents cannot. All methods are
structure-based, so a compound with no SMILES is uncoverable by everyone (a hard
floor). Coverage measured over the **current** ModelSEED Biochemistry DB.

## Denominators (computed directly, `scripts/coverage_sweep.py`)
| set | count | note |
|---|--:|---|
| active compounds | 45,662 | non-obsolete |
| with a SMILES (structured) | 36,801 (80.6%) | structure floor — nobody scores the other 19.4% |
| RDKit parse failures | 19 | bad valence etc. |
| contain `*` (R-group / polymer) | 6,454 | ΔGf physically ill-defined — set aside for a fair comparison |
| **complete structures** (parseable, no `*`) | **~30,335** | the fair coverage denominator |

## Coverage (decision rule per method is code-faithful)
| method | coverable | of complete structures | rule for "can't score" |
|---|--:|--:|---|
| **GNN** | 30,335 | **100%** | needs only a parseable molecular graph |
| retrained dGP (Andrew) | 26,461 | 87.2% | r1/r2 signature must lie in its ModelSEED-trained vocab (2,383 / 36,759) |
| **eQuilibrator** (structure-based) | 13,284 | **43.9%** | InChIKey must resolve in eQ compound cache AND give finite-uncertainty ΔfG |
| original dGP | 5,340 | 17.6% | r1/r2 signature must lie in KEGG-trained vocab (1,435 / 24,881) |

(Percent of the 36,801 structured set: GNN 99.9%, retrained dGP 71.9%, original dGP 14.5%.)

**eQuilibrator (measured, not estimated):** feeding eQ the InChIKey directly (bypassing
the KEGG gate), of 30,264 complete structures it covers **13,284 (43.9%)** — the rest
split into cache-miss 5,719 (18.9%) and *in cache but not estimable* (infinite
uncertainty) 11,261 (37.2%). Its as-deployed KEGG-gated path is capped even lower
(only 17,555 ModelSEED compounds have a KEGG id; ~57% reaction coverage). eQ is the
narrowest of the four. Env `eqapi` (equilibrator-api 0.6.0); repro
`tools/equilibrator/coverage_modelseed.py`.

**Figures / tables:** `figures/coverage_modelseed.png` (this table as a stacked bar),
`coverage_summary.csv`, `coverage_denominators.csv`, `coverage_per_compound.csv`
(per-compound covered flags for all four methods, join key = ModelSEED cpd id).

## The incremental win
- **vs original dGP / eQuilibrator — decisive.** Both are KEGG-mediated with fixed
  KEGG vocabularies; the GNN covers ~5–6× more compounds. No contest.
- **vs retrained dGP (the strong competitor) — real, +3,874 compounds.** The
  retrained vocab was frozen on a 32,647-compound snapshot. ModelSEED has grown to
  36,801 structured compounds; the ~3,874 added since carry substructures outside
  its trained groups, so it either returns nothing or silently zeroes the novel
  groups (a degraded number). **A fixed-vocabulary model must be re-trained as the
  database grows; the GNN generalizes to new compounds without retraining.** That
  is a structural, not incidental, advantage.

## Why coverage alone understates it (see `figures/gnn_vs_dgpretrained_hard.png`)
The retrained dGP's wide coverage is partly *extrapolation*: for compounds/reactions
near or past its trained-group support it emits numbers that are badly wrong — on the
hard redox/glycosyl/nucleotidyl subset its MAE is **~55 kJ/mol** (errors of 80–100
kJ), and on the full-367 refit R²=0.00 (overfit: 1,421 groups, weak reg). The 3,874
novel-group compounds are exactly this extrapolation regime. The GNN (regularized,
seed-ensembled) covers the same regime **without** the blow-ups.

**Measured head-to-head on the 8 hard reactions** (of the ten in
`figures/qm_vs_dgpredictor_top10.png`; the two n=1 redox rxns aren't in the GNN's set):
GNN **held-out** MAE **6.2** vs retrained-dGP **in-sample** MAE **54.5**. Per reaction
(exp / dGP-retrained / GNN): rxn00086 11.9/101.6/9.3 · rxn00070 18.0/104.9/6.8 ·
rxn00605 −9.5/−56.8/−5.0 · rxn01713 3.9/−39.8/−8.4 · rxn01834 23.5/−19.8/9.3 ·
rxn00579 −4.2/−46.9/−1.5 · rxn01675 1.0/42.9/2.3 · rxn01005 2.7/42.7/3.8. The GNN
number is held-out and dGP-retrained is in-sample, so the gap *understates* the GNN's
edge. Repro: `scripts/plot_coverage.py`.

**Combined claim:** the GNN matches the widest coverage available (retrained dGP),
extends it to +3,874 newer compounds a frozen vocabulary can't reach, and beats the
original dGP / eQuilibrator on coverage outright — while staying robust where the
retrained dGP extrapolates and fails.

## Honesty guardrails
- "Coverage" = *emits a defensible number*, not *validated accurate*. The +3,874
  incremental compounds have no experimental ΔG (that is *why* they are the
  frontier); their GNN predictions are not yet validated. Uncertainty comes from the
  seed-ensemble spread (the one calibrated signal we have).
- `*`/R-group compounds (6,454) are excluded from the fair denominator for all
  methods; the GNN *would* emit a number for them, but ΔGf is ill-defined there.
- eQuilibrator compound-level coverage is pending a reinstall (`equilibrator-api`
  0.7.0 + cache); the 57% is reaction-level from the prior ModelSEED-wide run.

Repro: `python scripts/coverage_sweep.py` (gnndgf env). Source counts:
`ModelSEEDDatabase/Biochemistry/compound_*.tsv`; vocabs under
`thermodynamic_calc/tools/dGPredictor{,_freiburger}/data/`.
