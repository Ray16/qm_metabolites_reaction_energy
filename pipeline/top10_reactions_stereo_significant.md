# Column guide — `top10_reactions_stereo_significant.csv`

The 10 reactions with the largest **confident** disagreement between the
ModelSEED-retrained dGPredictor and TECRDB experiment. "Confident" = the
`stereo_exact` (same-molecule) tier **and** `abs_diff_kJ > combined_err_kJ`
(the gap exceeds the combined uncertainty). Deduplicated to distinct chemistries.
All energies are in **kJ/mol**, expressed in the reaction direction written in
`equation_definition`.

| # | column | description |
|--:|---|---|
| 1 | `rank` | 1–10, by descending `abs_diff_kJ`. |
| 2 | `modelseed_rxn` | Representative ModelSEED reaction ID for this chemistry. |
| 3 | `name` | ModelSEED reaction name. |
| 4 | `ec` | EC number(s) from the matched TECRDB entries (`;`-separated). |
| 5 | `match_tier` | Always `stereo_exact` here (full-InChIKey, same-isomer match). |
| 6 | `equation_definition` | Human-readable equation (compound **names**), ModelSEED-written direction. |
| 7 | `dGpredictor_modelseed_dG_kJ` | **Fine-tuned** (retrained) dGPredictor prediction. |
| 8 | `other_dGPredictor_original_dG_kJ` | The **original** KEGG-based dGPredictor value (context; blank if it had none). |
| 9 | `tecrdb_dG_kJ` | **Experimental** ΔG′° = −RT·ln(K′), median over the TECRDB measurements. |
| 10 | `diff_kJ` | `dGpredictor_modelseed − tecrdb` (signed; + = model over-predicts). |
| 11 | `abs_diff_kJ` | \|`diff_kJ`\| — the ranking key. |
| 12 | `dGpredictor_modelseed_err_kJ` | The model's **own** uncertainty on its prediction. |
| 13 | `tecrdb_dG_sd_kJ` | Standard deviation of the per-measurement experimental ΔG′° (0 if a single measurement). |
| 14 | `combined_err_kJ` | √(`err`² + `sd`²); the disparity is kept only when `abs_diff_kJ` exceeds this. |
| 15 | `n_measurements` | Number of TECRDB measurements behind `tecrdb_dG_kJ`. |
| 16 | `pH_min` / `pH_max` | pH range of those measurements (predictions are at pH 7). |
| 17 | `pH_max` | (see above) |
| 18 | `ms_orientation_vs_canonical` | Internal orientation flag (ModelSEED vs a lexical canonical ordering); **not** MS-vs-TECRDB direction — energies are already oriented, ignore for interpretation. |
| 19 | `tecrdb_reaction` | The matched TECRDB reaction string(s), KEGG-keyed. |
| 20 | `reaction_smiles` | Full balanced reaction as `reactants>>products` (RDKit reaction SMILES; species repeated by integer coefficient; ModelSEED direction) — for downstream QM. |
| 21 | `reaction_smiles_complete` | True if every species had a parseable SMILES. |
| 22 | `reactants_smiles` | Reactant SMILES, coefficient-annotated, `; `-joined. |
| 23 | `products_smiles` | Product SMILES, coefficient-annotated, `; `-joined. |
| 24 | `duplicate_rxn_ids` | All ModelSEED reaction IDs collapsed into this row (same full structure). |
| 25 | `n_duplicate_ids` | Count of `duplicate_rxn_ids`. |

Notes
- SMILES are the ModelSEED **served** (pH-7 charged) forms; per-compound neutral
  SMILES are in `top10_metabolites_stereo_significant.csv`.
- Reverse-direction copies of a reaction (e.g. glutathione reductase written both
  ways) appear as separate ranks with sign-flipped energies.
