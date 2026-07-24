# Column guide — `top10_metabolites_stereo_significant.csv`

Every metabolite that appears in the 10 reactions of
`top10_reactions_stereo_significant.csv` (the confident, same-isomer,
significance-filtered top-10). One row per metabolite. SMILES are included for
downstream quantum-physics / DFT estimation.

| # | column | description |
|--:|---|---|
| 1 | `metabolite_id` | ModelSEED compound ID (`cpd…`). |
| 2 | `common_name` | Primary compound name. |
| 3 | `formula` | Molecular formula (as stored, i.e. the pH-7 charged form). |
| 4 | `smiles` | RDKit-canonical SMILES of the ModelSEED **served** structure (pH-7 charge state the model used). |
| 5 | `smiles_neutral` | Charge-**neutralised** parent SMILES (RDKit Uncharger) — usually preferred for gas-phase QM. Permanent charges that cannot be neutralised (e.g. NAD(P)⁺ ring N⁺) remain counter-balanced. |
| 6 | `kegg_ids` | KEGG compound alias(es), `;`-separated (blank if none). |
| 7 | `inchikey` | Standard InChIKey (the structure hash used for matching). |
| 8 | `role` | `reactant`, `product`, or both, across the top-10 reactions. |
| 9 | `reactions_in_top10` | Which top-10 reactions this metabolite is in, each tagged with its rank, e.g. `rxn00086(#1);rxn00070(#3)`. |
| 10 | `n_reactions_in_top10` | Count of `reactions_in_top10`. |
| 11 | `n_reactions_in_full_db` | How many reactions in the **whole ModelSEED database** this metabolite participates in (its connectivity/degree). The complete reaction list is in `top10_metabolites_stereo_significant_full_reaction_membership.tsv`. |
| 12 | `is_cofactor` | ModelSEED cofactor flag (0/1). |

Notes
- Rows are sorted by `n_reactions_in_top10` (most-shared metabolites first), then by ID.
- The `reactions_in_top10` ranks refer to the `rank` column of
  `top10_reactions_stereo_significant.csv`.
- For the full per-metabolite reaction lists (e.g. H⁺ appears in ~30k reactions),
  see the companion `..._full_reaction_membership.tsv`.
