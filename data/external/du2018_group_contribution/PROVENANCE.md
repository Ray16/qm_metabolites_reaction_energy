# Du et al. 2018 group-contribution dataset — provenance

Curated aqueous thermodynamic data used as **species-level validation targets** for the
ab-initio ΔG workflow (dG_f / dH_f / dS_f per compound).

## Source
- Repo: https://github.com/bdu91/group-contribution  (MIT license)
- Commit: `8eb106629ecba6d0b806a191afd630a64d5e3954` (see `.du_commit`)
- Paper: Du, Zhang, Grubner, Yurkovich, Palsson, Zielinski. "Temperature-Dependent
  Estimation of Gibbs Energies Using an Updated Group-Contribution Method."
  *Biophys. J.* 114:2691–2702 (2018). PDF in `paper/`.

## These are already-aggregated external datasets (no separate download needed)
Du's tables curate/compile from:
- Alberty, *Thermodynamics of Biochemical Reactions* (2003)
- SUPCRT92 (Johnson, Oelkers & Helgeson 1992)
- Organic Compounds Hydration Properties Database (Plyasunova, Plyasunov & Shock 2004)
- IUPAC Stability Constants Database (pKa, metal binding)
So `organic_cpd_thermo_data.csv` IS the merged ΔfG/ΔfH/ΔfS/Cp compilation.

## Canonical load path
Per project convention the formation table is copied to and loaded from:
`ModelSEED_FAISS/data/organic_cpd_thermo_data.csv`
(`build_validation_set.py` reads it from there; SMILES joins use the `raw/` tables below.)

## Files (raw/)
| file | contents |
|---|---|
| `organic_cpd_thermo_data.csv` | **formation values**: dG_f (312 cpds), dH_f (254), dS_f (669 species / ~489 cpds), dG_f_prime (203), Cp (244). Keyed by ChEBI/PubChem id + charge. No SMILES. |
| `TECRDB_compounds_data.csv` | per-species table incl. `smiles_form`, charge, groups (SMILES source for the join) |
| `dSf_pKMg_data.csv` | ΔfS° / pKMg training features per species incl. `smiles_form` |
| `TECRDB_rxn_thermo_data.csv` | reaction-level K′ and ΔrH′° (207 reactions) — for the ΔrH/ΔrS reaction-decomposition track |
| `dSr_training_data.csv` | 92 ΔrS° values from van't Hoff slopes |
| `cid_names.csv` | compound id → name → source DB |
| `metal_thermo_data.csv` | metal-binding thermo |

## Identifier prefixes
`CHB_` ChEBI · `PBC_` PubChem · `CHS_` ChemSpider · `MAN_` manual entry · `MNXM` MetaNetX.
Species ids append the charge, e.g. `CHB_15422_-1`. Matching to project species is done by
**InChIKey** (from SMILES) — see `build_validation_set.py` / `OVERLAP_REPORT.md`.
