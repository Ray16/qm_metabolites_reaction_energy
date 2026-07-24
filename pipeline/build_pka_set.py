#!/usr/bin/env python
"""Calibration set for the anion-solvation error, via experimental pKa.

The GSH diagnostic showed the pipeline puts the thiol/thiolate pair 42 kJ/mol off
(implied pKa 16.4 vs 9.0). That is an absolute-ion-solvation error, and it is the
same physics the redox reactions depend on. This builds small acid/base pairs
whose pKa is known to <0.1 units and whose functional groups are exactly the ones
carrying charge in the metabolites: thiol, carboxylate, phosphate, phenol.

Crucially the set also contains CATIONIC acids (BH+ -> B + H+), which involve no
anion at all. If the error appears in the anionic pairs but not the cationic
ones, it is specifically anion solvation rather than a bad proton reference --
the two are otherwise indistinguishable from the metabolite data alone.

Run:  /homes/rzhu/miniforge3/envs/palm/bin/python build_pka_set.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("FAST_ENS_DIR", os.path.join(HERE, "geometries_pka"))
os.environ.setdefault("FAST_ENS_JSON", os.path.join(HERE, "pka_xtb.json"))
os.environ.setdefault("FAST_EMBED", "200")
os.environ.setdefault("FAST_NSTART", "32")
os.environ.setdefault("FAST_WINDOW_KJ", "30")

MET_JSON = os.path.join(HERE, "pka_metabolites.json")
PAIRS_JSON = os.path.join(HERE, "pka_pairs.json")

# (key, acid SMILES, acid charge, base SMILES, base charge, exp pKa, group, kind)
PAIRS = [
    ("acetic",   "CC(=O)O",        0, "CC(=O)[O-]",       -1,  4.76, "carboxyl",  "anionic"),
    ("propanoic", "CCC(=O)O",      0, "CCC(=O)[O-]",      -1,  4.87, "carboxyl",  "anionic"),
    ("lactic",   "CC(O)C(=O)O",    0, "CC(O)C(=O)[O-]",   -1,  3.86, "carboxyl",  "anionic"),
    ("MeSH",     "CS",             0, "C[S-]",            -1, 10.33, "thiol",     "anionic"),
    ("EtSH",     "CCS",            0, "CC[S-]",           -1, 10.61, "thiol",     "anionic"),
    ("phenol",   "c1ccccc1O",      0, "c1ccccc1[O-]",     -1,  9.99, "phenol",    "anionic"),
    ("MeOPO3H",  "COP(=O)(O)[O-]", -1, "COP(=O)([O-])[O-]", -2,  6.31, "phosphate", "anionic"),
    ("H2PO4",    "OP(=O)(O)[O-]",  -1, "OP(=O)([O-])[O-]",  -2,  7.20, "phosphate", "anionic"),
    # cationic acids: BH+ -> B. No anion is created, so these isolate the
    # proton reference from the anion-solvation error.
    ("MeNH3",    "C[NH3+]",        1, "CN",                0, 10.66, "ammonium",  "cationic"),
    ("imidazolium", "c1c[nH+]c[nH]1", 1, "c1c[nH]cn1",     0,  6.95, "ammonium",  "cationic"),
    ("pyridinium", "c1cc[nH+]cc1", 1, "c1ccncc1",          0,  5.23, "ammonium",  "cationic"),
    # Charge ladder. The metabolites are polyanions (-2 to -4), but every pair
    # above forms only a MONO-anion, so applying a per-site value 3-4x over is
    # unsupported extrapolation. These reference pairs create anions of charge
    # -2, -3 and -4 so the correction can be resolved against charge state.
    # Pyrophosphate is itself in the metabolite set, and succinate/malonate/G6P
    # carry the intramolecular H-bonding that plain acetate lacks.
    ("succinate2", "OC(=O)CCC(=O)[O-]", -1, "[O-]C(=O)CCC(=O)[O-]", -2, 5.64, "carboxyl", "anionic"),
    ("malonate2",  "OC(=O)CC(=O)[O-]",  -1, "[O-]C(=O)CC(=O)[O-]",  -2, 5.70, "carboxyl", "anionic"),
    ("citrate3",   "OC(CC(=O)[O-])(CC(=O)[O-])C(=O)O", -2,
                   "OC(CC(=O)[O-])(CC(=O)[O-])C(=O)[O-]", -3, 6.40, "carboxyl", "anionic"),
    ("G6P2",       "OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1OP(=O)(O)[O-]", -1,
                   "OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1OP(=O)([O-])[O-]", -2,
                   6.11, "phosphate", "anionic"),
    ("PPi3",       "OP(=O)([O-])OP(=O)(O)[O-]",   -2, "OP(=O)([O-])OP(=O)([O-])[O-]", -3, 6.60, "phosphate", "anionic"),
    ("PPi4",       "OP(=O)([O-])OP(=O)([O-])[O-]", -3, "[O-]P(=O)([O-])OP(=O)([O-])[O-]", -4, 9.40, "phosphate", "anionic"),
    ("phosphate3", "OP(=O)([O-])[O-]",  -2, "[O-]P(=O)([O-])[O-]", -3, 12.35, "phosphate", "anionic"),
    # Nucleotide ladder. The small references above (PO4(3-), P2O7(4-)) are not
    # bound in a continuum solvent and fail the quality screen, leaving q<=-2
    # supported by almost nothing. AMP/ADP/ATP carry the same charge states on a
    # large, well-separated framework -- and are structurally the metabolites we
    # actually score -- so they are the references that can be trusted there.
    ("AMP2", "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)[O-])[C@@H](O)[C@H]1O", -1,
             "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])[O-])[C@@H](O)[C@H]1O", -2,
             6.20, "phosphate", "anionic"),
    ("ADP3", "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)(O)[O-])[C@@H](O)[C@H]1O", -2,
             "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O", -3,
             6.40, "phosphate", "anionic"),
    ("ATP4", "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)(O)[O-])[C@@H](O)[C@H]1O", -3,
             "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O", -4,
             6.50, "phosphate", "anionic"),
    ("glutarate2", "OC(=O)CCCC(=O)[O-]", -1, "[O-]C(=O)CCCC(=O)[O-]", -2, 5.41, "carboxyl", "anionic"),
    # Thiols at higher charge. Every thiol reference above forms a MONO-anion, so
    # the calibration has nothing to say about deprotonating a thiol on an
    # already-anionic framework -- which is exactly what GSH does, and why the
    # charge-ladder form fails the thiolate/thiol self-consistency test.
    ("thioglycolate2", "SCC(=O)[O-]",  -1, "[S-]CC(=O)[O-]",  -2, 10.22, "thiol", "anionic"),
    ("mercaptoprop2",  "SCCC(=O)[O-]", -1, "[S-]CCC(=O)[O-]", -2, 10.30, "thiol", "anionic"),
    # more support at k=3 and k=4, where the ladder currently rests on n=3 and n=1
    ("tricarballylate3", "OC(=O)CC(CC(=O)[O-])C(=O)[O-]", -2,
                         "[O-]C(=O)CC(CC(=O)[O-])C(=O)[O-]", -3, 6.28, "carboxyl", "anionic"),
    ("GTP4", "Nc1nc2c(ncn2[C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)(O)[O-])[C@@H](O)[C@H]2O)c(=O)[nH]1", -3,
             "Nc1nc2c(ncn2[C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]2O)c(=O)[nH]1", -4,
             6.50, "phosphate", "anionic"),
]


def main():
    species, pairs = [], []
    for key, ah, qa, a, qb, pka, group, kind in PAIRS:
        species.append(dict(id=f"{key}_AH", name=f"{key} acid", smiles=ah,
                            charge=qa, formula="", inchikey="", opentecr_species=""))
        species.append(dict(id=f"{key}_A", name=f"{key} base", smiles=a,
                            charge=qb, formula="", inchikey="", opentecr_species=""))
        pairs.append(dict(key=key, acid=f"{key}_AH", base=f"{key}_A",
                          pKa_exp=pka, group=group, kind=kind,
                          q_acid=qa, q_base=qb))
    json.dump(species, open(MET_JSON, "w"), indent=1)
    json.dump(pairs, open(PAIRS_JSON, "w"), indent=1)
    print(f"wrote {MET_JSON} ({len(species)} species), {PAIRS_JSON} ({len(pairs)} pairs)")

    import build_ensembles_fast as B
    B.MET_JSON = MET_JSON
    B.main()


if __name__ == "__main__":
    main()
