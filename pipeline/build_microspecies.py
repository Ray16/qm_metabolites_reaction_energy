#!/usr/bin/env python
"""Improvement: correct the pH-7 dominant microspecies for two compounds.

  GSH (cpd00042) is modelled as the THIOLATE (C[S-], q=-2), but the cysteine
  thiol pKa is ~9.0, so at pH 7 GSH is >99% the neutral THIOL (CS, q=-1).
  GSH enters the four redox reactions with coefficient +-2 and has no thiol
  counterpart on the product side, so the error cannot cancel.

  Methylglyoxal (cpd00428) is modelled as the free dialdehyde (CC(=O)C=O), but
  in water it is almost entirely the aldehyde gem-DIOL hydrate. The TECRDB
  glyoxalase measurement refers to total dissolved methylglyoxal, i.e. the
  hydrate. Handled as a separate species plus explicit water (see below).

In exact theory dG'^o is invariant to the microspecies chosen -- the Alberty
transform accounts for N_H and z. It is NOT invariant in practice, because our
QM carries the error of the *computed* pKa. Using the dominant species minimises
that sensitivity; the spread between the two choices measures it.

Builds the ensembles with the same ETKDG+xtb recipe as the production pipeline.
Stage B (UMA) is run separately by run_uma_microspecies.py.

Run:  /homes/rzhu/miniforge3/envs/palm/bin/python build_microspecies.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# reuse the production ensemble builder verbatim, redirected to its own outputs
os.environ.setdefault("FAST_ENS_DIR", os.path.join(HERE, "geometries_microspecies"))
os.environ.setdefault("FAST_ENS_JSON", os.path.join(HERE, "microspecies_xtb.json"))
os.environ.setdefault("FAST_EMBED", "400")
os.environ.setdefault("FAST_NSTART", "64")
os.environ.setdefault("FAST_WINDOW_KJ", "30")

MET_JSON = os.path.join(HERE, "microspecies_metabolites.json")

# id -> (name, SMILES, charge).  The "_alt" ids are the corrected species; they
# are scored alongside the originals so the two choices can be compared.
SPECIES = [
    dict(id="cpd00042_thiol", name="GSH (thiol, pH-7 dominant)", charge=-1,
         smiles="[NH3+][C@@H](CCC(=O)N[C@@H](CS)C(=O)NCC(=O)[O-])C(=O)[O-]"),
    dict(id="cpd00428_hydrate", name="Methylglyoxal aldehyde-hydrate", charge=0,
         smiles="CC(=O)C(O)O"),
    dict(id="h2o", name="Water", charge=0, smiles="O"),
]


def main():
    for s in SPECIES:
        s.setdefault("formula", "")
        s.setdefault("inchikey", "")
        s.setdefault("opentecr_species", "")
    json.dump(SPECIES, open(MET_JSON, "w"), indent=1)
    print(f"wrote {MET_JSON} ({len(SPECIES)} species)")

    os.environ["FAST_MET_JSON"] = MET_JSON
    import build_ensembles_fast as B
    B.MET_JSON = MET_JSON
    B.main()


if __name__ == "__main__":
    main()
