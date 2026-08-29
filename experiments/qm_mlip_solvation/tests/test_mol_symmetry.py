"""Regression test for the RRHO rotational term: geometry (linear vs nonlinear -> 5 vs 6 external modes)
and the external rotational symmetry number sigma. Locks the thermal_solv fix that replaced the
hard-coded geometry='nonlinear', sigma=1. Run: <uma-python> tests/test_mol_symmetry.py  (needs rdkit).

The high-impact cases (water, CO2/O2/H2/N2, NH3) MUST be exact -- water sigma=2 is the ~1.7 kJ/mol
hydrolysis correction; the linear species are the ModelSEED-ubiquitous ones the old code mis-scored.
Rare high-symmetry floppy species (benzene, H3PO4) may UNDERcount conservatively -- never overcount.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from mol_symmetry import geometry_and_sigma
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

# (SMILES, name, expected sigma, expected geometry)
MUST_PASS = [
    ("O", "water", 2, "nonlinear"),
    ("O=C=O", "CO2", 2, "linear"),
    ("O=O", "O2", 2, "linear"),
    ("[H][H]", "H2", 2, "linear"),
    ("N#N", "N2", 2, "linear"),
    ("[C-]#[O+]", "CO", 1, "linear"),
    ("C#N", "HCN", 1, "linear"),
    ("N", "NH3", 3, "nonlinear"),
    ("C", "CH4", 12, "nonlinear"),
    ("CC", "ethane", 6, "nonlinear"),
    ("CO", "methanol", 1, "nonlinear"),
    ("OC[C@@H](O)[C@@H](O)[C@H](O)[C@H](O)C=O", "glucose", 1, "nonlinear"),
]


def _geom(smi):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(m, randomSeed=3)
    try:
        AllChem.MMFFOptimizeMolecule(m)
    except Exception:
        pass
    c = m.GetConformer()
    sym = [a.GetSymbol() for a in m.GetAtoms()]
    crd = [[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y, c.GetAtomPosition(i).z]
           for i in range(m.GetNumAtoms())]
    return sym, crd


def main():
    fails = 0
    for smi, nm, want_sigma, want_geo in MUST_PASS:
        s, c = _geom(smi)
        geo, sigma, ndrop = geometry_and_sigma(s, c)
        ok = (sigma == want_sigma and geo == want_geo)
        if not ok:
            fails += 1
        print(f"  {nm:9s} got=({geo},{sigma},drop{ndrop})  want=(sigma={want_sigma},{want_geo})  "
              f"{'OK' if ok else 'FAIL'}")
    assert fails == 0, f"{fails} symmetry regressions"
    print(f"\nPASS: {len(MUST_PASS)}/{len(MUST_PASS)} high-impact symmetry cases exact")


if __name__ == "__main__":
    main()
