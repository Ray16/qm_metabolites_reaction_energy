"""Build defined-anomer variants of the flagged sugars and VERIFY the assignment before any GPU work.

Each flagged sugar has ONLY its anomeric carbon undefined (ring carbons already defined -> it is clean
alpha/beta mixing, not a gluco/manno/galacto scramble). We pin the PHYSICALLY DOMINANT aqueous anomer:
beta-D-glucopyranose (~62%) for the pyranoses, beta-D-fructofuranose for the stored furanoses.

Assignment method (robust, not hand-typed @/@@): for each sugar, enumerate both anomeric chiralities,
take CIP R/S of the anomeric carbon, and match to the CIP of the anomeric carbon in a VERIFIED beta
reference for the same ring system. The anomeric CIP is local (ring + immediate neighbours), so it
transfers from glucose->G6P and fructose->F6P/FBP (the distal phosphate does not touch C1/C2 priorities).
Prints stored/alpha/beta SMILES + InChIKey so the beta pin can be eyeballed before running.
"""
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import inchi, AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

D = json.load(open("scripts/reactions_tecrdb_all.json"))
SUGARS = ["D-Glucose", "D-glucose-6-phosphate", "D-Glucosamine phosphate",
          "D-Fructose", "D-fructose-6-phosphate", "D-fructose-1,6-bisphosphate"]


def anomeric_idx(m):
    """ring sp3 C bonded to a ring O and an exocyclic O -> the anomeric carbon."""
    for a in m.GetAtoms():
        if a.GetSymbol() != "C" or not a.IsInRing():
            continue
        ringO = any(n.GetSymbol() == "O" and n.IsInRing() for n in a.GetNeighbors())
        exoO = any(n.GetSymbol() == "O" and not n.IsInRing() for n in a.GetNeighbors())
        if ringO and exoO:
            return a.GetIdx()
    return None


def _ref_carbon(m, ring):
    """Configurational reference: the ring stereocentre bearing an EXOCYCLIC carbon (C5 in aldo-pyranose /
    keto-furanose, which carries the CH2OH that defines D/L). Return (ring_atom_idx, exo_C_idx)."""
    for a in m.GetAtoms():
        if a.GetSymbol() != "C" or not a.IsInRing() or a.GetIdx() == anomeric_idx(m):
            continue
        exoC = [n for n in a.GetNeighbors() if n.GetSymbol() == "C" and not n.IsInRing()]
        inO = [n for n in a.GetNeighbors() if n.GetSymbol() == "O" and n.IsInRing()]
        if exoC and inO:                                    # ring C next to the ring-O, bearing CH2OH
            return a.GetIdx(), exoC[0].GetIdx()
    return None, None


def anomer_label(smi):
    """3D-geometry alpha/beta of the (defined) anomeric centre: beta = anomeric exocyclic O is on the SAME
    ring face as the reference carbon's exocyclic CH2OH. Unambiguous, reference-free."""
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    if AllChem.EmbedMolecule(m, randomSeed=1) != 0:
        AllChem.EmbedMolecule(m, useRandomCoords=True, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(m, maxIters=200)
    conf = m.GetConformer(); P = conf.GetPositions()
    m2 = Chem.RemoveHs(m)
    ai = anomeric_idx(m2)
    ring = [r for r in m2.GetRingInfo().AtomRings() if ai in r][0]
    centroid = P[list(ring)].mean(axis=0)
    # ring normal via SVD of ring atom coords
    u, s, vt = np.linalg.svd(P[list(ring)] - centroid); normal = vt[2]
    aexoO = [n.GetIdx() for n in m2.GetAtomWithIdx(ai).GetNeighbors() if n.GetSymbol() == "O" and not n.IsInRing()][0]
    ri, rexo = _ref_carbon(m2, ring)
    if ri is None:
        return "?"
    s_ano = np.dot(P[aexoO] - P[ai], normal)
    s_ref = np.dot(P[rexo] - P[ri], normal)
    return "beta" if s_ano * s_ref > 0 else "alpha"


def both_anomers(smi):
    """Return {'alpha': smiles, 'beta': smiles} using the geometry label."""
    out = {}
    for tag in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
        m = Chem.MolFromSmiles(smi)
        i = anomeric_idx(m)
        m.GetAtomWithIdx(i).SetChiralTag(tag)
        iso = Chem.MolToSmiles(m)
        out[anomer_label(iso)] = iso
    return out


def ikey(smi):
    m = Chem.MolFromSmiles(smi)
    return inchi.MolToInchiKey(m) if m else "n/a"


def main():
    # VALIDATE the geometry detector on known glucose (beta must be WQZGKKKJIJFFOK-VFUOTHLCSA-N)
    gluc = "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"
    g = both_anomers(gluc)
    print("DETECTOR VALIDATION (D-glucopyranose):")
    print(f"   beta  -> {g.get('beta')}  ikey={ikey(g.get('beta',''))}  (expect ...-VFUOTHLCSA-N)")
    print(f"   alpha -> {g.get('alpha')} ikey={ikey(g.get('alpha',''))} (expect ...-DVKNGEFBSA-N)")
    ok = ikey(g.get("beta", "")).endswith("VFUOTHLCSA-N")
    print(f"   detector {'PASS' if ok else 'FAIL'}\n")

    def smi(nm):
        for rid, v in D.items():
            if nm in v["species"]:
                return v["species"][nm]
        return None

    variants = {}
    for nm in SUGARS:
        sp = smi(nm)
        if sp is None:
            print(f"  {nm}: not found"); continue
        c, q, s = sp
        m = Chem.MolFromSmiles(s)
        ring = "furanose" if any(len(r) == 5 for r in m.GetRingInfo().AtomRings()) else "pyranose"
        anos = both_anomers(s)                              # {'alpha':.., 'beta':..}
        beta_smi, alpha_smi = anos.get("beta"), anos.get("alpha")
        variants[nm] = dict(q=q, stored=s, beta=beta_smi, alpha=alpha_smi, ring=ring)
        print(f"{nm}  ({ring}, q={q})")
        print(f"   stored: {s}   ikey={ikey(s)}")
        print(f"   ALPHA : {alpha_smi}   ikey={ikey(alpha_smi) if alpha_smi else 'n/a'}")
        print(f"   BETA  : {beta_smi}   ikey={ikey(beta_smi) if beta_smi else 'n/a'}   <-- physical pin\n")
    json.dump(variants, open("artifacts/anomer_variants.json", "w"), indent=1)
    print("wrote artifacts/anomer_variants.json")


if __name__ == "__main__":
    main()
