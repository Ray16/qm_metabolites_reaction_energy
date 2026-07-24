#!/usr/bin/env python
"""Generic pH-7 microspecies auditing -- no compound ids hardcoded.

Two species-level mistakes cost tens of kJ/mol and neither is visible in a
reaction's atom balance, so both must be caught structurally rather than by name:

  1. Wrong protonation microstate. ModelSEED's "major species" is not always the
     dominant one at the measurement pH. A site whose typical group pKa is well
     ABOVE the pH should be protonated; if the structure carries it deprotonated
     the pipeline is scoring a minor species, and the resulting error is the
     error of the COMPUTED pKa (large, because anion solvation is hard).

  2. Un-modelled covalent hydration. Aldehydes -- especially those flanked by a
     carbonyl or other EWG, like methylglyoxal -- exist in water almost entirely
     as the gem-diol. Experimental thermodynamic values refer to the hydrated
     pool, not the free carbonyl.

audit() flags both from SMILES alone, so it applies to any reaction set. The
actual replacement energies still have to be computed (build_microspecies.py);
this module decides WHAT needs computing and supplies the swap table.

Typical group pKa values are textbook aqueous values for the parent functional
group; they are used only to decide which microspecies to model, never as an
energy, so moderate substituent shifts do not change the decision.
"""
from __future__ import annotations

import json
import os

from rdkit import Chem

HERE = os.path.dirname(os.path.abspath(__file__))
SWAP_TABLE = os.path.join(HERE, "microspecies_swaps.json")

# (label, SMARTS matching the DEPROTONATED site, typical pKa of the conjugate acid)
DEPROT_SITES = [
    ("thiolate",     "[$([S-][CX4]),$([S-]c)]",     9.0),
    ("phenolate",    "[$([O-]c)]",                 10.0),
    ("alkoxide",     "[$([O-][CX4])]",             15.5),
    ("amide anion",  "[$([N-]C=O)]",               15.0),
    ("carboxylate",  "[$([O-]C=O)]",                4.8),
    ("phosphate O-", "[$([O-][PX4])]",              6.5),
    ("sulfonate",    "[$([O-]S(=O)=O)]",           -1.0),
]

# carbonyls that are substantially hydrated in water
HYDRATION_SMARTS = [
    ("alpha-dicarbonyl aldehyde", "[CX3H1](=O)[CX3]=O"),
    ("alpha-halo/EWG aldehyde",   "[CX3H1](=O)[CX4]([F,Cl,Br])"),
    ("formaldehyde",              "[CX3H2]=O"),
]


def audit(smiles: str, pH: float = 7.0, margin: float = 1.0) -> list[dict]:
    """Flag species-level modelling problems for one structure.

    A deprotonated site is flagged when its typical pKa exceeds pH + margin,
    i.e. the protonated form dominates and we are modelling a minor species.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [dict(kind="parse_error", detail=smiles)]
    issues = []
    for label, sma, pka in DEPROT_SITES:
        n = len(mol.GetSubstructMatches(Chem.MolFromSmarts(sma)))
        if n and pka > pH + margin:
            issues.append(dict(kind="protonation", site=label, n=n, pKa=pka,
                               detail=f"{n}x {label} present but pKa~{pka} > pH {pH}"
                                      f"; protonated form dominates"))
    for label, sma in HYDRATION_SMARTS:
        n = len(mol.GetSubstructMatches(Chem.MolFromSmarts(sma)))
        if n:
            issues.append(dict(kind="hydration", site=label, n=n,
                               detail=f"{n}x {label}; likely present as the "
                                      f"gem-diol hydrate in water"))
    return issues


def corrected_structures(smiles: str, pH: float = 7.0) -> list[dict]:
    """Build the corrected structure(s) implied by audit(), from SMILES alone.

    protonation -> protonate every flagged site (charge 0, one more H)
    hydration   -> add water across the flagged aldehyde C=O to give the gem-diol

    Returns [{kind, smiles, charge, n_water}], where n_water is how many waters
    were consumed (the caller must subtract mu(H2O) for each).
    """
    out = []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out
    flags = audit(smiles, pH)

    prot = [f for f in flags if f["kind"] == "protonation"]
    if prot:
        rw = Chem.RWMol(mol)
        for label, sma, pka in DEPROT_SITES:
            if not any(f["site"] == label for f in prot):
                continue
            for match in mol.GetSubstructMatches(Chem.MolFromSmarts(sma)):
                a = rw.GetAtomWithIdx(match[0])
                if a.GetFormalCharge() < 0:
                    a.SetFormalCharge(a.GetFormalCharge() + 1)
                    a.SetNumExplicitHs(a.GetNumExplicitHs() + 1)
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        out.append(dict(kind="protonation", smiles=Chem.MolToSmiles(m),
                        charge=Chem.GetFormalCharge(m), n_water=0))

    hyd = [f for f in flags if f["kind"] == "hydration"]
    if hyd:
        rw = Chem.RWMol(mol)
        n = 0
        for label, sma in HYDRATION_SMARTS:
            if not any(f["site"] == label for f in hyd):
                continue
            for match in mol.GetSubstructMatches(Chem.MolFromSmarts(sma)):
                c = rw.GetAtomWithIdx(match[0])
                o = next((b.GetOtherAtom(c) for b in c.GetBonds()
                          if b.GetBondType() == Chem.BondType.DOUBLE
                          and b.GetOtherAtom(c).GetSymbol() == "O"), None)
                if o is None:
                    continue
                rw.GetBondBetweenAtoms(c.GetIdx(), o.GetIdx()).SetBondType(
                    Chem.BondType.SINGLE)
                o.SetNumExplicitHs(1)
                oh = rw.AddAtom(Chem.Atom(8))
                rw.AddBond(c.GetIdx(), oh, Chem.BondType.SINGLE)
                rw.GetAtomWithIdx(oh).SetNumExplicitHs(1)
                c.SetNumExplicitHs(max(0, c.GetNumExplicitHs()))
                n += 1
                break            # one hydration per group is enough
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        out.append(dict(kind="hydration", smiles=Chem.MolToSmiles(m),
                        charge=Chem.GetFormalCharge(m), n_water=n))
    return out


def load_swaps() -> dict:
    """cpd_id -> {species_key, kind}. Empty if no swaps have been computed."""
    return json.load(open(SWAP_TABLE)) if os.path.isfile(SWAP_TABLE) else {}


def audit_set(metabolites, pH=7.0):
    """Audit a list of {id,name,smiles} and print a report. Returns the flags."""
    flagged = {}
    for m in metabolites:
        iss = audit(m["smiles"], pH)
        if iss:
            flagged[m["id"]] = iss
            print(f"{m['id']:10} {m['name'][:34]:34}")
            for i in iss:
                print(f"           - [{i['kind']}] {i['detail']}")
    if not flagged:
        print("no microspecies issues found")
    return flagged


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "metabolites.json")
    ph = float(sys.argv[2]) if len(sys.argv) > 2 else 7.0
    print(f"=== microspecies audit @ pH {ph} | {os.path.basename(path)} ===")
    audit_set(json.load(open(path)), ph)
