"""Structural applicability flags, without a fabricated uncertainty floor.

These flags identify chemistry needing validation; neither broad UMA training
nor a small sampling spread establishes accurate aqueous thermodynamics.
Coordination/speciation, solvation and sampling errors can remain systematic.
The numerical sigma policy is unchanged: this layer supplies informational
flags and sigma_floor=0 for compatibility, not a guarantee of coverage.
"""
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

_CORE = {1, 6, 7, 8, 15, 16}            # H C N O P S -- the TECRDB calibration elements
_HALOGEN = {9, 17, 35, 53}              # F Cl Br I
_DIVALENT_CATION = {12, 20, 25, 26, 27, 28, 29, 30}   # Mg Ca Mn Fe Co Ni Cu Zn -- coordination-speciation note
_LARGE_HEAVY = 50                       # heavy-atom threshold for the "very large" flag
_PHOS_OR_CARBOX = Chem.MolFromSmarts("[$([PX4](=O)),$([CX3](=O)[OX1,OX2])]")   # phosphate or carboxyl(ate)


def _arom_N_rings(m):
    """Number of aromatic rings containing >=1 nitrogen (pyridine/pyrimidine/imidazole/purine...)."""
    ri = m.GetRingInfo(); c = 0
    for ring in ri.AtomRings():
        if (all(m.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
                and any(m.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in ring)):
            c += 1
    return c


def ood_assessment(species):
    """species = {name: [coeff, charge, smi]}. Returns
        {"ood": bool, "reasons": [str], "flags": [str], "sigma_floor": float}
    FLAG-ONLY layer: `flags` are informational notes for the consumer; `sigma_floor` is always 0.0 and
    `reasons` empty (no structural signal inflates sigma -- see module docstring). `ood` = bool(flags),
    i.e. "notable features present", NOT "sigma widened". Kept as a dict for API stability."""
    flags = []
    elements = set()
    max_heavy = 0
    has_phosphate_or_carboxylate = False
    r_arN = p_arN = 0
    for coeff, q, smi in (tuple(v) for v in species.values()):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        for a in m.GetAtoms():
            elements.add(a.GetAtomicNum())
        max_heavy = max(max_heavy, m.GetNumHeavyAtoms())
        if m.HasSubstructMatch(_PHOS_OR_CARBOX):
            has_phosphate_or_carboxylate = True
        n = _arom_N_rings(m)
        if coeff < 0:
            r_arN += abs(coeff) * n
        else:
            p_arN += abs(coeff) * n

    # Free metal plus a binding group does not model coordination equilibria.
    # This check does not establish whether electronic predictions are accurate.
    cations = elements & _DIVALENT_CATION
    if cations and has_phosphate_or_carboxylate:
        syms = " ".join(sorted(Chem.GetPeriodicTable().GetElementSymbol(z) for z in cations))
        flags.append(f"divalent cation [{syms}] + phosphate/carboxylate: coordination speciation not "
                     f"modelled; electronic and solution-state accuracy require validation")
    other_uncommon = elements - _CORE - _HALOGEN - _DIVALENT_CATION
    if other_uncommon:
        syms = " ".join(sorted(Chem.GetPeriodicTable().GetElementSymbol(z) for z in other_uncommon))
        flags.append(f"element(s) [{syms}] rare in the σ-calibration set; predictive accuracy and "
                     f"interval coverage are not established for this chemistry")
    if (p_arN - r_arN) != 0:
        flags.append("de-novo aromatic N-heterocycle formed/destroyed (multi-bond condensation class; "
                     "empirical, mechanism undiagnosed)")
    if max_heavy > _LARGE_HEAVY:
        flags.append(f"very large molecule ({max_heavy} heavy atoms): finite sampling may miss basins; "
                     f"U_samp does not bound that error")

    return {"ood": bool(flags), "reasons": [], "flags": flags, "sigma_floor": 0.0}
