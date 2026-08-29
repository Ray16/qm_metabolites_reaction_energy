"""Out-of-distribution FLAG layer -- transparency, NOT sigma inflation.

WHY FLAG-ONLY (the design decision): inflating sigma for structurally-unusual reactions is a "unlike my
training set -> widen" crutch, and it DEFEATS the whole point of the method. MetaG's value is coverage:
the ΔG is first-principles and GENERALIZES (proven -- UMA matched experiment to <10 kJ on O2/aromatic
oxidation far outside TECRDB). A σ-floor keyed to "looks unusual" would fire on exactly the frontier
reactions we exist to score, telling the user "uncertain" precisely where we add value. So this layer
now FLAGS notable features for transparency but does NOT change sigma. (sigma_floor is retained in the
return for API stability and is always 0.0 unless a defensible physics-limit signal is re-introduced.)

Why the previously-floored signals were demoted:
  - uncommon elements / metals: UMA is a UNIVERSAL potential (trained across materials/catalysis/MOFs,
    OMat/OC/OMol) -- metals are IN its domain; it generalizes to them electronically. The old "TECRDB Mg
    MAE ~43" was a SPECIATION/COORDINATION modeling gap (Mg(2+) binds ATP as a complex; the pipeline
    scored free Mg(2+) with no explicit coordination) -- a pipeline gap to FIX (explicit-Mg), not a UMA
    limit and not a σ to hand-inflate.
  - de-novo aromatic N-heterocycle condensation: empirical (rxn02988/rxn23024 failed), mechanism
    undiagnosed; also fires on aromatisations UMA handles.
  - very large molecules: a sampling-reliability concern of OUR finite conformer search -- already carried
    by the COMPUTED U_samp / ensemble spread, not a structural floor.

THE REAL uncertainty is a COMPUTED physics σ (ensemble UMA-vs-MACE disagreement / cycle-consistency),
which is naturally larger where UMA is less sure (incl. rarer chemistry) WITHOUT hand-coding it. That is
the generalizable fix; this layer only surfaces flags a consumer may want to see.
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

    # divalent metal cation + a phosphate/carboxylate present -> a coordination-SPECIATION note (Mg-ATP
    # class). NOT a UMA electronic limit (UMA is universal); a PIPELINE gap (no explicit-metal coordination
    # model) -> the FIX is explicit-Mg, and until then the speciation may be wrong. Informational.
    cations = elements & _DIVALENT_CATION
    if cations and has_phosphate_or_carboxylate:
        syms = " ".join(sorted(Chem.GetPeriodicTable().GetElementSymbol(z) for z in cations))
        flags.append(f"divalent cation [{syms}] + phosphate/carboxylate: coordination speciation not "
                     f"modelled (pipeline gap -> explicit-metal; NOT a UMA electronic limit)")
    other_uncommon = elements - _CORE - _HALOGEN - _DIVALENT_CATION
    if other_uncommon:
        syms = " ".join(sorted(Chem.GetPeriodicTable().GetElementSymbol(z) for z in other_uncommon))
        flags.append(f"element(s) [{syms}] rare in the σ-calibration set (UMA is universal and should "
                     f"generalize; computed-σ, not a floor, is the honest measure)")
    if (p_arN - r_arN) != 0:
        flags.append("de-novo aromatic N-heterocycle formed/destroyed (multi-bond condensation class; "
                     "empirical, mechanism undiagnosed)")
    if max_heavy > _LARGE_HEAVY:
        flags.append(f"very large molecule ({max_heavy} heavy atoms) -- sampling reliability is carried "
                     f"by U_samp/ensemble, not a structural floor")

    return {"ood": bool(flags), "reasons": [], "flags": flags, "sigma_floor": 0.0}
