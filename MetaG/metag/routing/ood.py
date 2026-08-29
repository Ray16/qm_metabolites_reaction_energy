"""Out-of-distribution gate for the calibrated uncertainty.

WHY: sigma_class is calibrated on TECRDB (CHNOPS organics, |charge| small, no metals, modest size). This
gate detects, from STRUCTURE, where that calibrated sigma is untrustworthy. It never NARROWS sigma.

TWO KINDS OF SIGNAL, deliberately separated -- because a blunt "unlike my training set -> widen" would
fire on EVERY frontier reaction and thus undercut the whole coverage claim (the point of the method is to
generalize; the ΔG does -- proven on O2/aromatic oxidation vs experiment). Only the first kind floors σ:

  * PHYSICS-LIMIT signals (FLOOR sigma): known approximation failures a first-principles method should
    OWN, computable from structure for any reaction -- NOT "unlike the training set":
      - uncommon elements / metals: untrained MLIP domain + likely multireference (TECRDB Mg-coord MAE ~43).
      - very large molecules: conformer-sampling / floppiness reliability limit.
  * FLAG-ONLY signals (surfaced, do NOT floor sigma): empirical "we have SEEN this fail but have not
    pinned the physical mechanism". Silently widening these is the crutch that would defeat coverage:
      - de-novo aromatic N-heterocycle formation/destruction: UMA missed rxn02988/rxn23024, but WHY is
        undiagnosed (product sampling? electronic?), and the same signal fires on aromatisation
        isomerisations UMA handles -> informational only until it is shown to be a physics limit.

HONESTY: floors are conservative heuristics (no held-out OOD ground truth). The GENERALIZABLE fix for
uncertainty is a COMPUTED physics σ (ensemble disagreement / cycle-consistency), not this gate; this gate
only owns the honest physics-limit floors + surfaces empirical flags.
"""
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

_CORE = {1, 6, 7, 8, 15, 16}            # H C N O P S -- the TECRDB calibration elements
_HALOGEN = {9, 17, 35, 53}              # F Cl Br I -- mildly OOD (sparse in TECRDB)

# conservative sigma FLOORS (kJ/mol) for PHYSICS-LIMIT signals only -- "off-distribution -> at least this
# uncertain". Tied to documented scales (hardest calibrated class ~ glycosyl 24.6; TECRDB Mg MAE ~43),
# NOT error-calibrated. Max wins. The aromatic-heterocycle signal is FLAG-ONLY (no floor) -- see docstring.
_FLOOR_METAL = 30.0                     # metals / uncommon elements (strong, measured OOD)
_FLOOR_HALOGEN = 18.0                   # halogens (mild)
_FLOOR_LARGE = 18.0                     # very large molecule
_LARGE_HEAVY = 50                       # heavy-atom threshold for "very large"


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
    `reasons`  = PHYSICS-LIMIT signals that FLOORED sigma (elements, size). `ood` = bool(reasons).
    `flags`    = FLAG-ONLY empirical notes that do NOT change sigma (aromatic-heterocycle condensation).
    `sigma_floor` = max floor over physics-limit signals (0 if none). Never narrows sigma."""
    reasons, flags, floor = [], [], 0.0
    elements = set()
    max_heavy = 0
    r_arN = p_arN = 0
    for coeff, q, smi in (tuple(v) for v in species.values()):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        for a in m.GetAtoms():
            elements.add(a.GetAtomicNum())
        max_heavy = max(max_heavy, m.GetNumHeavyAtoms())
        n = _arom_N_rings(m)
        if coeff < 0:
            r_arN += abs(coeff) * n
        else:
            p_arN += abs(coeff) * n

    # --- PHYSICS-LIMIT signals: floor sigma ---
    uncommon = elements - _CORE - _HALOGEN
    halogens = elements & _HALOGEN
    if uncommon:
        syms = " ".join(sorted(Chem.GetPeriodicTable().GetElementSymbol(z) for z in uncommon))
        reasons.append(f"uncommon element(s) [{syms}] outside the CHNOPS calibration set")
        floor = max(floor, _FLOOR_METAL)
    elif halogens:
        syms = " ".join(sorted(Chem.GetPeriodicTable().GetElementSymbol(z) for z in halogens))
        reasons.append(f"halogen(s) [{syms}] (sparse in calibration)")
        floor = max(floor, _FLOOR_HALOGEN)
    if max_heavy > _LARGE_HEAVY:
        reasons.append(f"very large molecule ({max_heavy} heavy atoms > {_LARGE_HEAVY})")
        floor = max(floor, _FLOOR_LARGE)

    # --- FLAG-ONLY signals: surfaced, do NOT floor sigma (mechanism undiagnosed; would defeat coverage) ---
    if (p_arN - r_arN) != 0:
        flags.append("de-novo aromatic N-heterocycle formed/destroyed (multi-bond condensation class; "
                     "empirical, no sigma effect until mechanism diagnosed)")

    return {"ood": bool(reasons), "reasons": reasons, "flags": flags, "sigma_floor": round(floor, 1)}
