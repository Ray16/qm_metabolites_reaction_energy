"""Out-of-distribution gate for the calibrated uncertainty.

WHY: sigma_class is calibrated on TECRDB (CHNOPS organics, |charge| small, no metals, modest size). A
ModelSEED reaction can be structurally UNLIKE that set, so the calibrated sigma is then an UNDERestimate
and the interval is falsely confident. This gate detects OOD features from STRUCTURE and (a) flags the
reaction OOD with reasons, (b) FLOORS sigma at a conservative level ("if it's off-distribution, treat it
as at least as uncertain as our hardest calibrated class"). It never NARROWS sigma.

HONESTY: the floors are conservative heuristics tied to documented failure scales, NOT error-calibrated
(we have no held-out OOD ground truth). The FLAG is the primary honest output; the floor is a guardrail.
Signals are chosen to be evidence-backed and validated against reactions with known verdicts:
  - uncommon elements / metals: measured OOD (TECRDB Mg-coordination MAE ~43). Cleanest signal.
  - de-novo aromatic N-heterocycle formation/destruction: the multi-bond ring-condensation class UMA
    struggles with (rxn02988 quinolinate synth, rxn23024 PLP synth); does NOT fire on O2/aromatic
    oxidation (UMA validated accurate there) or carbocyclisations (squalene/Diels-Alder, UMA fine).
    CONSERVATIVE: also fires on aromatisation isomerisations (e.g. dopachrome) UMA may handle -> errs
    toward caution by design.
  - very large molecules: beyond the calibration size range (sampling/floppy reliability).
"""
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

_CORE = {1, 6, 7, 8, 15, 16}            # H C N O P S -- the TECRDB calibration elements
_HALOGEN = {9, 17, 35, 53}              # F Cl Br I -- mildly OOD (sparse in TECRDB)

# conservative sigma FLOORS (kJ/mol): "off-distribution -> at least this uncertain". Tied to documented
# scales (hardest calibrated class ~ glycosyl 24.6; TECRDB Mg MAE ~43), NOT error-calibrated. Max wins.
_FLOOR_METAL = 30.0                     # metals / uncommon elements (strong, measured OOD)
_FLOOR_HALOGEN = 18.0                   # halogens (mild)
_FLOOR_HETEROAROM = 22.0               # de-novo aromatic N-heterocycle condensation
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
    """species = {name: [coeff, charge, smi]}. Returns {"ood": bool, "reasons": [str], "sigma_floor": float}.
    sigma_floor is 0 when in-distribution; otherwise the max conservative floor over triggered signals."""
    reasons, floor = [], 0.0
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
    if (p_arN - r_arN) != 0:
        reasons.append("de-novo aromatic N-heterocycle formed/destroyed (multi-bond condensation class)")
        floor = max(floor, _FLOOR_HETEROAROM)
    if max_heavy > _LARGE_HEAVY:
        reasons.append(f"very large molecule ({max_heavy} heavy atoms > {_LARGE_HEAVY})")
        floor = max(floor, _FLOOR_LARGE)

    return {"ood": bool(reasons), "reasons": reasons, "sigma_floor": round(floor, 1)}
