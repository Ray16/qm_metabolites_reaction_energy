"""Triage every scored reaction into {adjudication-GC, adjudication-EQ, split, FAILURE} and attach a
STRUCTURAL SIGNATURE to each, so UMA's failure modes can be addressed by structural class (not by
hand-labelling reactions). Reads any *_results/ dir + its matching *_inputs.json.

Verdict logic (ground-truth-free):
  * FAILURE  = UMA is implausibly far from BOTH incumbents (min |UMA-gc|,|UMA-eq| > FAIL_KJ) OR |UMA|
               is enormous (> HUGE_KJ). These are the modes to address.
  * split    = UMA disagrees in SIGN with both incumbents but within range (ambiguous, worth a look).
  * adj-GC / adj-EQ = UMA lands near one incumbent (the useful tie-break).

Structural signatures (from SMILES, reactant vs product side):
  n_O2            dioxygen created/destroyed  -> O2-redox mode (EC1.13/1.14)
  d_charge        sum |Δ formal charge|       -> anion/redox bookkeeping
  d_rings         product-ring - reactant-ring -> cyclization / ring cleavage
  d_arom_rings    aromatic-ring change         -> aromatization / ring-opening
  d_heavy_bonds   |Σ product bonds - Σ reactant bonds| over heavy atoms -> multi-bond rearrangement
"""
import json, os, sys, glob
from collections import defaultdict
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

FAIL_KJ = 100.0     # > this from BOTH incumbents = failure
HUGE_KJ = 250.0     # |UMA| beyond this is implausible for these tractable reactions

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "category_results")
INPUTS = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "category_inputs.json")
inp = json.load(open(INPUTS))


def _mol(smi):
    return Chem.MolFromSmiles(smi)


def signatures(species):
    """species: {name: [coeff, charge, smi]}. Reactants coeff<0, products coeff>0."""
    n_O2 = d_charge = 0
    r_rings = p_rings = r_arom = p_arom = r_bonds = p_bonds = 0
    for name, (coeff, q, smi) in species.items():
        m = _mol(smi)
        if m is None:
            continue
        n = abs(coeff)
        is_o2 = (sorted(a.GetAtomicNum() for a in m.GetAtoms()) == [8, 8])
        ri = m.GetRingInfo()
        nring = ri.NumRings()
        narom = sum(1 for r in ri.AtomRings() if all(m.GetAtomWithIdx(i).GetIsAromatic() for i in r))
        nbond = m.GetNumBonds()
        if is_o2:
            n_O2 += coeff                                  # net O2 (reactant side negative)
        d_charge += coeff * q                              # net charge flow (should ~0 with H+)
        if coeff < 0:
            r_rings += n * nring; r_arom += n * narom; r_bonds += n * nbond
        else:
            p_rings += n * nring; p_arom += n * narom; p_bonds += n * nbond
    return {
        "n_O2": abs(n_O2),
        "d_charge": d_charge,
        "d_rings": p_rings - r_rings,
        "d_arom_rings": p_arom - r_arom,
        "d_heavy_bonds": p_bonds - r_bonds,
    }


def verdict(dG, gc, eq):
    dgc, deq = dG - gc, dG - eq
    if abs(dG) > HUGE_KJ or (abs(dgc) > FAIL_KJ and abs(deq) > FAIL_KJ):
        return "FAILURE"
    # sign disagreement with both, but in range
    if (dG > 0) != (gc > 0) and (dG > 0) != (eq > 0):
        return "split"
    return "adj-GC" if abs(dgc) <= abs(deq) else "adj-EQ"


rows = []
for f in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
    r = json.load(open(f))
    rid = r["reaction"]
    if "error" in r:
        rows.append((rid, "ERROR", r["error"][:40], {}, r))
        continue
    v = verdict(r["dG"], r["gc"], r["eq"])
    sig = signatures(inp[rid]["species"]) if rid in inp else {}
    rows.append((rid, v, inp.get(rid, {}).get("note", ""), sig, r))

order = {"FAILURE": 0, "split": 1, "adj-EQ": 2, "adj-GC": 3, "ERROR": 4}
rows.sort(key=lambda x: (order.get(x[1], 9), x[0]))

print(f"{'rxn':10s} {'verdict':8s} {'UMA':>7s} {'gc':>7s} {'eq':>7s} {'O2':>3s} {'dQ':>3s} {'dRng':>4s} {'dAr':>3s} {'dBond':>5s}  EC / note")
counts = defaultdict(int)
for rid, v, note, sig, r in rows:
    counts[v] += 1
    if v == "ERROR":
        print(f"{rid:10s} {v:8s}  {note}")
        continue
    ec = ""
    if rid in inp:
        n = inp[rid]["note"]
        ec = n.split("EC=")[-1] if "EC=" in n else ""
    print(f"{rid:10s} {v:8s} {r['dG']:+7.1f} {r['gc']:+7.1f} {r['eq']:+7.1f} "
          f"{sig.get('n_O2',0):>3d} {sig.get('d_charge',0):>+3d} {sig.get('d_rings',0):>+4d} "
          f"{sig.get('d_arom_rings',0):>+3d} {sig.get('d_heavy_bonds',0):>+5d}  {ec:12s}")
print("\ncounts:", dict(counts))
