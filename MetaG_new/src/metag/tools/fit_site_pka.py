"""Fit the generic per-site acid pKa model used by metag.routing.pka_transform (SITE_PKA model) to an
INDEPENDENT pKa reference -- never to reaction ΔG.

Physics (textbook, two effects):
  1. inductive / resonance increment by site class: pKa_intr = base[class] (carboxyl alpha-environment:
     plain, alpha-oxo, alpha-oxygen, alpha-O-phosphate/sulfate, vinyl-conjugated, aromatic, formate, plus a
     beta-oxo/beta-oxygen term; phosphate: free Pi ladder, monoester, anhydride-terminal, internal; PPi);
  2. electrostatic coupling between anionic sites: deprotonating sites in order of increasing intrinsic
     pKa, site j is raised by  c / d_ij  for every already-deprotonated site i (d = topological distance
     in bonds between the two acidic heavy atoms).
The per-species transform used by the pipeline is the independent-site product over these effective
(microscopic, sequential) pKa's, which equals the exact macroscopic polynomial for a well-separated ladder.

Reference: ChemAxon macroscopic pKa's (eQuilibrator cache) for ModelSEED compounds with only acidic sites
(no basic N) whose ladder length equals the pipeline's acid-site count (unambiguous site assignment).
Fitted by least squares on sorted ladders, 5-fold CV by compound. Output: metag/data/site_pka_model.json.

    python -m metag.tools.fit_site_pka analysis/review_fixes/pka_cands_cx.json
"""
import hashlib
import json
import math
import os
import sys

import numpy as np
from rdkit import Chem

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_PATH = os.path.join(_DATA, "site_pka_model.json")
PH = 7.0
RTLN10 = 2.303 * 8.314e-3 * 298.15

CLASSES = ["carb_ammonium", "carb_amine", "carb_plain", "carb_oxo", "carb_oxygen", "carb_Oacid", "carb_vinyl", "carb_arom", "carb_formate",
           "Pi_1", "Pi_2", "Pi_3", "mono_1", "mono_2", "anh_1", "anh_2", "internal", "ppi_1", "ppi_2", "ppi_3",
           "ppi_4", "sulf", "sulfate_1", "sulfate_2"]
EXTRA = ["beta_oxo", "beta_oxygen", "coulomb_c"]


def site_features(mol, sites):
    """sites = [(o_idx, cls_group)] from pka_transform._anion_sites. Returns list of dicts:
    {"atom": acidic heavy atom idx (C of carboxyl / P / S), "cls": class name, "beta_oxo": 0/1,
     "beta_oxygen": 0/1}. Phosphate/sulfate O's on the same centre get ladder positions 1..k."""
    out = []
    groups = {}
    for o, grp in sites:
        oa = mol.GetAtomWithIdx(o)
        centre = next((n for n in oa.GetNeighbors() if n.GetSymbol() in ("C", "P", "S")), None)
        if centre is None:
            continue
        groups.setdefault((grp, centre.GetIdx()), []).append(o)
    zs = [a.GetAtomicNum() for a in mol.GetAtoms()]
    free_ppi = zs.count(15) == 2 and all(z in (8, 15) for z in zs)
    ppi_count = 0
    for (grp, c), os_ in sorted(groups.items(), key=lambda kv: kv[0][1]):
        ca = mol.GetAtomWithIdx(c)
        if grp == "carboxyl":
            for o in os_:
                cls, bo, box = _carboxyl_class(mol, ca)
                out.append({"o": o, "atom": c, "cls": cls, "beta_oxo": bo, "beta_oxygen": box})
        elif grp == "sulfonate":
            out += [{"o": o, "atom": c, "cls": "sulf", "beta_oxo": 0, "beta_oxygen": 0} for o in os_]
        elif grp == "sulfate":
            out += [{"o": o, "atom": c, "cls": f"sulfate_{min(k + 1, 2)}", "beta_oxo": 0, "beta_oxygen": 0}
                    for k, o in enumerate(sorted(os_))]
        else:                                                      # phosphate centre
            n_bridge = sum(1 for n in ca.GetNeighbors() if n.GetSymbol() == "O" and n.GetDegree() >= 2)
            to_p = any(n2.GetSymbol() == "P" and n2.GetIdx() != c
                       for n in ca.GetNeighbors() if n.GetSymbol() == "O" for n2 in n.GetNeighbors())
            for k, o in enumerate(sorted(os_)):
                if free_ppi:
                    ppi_count += 1; cls = f"ppi_{min(ppi_count, 4)}"
                elif n_bridge == 0:
                    cls = f"Pi_{min(k + 1, 3)}"
                elif n_bridge == 1:
                    cls = (f"anh_{min(k + 1, 2)}" if to_p else f"mono_{min(k + 1, 2)}")
                else:
                    cls = "internal"
                out.append({"o": o, "atom": c, "cls": cls, "beta_oxo": 0, "beta_oxygen": 0})
    return out


def _carboxyl_class(mol, c):
    alpha = [n for n in c.GetNeighbors() if n.GetSymbol() == "C"]
    if not alpha:
        return "carb_formate", 0, 0
    a = alpha[0]
    cls = "carb_plain"
    if a.GetHybridization() == Chem.HybridizationType.SP3:
        for nb in a.GetNeighbors():                           # alpha-N: amino acid (textbook, fixed classes)
            if nb.GetSymbol() != "N":
                continue
            if nb.GetFormalCharge() == 1 and nb.GetTotalNumHs() >= 1:
                return "carb_ammonium", 0, 0                  # N stays protonated in the QM species
            amide = any(x.GetSymbol() == "C" and any(
                mol.GetBondBetweenAtoms(x.GetIdx(), y.GetIdx()).GetBondTypeAsDouble() == 2 and y.GetSymbol() in "ONS"
                for y in x.GetNeighbors()) for x in nb.GetNeighbors() if x.GetIdx() != a.GetIdx())
            if nb.GetFormalCharge() == 0 and nb.GetTotalNumHs() >= 1 and not nb.GetIsAromatic() and not amide:
                return "carb_amine", 0, 0                     # neutral NH2 in the QM species: microscopic pKa
    if a.GetIsAromatic():
        cls = "carb_arom"
    elif any(nb.GetSymbol() == "O" and any(x.GetSymbol() in ("P", "S") for x in nb.GetNeighbors())
             for nb in a.GetNeighbors()):
        cls = "carb_Oacid"                                   # alpha-O-phosphate/sulfate (2-PG, PEP): 3.5
    else:
        for nb in a.GetNeighbors():
            if nb.GetIdx() == c.GetIdx():
                continue
            b = mol.GetBondBetweenAtoms(a.GetIdx(), nb.GetIdx()).GetBondTypeAsDouble()
            if nb.GetSymbol() == "O" and b == 2:
                cls = "carb_oxo"; break
            if nb.GetSymbol() == "C" and b == 2:
                cls = "carb_vinyl"
        if cls == "carb_plain":
            for nb in a.GetNeighbors():
                if nb.GetSymbol() == "O" and nb.GetIdx() != c.GetIdx():
                    acid_ester = any(x.GetSymbol() in ("P", "S") for x in nb.GetNeighbors())
                    cls = "carb_Oacid" if acid_ester else "carb_oxygen"
                    if acid_ester:
                        break
    bo = box = 0
    if cls in ("carb_plain", "carb_oxygen", "carb_Oacid"):
        for nb in a.GetNeighbors():
            if nb.GetIdx() == c.GetIdx() or nb.GetSymbol() != "C":
                continue
            for x in nb.GetNeighbors():
                if x.GetIdx() == a.GetIdx():
                    continue
                bt = mol.GetBondBetweenAtoms(nb.GetIdx(), x.GetIdx()).GetBondTypeAsDouble()
                if x.GetSymbol() == "O" and bt == 2 and not any(
                        y.GetSymbol() == "O" and y.GetIdx() != x.GetIdx() and y.GetFormalCharge() < 0
                        for y in nb.GetNeighbors()):
                    bo = 1                                          # beta ketone/aldehyde (not carboxylate)
                elif x.GetSymbol() == "O" and bt == 1:
                    box = 1
    return cls, bo, box


def effective_pkas(mol, feats, params):
    """Sequential microscopic pKa's: intrinsic = base[cls] + beta terms; deprotonate in increasing order,
    each raised by coulomb_c / d for every already-deprotonated site."""
    if not feats:
        return []
    dm = Chem.GetDistanceMatrix(mol)
    intr = [params[f["cls"]] + params["beta_oxo"] * f["beta_oxo"] + params["beta_oxygen"] * f["beta_oxygen"]
            for f in feats]
    order = np.argsort(intr)
    done, eff = [], [0.0] * len(feats)
    for j in order:
        # coupling only BETWEEN CARBOXYLATES (what it is calibrated on). Experimental phosphate / PPi /
        # sulfate ladders are macroscopic values that already contain their intra-molecular repulsion
        # (ATP 7.60, PPi 0.9/2.1/6.7/9.3), so adding it there would double-count.
        if not feats[j]["cls"].startswith("carb_"):
            eff[j] = intr[j]; done.append(j); continue
        shift = sum(params["coulomb_c"] / max(dm[feats[j]["atom"]][feats[i]["atom"]], 1.0)
                    for i in done if feats[i]["cls"].startswith("carb_"))
        eff[j] = intr[j] + shift
        done.append(j)
    return eff


def transform(pkas):
    """-RT ln P from the neutral acid to pH 7 for independent sites (kJ)."""
    return -RTLN10 * sum(math.log10(1.0 + 10.0 ** (PH - p)) for p in pkas)


def ref_transform(pkas):
    s = sorted(pkas); logs = [0.0]; acc = 0.0
    for p in s:
        acc += PH - p; logs.append(acc)
    m = max(logs)
    return -RTLN10 * (m + math.log10(sum(10 ** (x - m) for x in logs)))


def _prep(path):
    import metag.routing.pka_transform as T
    data = json.load(open(path))
    rows = []
    for cid, d in data.items():
        ref = sorted(p for p in d["pkas"] if p < 14)
        if len(ref) != d["n"]:
            continue
        mol = Chem.MolFromSmiles(d["smi"])
        if mol is None:
            continue
        feats = site_features(mol, T._anion_sites(mol))
        if len(feats) != len(ref):
            continue
        rows.append((cid, mol, feats, ref))
    return rows


def _default_params():
    p = {c: 4.0 for c in CLASSES}
    p.update({"carb_ammonium": 2.3, "carb_amine": 4.4, "carb_Oacid": 3.5, "carb_plain": 4.75, "Pi_1": 2.15, "Pi_2": 7.2, "Pi_3": 12.35, "mono_1": 1.5, "mono_2": 6.5,
              "anh_1": 1.0, "anh_2": 7.2, "internal": 1.0, "ppi_1": 0.9, "ppi_2": 2.1, "ppi_3": 6.7,
              "ppi_4": 9.3, "sulf": -1.5, "sulfate_1": -3.0, "sulfate_2": 2.0,
              "beta_oxo": 0.0, "beta_oxygen": 0.0, "coulomb_c": 0.0})
    return p


FIXED = {"carb_ammonium", "carb_amine", "carb_Oacid", "Pi_1", "Pi_2", "Pi_3", "ppi_1", "ppi_2", "ppi_3", "ppi_4", "sulf", "sulfate_1", "sulfate_2",
         "mono_1", "mono_2", "anh_1", "anh_2", "internal"}
# ^ experimental values (I->0), NOT refit: ChemAxon is systematically low on phosphate pKa2 (fits 5.7 vs
#   Alberty I=0: G6P 6.42, G1P 6.50, F6P 6.27, glycerol-3-P 6.67, AMP 6.73) and weak on P2O7/SO4; phosphate
#   monoester/anhydride sites straddle pH 7 and dominate metabolic transforms, so they stay experimental:
#   monoester 1.5/6.5; anhydride-terminal 1.0/7.2 (ADP 7.18, ATP 7.60 at I=0); internal diester 1.0.
#   Carboxyl classes with no phosphate-free training data are textbook-fixed: alpha-O-phosphate 3.5
#   (2-phosphoglycerate 3.42, PEP 3.5); alpha-ammonium 2.3 (Gly/Ala pKa1); alpha-amine neutral in the QM
#   species 4.4 (glycine microconstant pk(COOH|NH2) = 9.78 - log K_taut 5.35).


def fit(rows, init=None):
    from scipy.optimize import least_squares
    p0 = dict(init or _default_params())
    free = [k for k in CLASSES + EXTRA if k not in FIXED
            and (k in EXTRA or any(f["cls"] == k for _, _, fs, _ in rows for f in fs))]

    def resid(x):
        p = dict(p0); p.update(dict(zip(free, x)))
        r = []
        for _, mol, feats, ref in rows:
            eff = sorted(effective_pkas(mol, feats, p))
            r += [min(e, 12.0) - min(t, 12.0) for e, t in zip(eff, ref)]
        return np.array(r)

    lo = [0.0 if k == "coulomb_c" else -np.inf for k in free]   # physics: repulsion only raises pKa
    x0 = [max(p0[k], 0.0) if k == "coulomb_c" else p0[k] for k in free]
    sol = least_squares(resid, x0, loss="soft_l1", f_scale=0.5, bounds=(lo, [np.inf] * len(free)))
    p = dict(p0); p.update({k: round(float(v), 3) for k, v in zip(free, sol.x)})
    return p


def evaluate(rows, p):
    e = [transform(effective_pkas(mol, feats, p)) - ref_transform(ref) for _, mol, feats, ref in rows]
    e = np.array(e)
    return {"n": len(e), "MAE_kJ": round(float(np.abs(e).mean()), 2), "bias_kJ": round(float(e.mean()), 2),
            "frac_gt5kJ": round(float((np.abs(e) > 5).mean()), 3)}


def main(path):
    rows_all = _prep(path)
    # FIT only on carboxylate-only compounds (ChemAxon reliable there); phosphate/sulfate classes are fixed
    # experimental values. Evaluate on everything.
    rows = [r for r in rows_all if all(f["cls"].startswith("carb_") for f in r[2])]
    fold = [int(hashlib.md5(cid.encode()).hexdigest(), 16) % 5 for cid, *_ in rows]
    cv = []
    for f in range(5):
        tr = [r for r, k in zip(rows, fold) if k != f]
        te = [r for r, k in zip(rows, fold) if k == f]
        p = fit(tr)
        cv.append(evaluate(te, p))
    p_all = fit(rows)
    res = {"params": p_all, "n_compounds": len(rows),
           "cv_heldout": {"MAE_kJ": round(float(np.mean([c["MAE_kJ"] for c in cv])), 2),
                          "bias_kJ": round(float(np.mean([c["bias_kJ"] for c in cv])), 2),
                          "frac_gt5kJ": round(float(np.mean([c["frac_gt5kJ"] for c in cv])), 3)},
           "in_sample": evaluate(rows, p_all),
           "all_acid_compounds": {"model": evaluate(rows_all, p_all),
                                  "baseline_default_params": evaluate(rows_all, _default_params())},
           "reference": "ChemAxon macroscopic pKa (eQuilibrator cache), ModelSEED acid-only compounds, "
                        "unambiguous ladders; fixed experimental Pi/PPi/sulfate ladders",
           "baseline_default_params": evaluate(rows, _default_params())}
    json.dump(res, open(MODEL_PATH, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "params"}, indent=1))
    print(json.dumps(p_all))


if __name__ == "__main__":
    main(sys.argv[1])
