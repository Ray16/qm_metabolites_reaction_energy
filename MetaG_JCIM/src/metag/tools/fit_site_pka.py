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
(effective sequential) pKa's. This approximates a macroscopic binding polynomial
when the ladder is well separated; it is not an exact coupled-state model.

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

from metag.routing.site_pka import CLASSES, EXTRA, site_features, effective_pkas


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
