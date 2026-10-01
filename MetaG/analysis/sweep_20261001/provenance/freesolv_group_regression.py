"""Per-functional-group error of the implicit solvation models on FreeSolv (642 neutral molecules).
Least squares of (ΔG_solv,model - ΔG_exp) on SMARTS group counts + a constant; input = freesolv_solv_models.json
(produced by freesolv_solv_models.py from data/FreeSolv-master/database.txt, MMFF geometries, xtb 6.7.1)."""
import json, os
import numpy as np
from rdkit import Chem
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "freesolv_solv_models.json")))
G = {"alcoholOH": "[OX2H][CX4]", "phenolOH": "[OX2H]c", "COOH": "C(=O)[OX2H]", "amine": "[NX3;H2,H1;!$(NC=O);!$(Na)]",
     "anilineN": "[NX3;H2,H1]a", "amide": "C(=O)[NX3]", "ketone": "[CX3](=O)([#6])[#6]", "aldehyde": "[CX3H1](=O)",
     "ester": "C(=O)O[#6]", "ether": "[OD2]([#6])[#6]", "alkeneCC": "[CX3]=[CX3]", "arom_ring": "a1aaaaa1",
     "halogen": "[F,Cl,Br,I]", "nitro": "[N+](=O)[O-]", "nitrile": "C#N", "S": "[#16]", "P": "[#15]", "aromN": "n"}
P = {k: Chem.MolFromSmarts(v) for k, v in G.items()}
rows = [r for r in d if Chem.MolFromSmiles(r["smi"]) is not None and r.get("cpcmx") is not None]
X = np.array([[len(Chem.MolFromSmiles(r["smi"]).GetSubstructMatches(P[k])) for k in G] + [1] for r in rows], float)
out = {"n": len(rows), "groups": list(G) + ["const"]}
for m in ("cosmo", "alpb", "cpcmx"):
    y = np.array([r[m] - r["exp"] for r in rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    out[m] = {"MAE": round(float(np.abs(y).mean()), 2), "MAE_after_group_fit": round(float(np.abs(y - X @ coef).mean()), 2),
              "per_group_kJ": {k: round(float(c), 2) for k, c in zip(out["groups"], coef)},
              "n_with_group": {k: int((X[:, i] > 0).sum()) for i, k in enumerate(out["groups"])}}
json.dump(out, open(os.path.join(HERE, "freesolv_group_regression.json"), "w"), indent=1)
for m in ("cosmo", "alpb", "cpcmx"):
    print(m, out[m]["MAE"], {k: out[m]["per_group_kJ"][k] for k in ("alcoholOH", "COOH", "amine")})
