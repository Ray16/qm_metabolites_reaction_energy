"""Independent test of the implicit solvation model used by MetaG: xtb --cosmo (current) vs xtb --alpb
vs experiment on FreeSolv (642 neutral molecules, experimental hydration free energies, Ben-Naim 1 M/1 M
convention). Same ΔG_solv definition as metag.energetics.thermal.xtb_dgsolv (E_solv - E_gas single points on
one MMFF geometry). Reports MAE/bias overall and on the polar, metabolite-like subset (>=2 O/N H-bond
donor/acceptor groups), which is where the reaction pipeline lives."""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from metag.energetics.thermal import xtb_dgsolv
KCAL = 4.184
rows = []
for line in open(sys.argv[1]):
    if line.startswith("#"):
        continue
    f = [x.strip() for x in line.split(";")]
    rows.append((f[0], f[1], float(f[3]) * KCAL))

def one(r):
    cid, smi, exp = r
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    if AllChem.EmbedMolecule(m, randomSeed=1) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    sym = [a.GetSymbol() for a in m.GetAtoms()]; xyz = m.GetConformer().GetPositions()
    c = xtb_dgsolv(sym, xyz, 0, "cosmo"); a = xtb_dgsolv(sym, xyz, 0, "alpb")
    if c is None or a is None:
        return None
    mh = Chem.MolFromSmiles(smi)
    polar = rdMolDescriptors.CalcNumHBD(mh) + rdMolDescriptors.CalcNumHBA(mh)
    return {"id": cid, "smi": smi, "exp": exp, "cosmo": c, "alpb": a, "polar": polar}

with ThreadPoolExecutor(max_workers=16) as ex:
    res = [x for x in ex.map(one, rows) if x]
json.dump(res, open(os.path.join(os.path.dirname(__file__), "freesolv_solv_models.json"), "w"))
for lab, sub in (("all", res), ("polar>=3 (metabolite-like)", [r for r in res if r["polar"] >= 3])):
    e = np.array([r["exp"] for r in sub])
    for mdl in ("cosmo", "alpb"):
        p = np.array([r[mdl] for r in sub]); d = p - e
        slope = np.polyfit(e, p, 1)[0]
        print(f"{lab:28s} n={len(sub):3d} {mdl:5s}: MAE {np.abs(d).mean():5.1f} kJ  bias {d.mean():+6.1f}  "
              f"R {np.corrcoef(e, p)[0,1]:.3f}  slope {slope:.2f}")
