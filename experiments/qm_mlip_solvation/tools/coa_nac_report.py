"""CoA class: production (no NAC) vs NAC-truncated ΔG. Reads logs/coa_nac_val/<rid>.log."""
import json, os, re, glob
import numpy as np

QM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = json.load(open(os.path.join(QM, "..", "gnn_dgf", "artifacts", "per_rxn_benchmark.json")))
RXN = json.load(open(os.path.join(QM, "scripts", "reactions_tecrdb_all.json")))


def new_dG(rid):
    f = os.path.join(QM, "logs", "coa_nac_val", f"{rid}.log")
    if not os.path.exists(f):
        return None, False
    txt = open(f, errors="replace").read()
    m = re.findall(r"ΔG = ([+-]?\d+\.?\d*)", txt)
    fired = "NAC" in txt or "CC(=O)NCCS" in txt or "nicotinamide" in txt
    return (float(m[-1]) if m else None), fired


rows = []
for rid in B:
    if B[rid]["class"] != "CoA-thioester" or B[rid].get("uma_qm_err") is None:
        continue
    exp = RXN[rid]["exp"][0]
    prod_err = B[rid]["uma_qm_err"]
    nd, fired = new_dG(rid)
    new_err = (nd - exp) if nd is not None else None
    rows.append(dict(rid=rid, exp=exp, prod_err=prod_err, new_dG=nd, new_err=new_err, fired=fired))

rows.sort(key=lambda r: -abs(r["prod_err"]))
print(f"{'rid':<10}{'exp':>7}{'prod_err':>9}{'newΔG':>8}{'new_err':>9}  fired?")
print("-" * 52)
pe, ne = [], []
for r in rows:
    if r["new_err"] is None:
        print(f"{r['rid']:<10}{r['exp']:>7.1f}{r['prod_err']:>9.1f}   (no result)"); continue
    print(f"{r['rid']:<10}{r['exp']:>7.1f}{r['prod_err']:>9.1f}{r['new_dG']:>8.1f}{r['new_err']:>9.1f}  {'NAC' if r['fired'] else '-'}")
    pe.append(abs(r["prod_err"])); ne.append(abs(r["new_err"]))
if pe:
    print(f"\nCoA class (n={len(pe)}):  MAE production {np.mean(pe):.1f}  ->  NAC-truncated {np.mean(ne):.1f}")
    print(f"  improved/worsened/same: {sum(1 for a,b in zip(pe,ne) if b<a-2)}/"
          f"{sum(1 for a,b in zip(pe,ne) if b>a+2)}/{sum(1 for a,b in zip(pe,ne) if abs(a-b)<=2)}")
    print(f"  |err|>20:  production {sum(1 for a in pe if a>20)}  ->  NAC {sum(1 for a in ne if a>20)}")
