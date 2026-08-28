"""Is the Mg/NTP 'scatter' actually the Mg2+-binding equilibrium (a KNOWN physical quantity, not irreducible)?
QM scores Mg-FREE anions; TECRDB values were measured in Mg buffers. If Mg-binding is folded into the
experiment, the error should track each reaction's Mg-binding CHANGE, computable from tabulated binding
constants (Alberty). Test: (1) correlation of error with Δ(Mg-binding transform); (2) does applying the
physics correction ΔG_corr = ΔG_QM + Δ(transform) collapse the Mg/NTP MAE? Zero-GPU."""
import os, re, sys, json
import numpy as np
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(EXP, "scripts"))
d = json.load(open(os.path.join(EXP, "scripts", "reactions_tecrdb_all.json")))
RT = 2.478
MGFREE = float(os.environ.get("MG_MM", "1.0")) * 1e-3         # free [Mg2+], default 1 mM

# tabulated Mg2+-binding log K (Alberty, I~0.1, pH7) for the metabolite classes that bind Mg
LOGK = {"ATP": 4.0, "GTP": 4.0, "UTP": 4.0, "CTP": 4.0, "ITP": 4.0, "dATP": 4.0,
        "ADP": 2.9, "GDP": 2.9, "UDP": 2.9, "CDP": 2.9, "IDP": 2.9,
        "AMP": 1.9, "GMP": 1.9, "UMP": 1.9, "CMP": 1.9, "IMP": 1.9,
        "PPi": 5.5, "Pyrophosphate": 5.5, "Diphosphate": 5.5,
        "Phosphate": 1.9, "PRPP": 4.0, "APS": 3.0, "3-Phosphoadenylyl-sulfate": 3.5}
POP = Chem.MolFromSmarts("[PX4]-O-[PX4]")


def uma_dG(rid):
    p = os.path.join(EXP, "logs", "production", f"{rid}.log")
    if not os.path.exists(p): return None
    m = re.search(r"ΔG = ([+-]?\d+\.\d+)", open(p, errors="ignore").read())
    return float(m.group(1)) if m else None


def logk_for(name):
    if name in LOGK:
        return LOGK[name]
    # fuzzy: match on suffix tokens
    for key in LOGK:
        if key.lower() in name.lower():
            return LOGK[key]
    return None


def mg_transform(rid):
    """Δ(Mg-binding transform) = Σ_prod (-RT ln(1+K[Mg])) - Σ_react ... . coeff sign gives side."""
    dG = 0.0; hit = False
    for name, (c, q, s) in d[rid]["species"].items():
        lk = logk_for(name)
        if lk is None:
            continue
        hit = True
        P = 1.0 + 10 ** lk * MGFREE
        dG += c * (-RT * np.log(P))
    return dG if hit else None


def is_ntp(rid):
    for c, q, s in d[rid]["species"].values():
        m = Chem.MolFromSmiles(s)
        if m and m.HasSubstructMatch(POP): return True
    return "mg-prone" in d[rid].get("note", "").lower()


def main():
    exp = {r: v["exp"][0] for r, v in d.items()}
    rows = []
    for rid in d:
        if not is_ntp(rid): continue
        u = uma_dG(rid)
        if u is None: continue
        e = u - exp[rid]
        if abs(e) > 200: continue
        mt = mg_transform(rid)
        if mt is None: continue
        rows.append((rid, e, mt))
    es = np.array([r[1] for r in rows]); mt = np.array([r[2] for r in rows])
    # hypothesis: error ~ -Δtransform (exp has Mg folded in, QM is Mg-free) -> corrected = e + Δtransform
    corr = es + mt
    print(f"Mg/NTP reactions with binding data: {len(rows)}   [Mg2+]={MGFREE*1e3:.1f} mM")
    print(f"  corr(error, -Δtransform) = {np.corrcoef(es, -mt)[0,1]:+.2f}")
    print(f"  raw MAE {np.mean(np.abs(es)):.1f}  bias {np.mean(es):+.1f}")
    print(f"  Mg-corrected MAE {np.mean(np.abs(corr)):.1f}  bias {np.mean(corr):+.1f}")
    print(f"\n{'rxn':10s} {'err':>7s} {'Δtransform':>10s} {'corrected':>10s}  note")
    for rid, e, m in sorted(rows, key=lambda x: -abs(x[1]))[:15]:
        print(f"{rid:10s} {e:+7.1f} {m:+10.1f} {e+m:+10.1f}  {d[rid].get('note','')[7:44]}")
    print("\nIf correlation is strong (>0.4) and corrected MAE << raw, the Mg/NTP scatter is largely the"
          " Mg-binding equilibrium — a PHYSICS-based fixable term (tabulated constants), not irreducible.")


if __name__ == "__main__":
    main()
