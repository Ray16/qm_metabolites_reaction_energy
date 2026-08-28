"""Physics-based reframe of the 'Mg/NTP scatter': the error lives in the COUPLED bond-formation chemistry,
not the phosphate/Mg. QM's systematic error mode is BOND-TYPE reference error (sign-consistent per bond
type — proven for redox NAD & phosphagen P-N). So regroup the ATP-coupled reactions by the bond they FORM
(net functional-group change) and test whether each mechanism subclass has a COMMON, cancellable error
(leave-one-out isodesmic). A subclass with low LOO-MAE + clean bias = a real fixable bond-type class; pure
scatter = not. Zero-GPU."""
import os, re, sys, json
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "..")
d = json.load(open(os.path.join(EXP, "scripts", "reactions_tecrdb_all.json")))

# net functional-group change detectors (the COUPLED bond formed/broken), priority order
GROUPS = [
    ("thioester",        Chem.MolFromSmarts("[#6X3](=O)[SX2]")),
    ("acyl-phosphate",   Chem.MolFromSmarts("[#6X3](=O)[OX2][PX4]")),
    ("amide (C-N=O)",    Chem.MolFromSmarts("[CX3](=O)[NX3]")),
    ("guanidino/amidine",Chem.MolFromSmarts("[NX3][CX3]=[NX2]")),
    ("sulfate/adenylyl", Chem.MolFromSmarts("[SX4](=O)(=O)[OX2]")),
    ("phospho-ester(kinase)", Chem.MolFromSmarts("[#6][OX2][PX4](=O)([OX1])[OX1]")),
    ("carboxyl-CoA/acyl",Chem.MolFromSmarts("[#6X3](=O)[OX1]")),
]
POP = Chem.MolFromSmarts("[PX4]-O-[PX4]")


def uma_dG(rid):
    p = os.path.join(EXP, "logs", "production", f"{rid}.log")
    if not os.path.exists(p): return None
    m = re.search(r"ΔG = ([+-]?\d+\.\d+)", open(p, errors="ignore").read())
    return float(m.group(1)) if m else None


def net(rid, patt):
    n = 0
    for c, q, s in d[rid]["species"].values():
        m = Chem.MolFromSmiles(s)
        if m: n += c * len(m.GetSubstructMatches(patt))
    return n


def is_ntp(rid):
    return net(rid, POP) != 0 or "mg-prone" in d[rid].get("note", "").lower()


def coupled_class(rid):
    for name, patt in GROUPS:
        if net(rid, patt) != 0:
            return name
    return "phosphoryl/other"


def main():
    exp = {r: v["exp"][0] for r, v in d.items()}
    err = {}
    for rid in d:
        if not is_ntp(rid): continue
        u = uma_dG(rid)
        if u is None: continue
        e = u - exp[rid]
        if abs(e) <= 200: err[rid] = e
    cls = defaultdict(list)
    for rid in err:
        cls[coupled_class(rid)].append(rid)

    print(f"{'coupled-bond subclass':24s} {'n':>3s} {'rawMAE':>7s} {'LOO-iso':>8s} {'bias':>7s} {'residσ':>7s}  verdict")
    tot_raw = tot_iso = N = 0
    for sc, rids in sorted(cls.items(), key=lambda x: -len(x[1])):
        es = np.array([err[r] for r in rids])
        if len(es) < 2:
            print(f"{sc:24s} {len(es):3d} {np.mean(np.abs(es)):7.1f}   (n<2, need anchor)"); continue
        iso = np.array([es[i] - np.mean(np.delete(es, i)) for i in range(len(es))])
        verdict = "FIXABLE (common err)" if np.mean(np.abs(iso)) < np.mean(np.abs(es)) - 3 else "scatter"
        print(f"{sc:24s} {len(es):3d} {np.mean(np.abs(es)):7.1f} {np.mean(np.abs(iso)):8.1f} {np.mean(es):+7.1f} {np.std(iso):7.1f}  {verdict}")
        tot_raw += np.abs(es).sum(); tot_iso += np.abs(iso).sum(); N += len(es)
    print(f"\n  ATP-coupled combined: raw MAE {tot_raw/N:.1f} -> mechanism-referenced LOO {tot_iso/N:.1f}  (n={N})")
    # show the members of the cleanest fixable classes
    print("\n=== members of subclasses with a clean common bias ===")
    for sc, rids in cls.items():
        es = np.array([err[r] for r in rids])
        if len(es) >= 2 and abs(np.mean(es)) > 10 and np.std(es) < abs(np.mean(es)):
            print(f"  {sc}: bias {np.mean(es):+.1f}, n={len(rids)} -> " +
                  ", ".join(f"{r}({err[r]:+.0f})" for r in rids[:6]))


if __name__ == "__main__":
    main()
