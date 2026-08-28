"""Isodesmic reference-reaction test (zero-GPU). The wall error is a large phosphate-solvation term QM/COSMO
gets wrong. Referencing a target to a MEASURED reaction in the SAME solvation subclass cancels it:
target - reference = a well-conditioned ester<->ester (or anhydride<->anhydride) transfer where the free
anion + H2O cancel and the phosphate stays esterified on both sides. So the isodesmic prediction error is

    err_isodesmic(target | anchor) = err_target - err_anchor

and if the subclass shares a common wall error, LEAVE-ONE-OUT referencing collapses the MAE. The reference
must be CHEMICALLY MATCHED (monoester->monoester etc.) or the errors don't correlate. We test which
subclass groupings actually cancel, using the current pipeline per-reaction errors.
"""
import os, re, sys, json
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(EXP, "scripts"))
import ph0_auto as pfa
import aldehyde_hydration as ah
d = json.load(open(os.path.join(EXP, "scripts", "reactions_tecrdb_all.json")))


def uma_dG(rid):
    p = os.path.join(EXP, "logs", "production", f"{rid}.log")
    if not os.path.exists(p): return None
    m = re.search(r"ΔG = ([+-]?\d+\.\d+)", open(p, errors="ignore").read())
    return float(m.group(1)) if m else None


# SMARTS for solvation subclasses (matched phosphate environment)
PN = Chem.MolFromSmarts("[#7]-[PX4](=O)")             # phosphoramidate (phosphagen)
POP = Chem.MolFromSmarts("[PX4]-O-[PX4]")             # anhydride / NTP / PPi
MONO = Chem.MolFromSmarts("[#6]-[OX2]-[PX4](=O)([OX1])[OX1]")  # phosphomonoester
SULF = Chem.MolFromSmarts("[SX4](=O)(=O)[OX1]")


def _has(rid, patt):
    for c, q, s in d[rid]["species"].values():
        m = Chem.MolFromSmiles(s)
        if m and m.HasSubstructMatch(patt): return True
    return False


def _net(rid, patt):
    n = 0
    for c, q, s in d[rid]["species"].values():
        m = Chem.MolFromSmiles(s)
        if m: n += c * len(m.GetSubstructMatches(patt))
    return n


def subclass(rid):
    if _net(rid, PN) != 0: return "phosphagen (P-N)"
    if _net(rid, POP) != 0: return "anhydride/NTP/PPi"
    if _net(rid, MONO) != 0: return "phosphomonoester"
    if _net(rid, SULF) != 0: return "sulfo/adenylyl"
    return None


def main():
    exp = {r: v["exp"][0] for r, v in d.items()}
    err = {}
    for rid in d:
        u = uma_dG(rid)
        if u is None: continue
        e = u - exp[rid]
        if abs(e) > 200: continue
        err[rid] = e
    cls = defaultdict(list)
    for rid in err:
        sc = subclass(rid)
        if sc: cls[sc].append(rid)

    print(f"{'subclass':22s} {'n':>3s} {'raw MAE':>8s} {'iso MAE (LOO)':>13s} {'raw bias':>9s} {'iso resid σ':>11s}")
    tot_raw, tot_iso, N = 0.0, 0.0, 0
    for sc, rids in sorted(cls.items(), key=lambda x: -len(x[1])):
        if len(rids) < 2: continue
        es = np.array([err[r] for r in rids])
        raw_mae = np.mean(np.abs(es))
        # leave-one-out isodesmic: predict each from the MEAN of the OTHERS in the subclass (matched anchor)
        iso = []
        for i in range(len(es)):
            anchor = np.mean(np.delete(es, i))            # common wall error estimated from the rest
            iso.append(es[i] - anchor)
        iso = np.array(iso)
        print(f"{sc:22s} {len(rids):3d} {raw_mae:8.1f} {np.mean(np.abs(iso)):13.1f} {np.mean(es):+9.1f} {np.std(iso):11.1f}")
        tot_raw += np.abs(es).sum(); tot_iso += np.abs(iso).sum(); N += len(es)
    print(f"\n  wall subclasses combined: raw MAE {tot_raw/N:.1f} -> isodesmic LOO MAE {tot_iso/N:.1f}  (n={N})")
    print("  (iso MAE << raw MAE means the subclass shares a common wall error that referencing cancels;")
    print("   'iso resid σ' is the irreducible scatter after the matched anchor removes the common term.)")


if __name__ == "__main__":
    main()
