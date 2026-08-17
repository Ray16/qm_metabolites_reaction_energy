"""Final coherent-pipeline accuracy report: GATED pH-0 vs baseline vs TECRDB.

Gate (validated, general, structural — no reaction-id logic): apply the pH-0 result UNLESS the
reaction is an ISOMERIZATION (every reactant molecular-formula has a matching product formula, i.e.
a rearrangement with conserved anionic groups — pH-0 has no solvation change to fix there and only
adds neutral-vs-anion sampling noise). Otherwise use the pH-0 result. This is exactly what the
in-code gate will do, so this report equals the automatic pipeline's accuracy.

Reads logs/full367 (baseline) + logs/ph0_sweep (pH-0). Drops |err|>200 garbage (loader-collision /
species-failure) with a count. No GPU."""
import re, glob, json, os, math, collections
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD

HERE = os.path.dirname(__file__)
d = json.load(open(os.path.join(HERE, "..", "scripts", "reactions_tecrdb_all.json")))


def _err(path):
    try:
        t = open(path).read()
    except FileNotFoundError:
        return None, None
    m = re.search(r"ΔG = ([+-]?\d+\.\d+).*?err \[([^\]]+)\]", t)
    if not m:
        return None, t
    return float(m.group(2).split(",")[0]), t


def is_isomerization(sp):
    def formula(s):
        mol = Chem.MolFromSmiles(s)
        return rdMD.CalcMolFormula(mol) if mol else None
    R = [formula(s) for c, q, s in sp.values() if c < 0 for _ in range(abs(c))]
    P = [formula(s) for c, q, s in sp.values() if c > 0 for _ in range(abs(c))]
    if None in R or None in P:
        return False
    return sorted(R) == sorted(P)


def rxn_class(rid):
    r = d[rid]; note = r["note"]; smis = " ".join(s[2] for s in r["species"].values())
    if "C(=O)S" in smis or "SC(=O)" in smis:
        return "thioester"
    if is_isomerization(r["species"]) or "isomerase" in note:
        return "isomerase"
    if "huge/floppy" in note:
        return "huge/floppy"
    if "Mg-prone" in note or "anion-count" in note:
        return "anion"
    return "clean"


def main():
    base, ph0 = {}, {}
    garbage, missing = [], []
    for rid in d:
        be, _ = _err(os.path.join(HERE, "..", "logs", "full367", f"{rid}.log"))
        pe, pt = _err(os.path.join(HERE, "..", "logs", "ph0_sweep", f"{rid}.log"))
        if be is not None and abs(be) <= 200:
            base[rid] = be
        if pe is not None and abs(pe) <= 200:
            ph0[rid] = pe
        elif pe is not None:
            garbage.append(rid)
        if pe is None:
            missing.append(rid)

    # GATED coherent: pH-0 unless isomerization (then baseline)
    coh = {}
    for rid in base:
        iso = is_isomerization(d[rid]["species"])
        if iso or rid not in ph0:
            coh[rid] = base[rid]
        else:
            coh[rid] = ph0[rid]

    def mae(x): return sum(abs(v) for v in x) / len(x) if x else float("nan")
    common = [rid for rid in base if rid in ph0]
    print(f"reactions: baseline {len(base)}, pH-0 done {len(ph0)}, both {len(common)} "
          f"| pH-0 garbage {len(garbage)} | pH-0 not-done {len(missing)}")
    print(f"\nMAE baseline   : {mae([base[r] for r in common]):5.1f} kJ")
    print(f"MAE pH-0 (all) : {mae([ph0[r] for r in common]):5.1f} kJ  (blind, incl. isomerase-hurts)")
    print(f"MAE GATED coh. : {mae([coh[r] for r in common]):5.1f} kJ  <== automatic pipeline accuracy")
    within = sum(1 for r in common if (d[r].get('exp_sd') or 0) > 0 and abs(coh[r]) <= d[r]['exp_sd'])
    tight = sum(1 for r in common if abs(coh[r]) < 10)
    print(f"  |err|<10: {tight}/{len(common)} ({100*tight/len(common):.0f}%)  within-exp-sd: {within}")

    print(f"\n{'class':12s} {'n':>3s} {'base':>6s} {'pH0':>6s} {'GATED':>6s}")
    byc = collections.defaultdict(list)
    for rid in common:
        byc[rxn_class(rid)].append(rid)
    for c, rids in sorted(byc.items(), key=lambda x: -len(x[1])):
        print(f"{c:12s} {len(rids):3d} {mae([base[r] for r in rids]):6.1f} "
              f"{mae([ph0[r] for r in rids]):6.1f} {mae([coh[r] for r in rids]):6.1f}")
    if garbage:
        print(f"\ngarbage (needs re-run): {garbage}")
    if missing:
        print(f"pH-0 not done ({len(missing)}): {missing[:12]}{'...' if len(missing)>12 else ''}")


if __name__ == "__main__":
    main()
