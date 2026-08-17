"""Per-reaction error table for the FULL pipeline (pH-0 + COFACTOR_RING + truncation).
Dumps rid, class, exp, uma_pred, signed error, and mechanistic features so we can look for
SYSTEMATIC patterns in the >20 kJ tail AND the 10-20 kJ intermediate band.
Reuses uma_dG / is_redox from make_error_histogram and rxn_class from ph0_final_analysis.
"""
import os, re, sys, json, importlib.util, collections
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors as rdMD

HERE = os.path.dirname(os.path.abspath(__file__))

def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

pfa = _load("pfa", "ph0_final_analysis.py")
meh = _load("meh", "make_error_histogram.py")
d = pfa.d

# --- mechanistic feature detectors (cheap SMARTS) ---------------------------------------------
PATS = {
    "phosphate":  Chem.MolFromSmarts("P(=O)([O,OH,O-])"),
    "thioester":  Chem.MolFromSmarts("[#6]C(=O)[#16X2]"),
    "flavin":     Chem.MolFromSmarts("c1cc2nc3c(=O)[nH]c(=O)nc3nc2cc1"),
    "coa_pant":   Chem.MolFromSmarts("SCCNC(=O)CCNC(=O)"),
    "nicotinam":  Chem.MolFromSmarts("[n,N]1cccc(c1)C(=O)[NX3]"),
    "disulfide":  Chem.MolFromSmarts("[#16X2][#16X2]"),
    "guanidine":  Chem.MolFromSmarts("[NX3][CX3](=[NX2])[NX3]"),
    "aldehyde":   Chem.MolFromSmarts("[CX3H1](=O)[#6]"),
    "carboxyl":   Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]"),
    "amine":      Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]"),
    "quinone":    Chem.MolFromSmarts("O=C1C=CC(=O)C=C1"),
    "sugar_ring": Chem.MolFromSmarts("[#6;R][OX2;R][#6;R]([OX2])"),
}

def feats(rid):
    sp = d[rid]["species"]
    out = set()
    tot_charge_species = 0
    max_anion = 0
    n_heavy_max = 0
    for c, q, s in sp.values():
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        n_heavy_max = max(n_heavy_max, m.GetNumHeavyAtoms())
        max_anion = max(max_anion, -q)
        for name, pat in PATS.items():
            if pat is not None and m.HasSubstructMatch(pat):
                out.add(name)
    return out, max_anion, n_heavy_max

def main():
    rows = []
    for rid in d:
        u = meh.uma_dG(rid)
        if u is None or abs(u) > 200: continue
        exp = d[rid]["exp"][0]
        err = u - exp
        cls = pfa.rxn_class(rid)
        f, max_anion, nheavy = feats(rid)
        note = d[rid]["note"]
        rows.append(dict(rid=rid, cls=cls, exp=exp, uma=u, err=err, aerr=abs(err),
                         feats=f, max_anion=max_anion, nheavy=nheavy, note=note,
                         nspecies=len(d[rid]["species"])))
    rows.sort(key=lambda r: -r["aerr"])
    n = len(rows)
    mae = np.mean([r["aerr"] for r in rows])
    med = np.median([r["aerr"] for r in rows])
    print(f"N={n}  MAE={mae:.1f}  median={med:.1f}\n")

    bands = [("TAIL >20", lambda a: a > 20),
             ("MID 10-20", lambda a: 10 <= a <= 20),
             ("GOOD <10", lambda a: a < 10)]
    for label, cond in bands:
        sub = [r for r in rows if cond(r["aerr"])]
        print(f"=== {label}: {len(sub)}/{n} ({100*len(sub)/n:.0f}%)  MAE={np.mean([r['aerr'] for r in sub]):.1f} ===")
        # class breakdown
        cc = collections.Counter(r["cls"] for r in sub)
        print("  by class:", dict(cc.most_common()))
        # feature enrichment vs overall
        ff = collections.Counter()
        for r in sub:
            for x in r["feats"]: ff[x]+=1
        allf = collections.Counter()
        for r in rows:
            for x in r["feats"]: allf[x]+=1
        enr = []
        for feat, cnt in ff.items():
            frac_band = cnt/len(sub)
            frac_all = allf[feat]/n
            enr.append((feat, cnt, frac_band, frac_band/max(frac_all,1e-9)))
        enr.sort(key=lambda x:-x[3])
        print("  feature enrichment (feat: n_in_band, %band, xEnrich):")
        for feat,cnt,fb,e in enr:
            print(f"     {feat:12s} {cnt:3d}  {100*fb:4.0f}%  {e:.1f}x")
        # signed bias
        signs = [r["err"] for r in sub]
        print(f"  signed bias: mean {np.mean(signs):+.1f}  (pos={sum(1 for s in signs if s>0)} neg={sum(1 for s in signs if s<0)})")
        print()

    print("=== TAIL reactions (>20), sorted ===")
    for r in [x for x in rows if x["aerr"]>20]:
        print(f"  {r['rid']} {r['cls']:10s} err {r['err']:+7.1f}  exp {r['exp']:+6.1f}  "
              f"anion{r['max_anion']} nH{r['nheavy']:3d} feats={sorted(r['feats'])}  | {r['note'][:45]}")
    print("\n=== MID reactions (10-20), sorted ===")
    for r in [x for x in rows if 10<=x["aerr"]<=20]:
        print(f"  {r['rid']} {r['cls']:10s} err {r['err']:+7.1f}  exp {r['exp']:+6.1f}  "
              f"anion{r['max_anion']} nH{r['nheavy']:3d} feats={sorted(r['feats'])}  | {r['note'][:45]}")

if __name__ == "__main__":
    main()
