"""Decompose the SOURCE of each reaction's error using the structures + QM logs we already have.
For each reaction (routed exactly like the reported number: ringcofactor/ph0_sweep/full367):
  - sampling:  max per-species conformer sigma + #CAPPED species  (undersampling indicator)
  - balance:   mass(heavy-atom formula) + charge balance of the STORED reaction (imbalance indicator)
  - floppy:    max rotatable bonds / heavy atoms among species    (floppiness indicator)
  - species Q: the scored charges from the log (catches NH4+ vs NH3 speciation)
Then bucket each reaction's dominant error source and tabulate against |err|.
"""
import os, re, importlib.util, collections
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD, Descriptors
def L(n,f):
    s=importlib.util.spec_from_file_location(n,os.path.join("tools",f)); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
pfa=L("pfa","ph0_final_analysis.py"); meh=L("meh","make_error_histogram.py"); d=pfa.d

LOGDIR="logs"
def routed_log(rid):
    """Return the log path that produced the REPORTED uma_dG (mirror make_error_histogram.uma_dG)."""
    if meh.is_redox(rid):
        p=os.path.join(LOGDIR,"ringcofactor",f"{rid}.log")
        if os.path.exists(p) and meh._dG("ringcofactor",rid) is not None and abs(meh._dG("ringcofactor",rid))<1e5:
            return p
    if pfa.is_isomerization(d[rid]["species"]):
        return os.path.join(LOGDIR,"full367",f"{rid}.log")
    p=os.path.join(LOGDIR,"ph0_sweep",f"{rid}.log")
    if meh._dG("ph0_sweep",rid) is not None:
        return p
    return os.path.join(LOGDIR,"full367",f"{rid}.log")

SIG=re.compile(r"σ=([\d.]+)")
CAP=re.compile(r"CAPPED")
QLINE=re.compile(r"^\s+(\S+)\s+q([+-]?\d+)\s+\[")
def parse_log(path):
    if not os.path.exists(path): return None
    sigs=[]; caps=0; charges=[]
    for ln in open(path,errors="ignore"):
        if "seeds=" in ln or "water ref" in ln or "q+" in ln or "q-" in ln:
            m=SIG.search(ln)
            if m: sigs.append(float(m.group(1)))
            if CAP.search(ln): caps+=1
            qm=QLINE.match(ln)
            if qm: charges.append(int(qm.group(2)))
    return dict(max_sig=max(sigs) if sigs else 0.0, n_cap=caps, scored_charges=charges)

def balance(sp):
    """mass (heavy-atom composition) + charge balance of the STORED reaction. Returns (dHeavy, dCharge)."""
    from collections import Counter
    ratom=Counter(); patom=Counter(); rq=0; pq=0
    for c,q,s in sp.values():
        m=Chem.MolFromSmiles(s)
        if m is None: return None
        comp=Counter()
        for a in m.GetAtoms():
            comp[a.GetSymbol()]+=1
        for el,n in comp.items():
            if c<0: ratom[el]+=abs(c)*n
            else:   patom[el]+=abs(c)*n
        if c<0: rq+=abs(c)*q
        else:   pq+=abs(c)*q
    dHeavy=sum(abs(ratom[e]-patom[e]) for e in set(ratom)|set(patom) if e!="H")
    dH = ratom["H"]-patom["H"]
    return dHeavy, dH, rq-pq

def floppy(sp):
    mr=0; mh=0
    for c,q,s in sp.values():
        m=Chem.MolFromSmiles(s)
        if m:
            mr=max(mr,rdMD.CalcNumRotatableBonds(m)); mh=max(mh,m.GetNumHeavyAtoms())
    return mr,mh

rows=[]
for rid in d:
    u=meh.uma_dG(rid)
    if u is None or abs(u)>200: continue
    e=u-d[rid]["exp"][0]
    lg=parse_log(routed_log(rid)) or dict(max_sig=0,n_cap=0,scored_charges=[])
    bal=balance(d[rid]["species"])
    mr,mh=floppy(d[rid]["species"])
    rows.append(dict(rid=rid,err=e,aerr=abs(e),max_sig=lg["max_sig"],n_cap=lg["n_cap"],
                     scored_charges=lg["scored_charges"],
                     dHeavy=bal[0] if bal else None,dH=bal[1] if bal else None,dQ=bal[2] if bal else None,
                     rot=mr,heavy=mh,cls=pfa.rxn_class(rid)))

# --- correlations of |err| with each candidate driver ---
E=np.array([r["aerr"] for r in rows])
def corr(key):
    x=np.array([r[key] for r in rows],dtype=float)
    return np.corrcoef(x,E)[0,1]
print(f"N={len(rows)}  MAE={E.mean():.1f}")
print("Pearson r of |err| vs driver:")
for k in ["max_sig","n_cap","rot","heavy"]:
    print(f"   {k:10s}: r={corr(k):+.2f}")

# stored-reaction imbalance
imbal=[r for r in rows if (r["dHeavy"] or 0)>0 or (r["dQ"] or 0)!=0]
print(f"\nStored reactions heavy-atom-imbalanced: {sum(1 for r in rows if (r['dHeavy'] or 0)>0)} ; charge-imbalanced: {sum(1 for r in rows if (r['dQ'] or 0)!=0)}")

# scored-charge net (from log) — did the SCORED species balance? and did NH3 get scored as NH4+?
def scored_net(r): return sum(r["scored_charges"])
print("\nSampling-converged but high error (max_sig<3, |err|>20)  -> NOT undersampling:")
nb=[r for r in rows if r["max_sig"]<3 and r["aerr"]>20]
print(f"   {len(nb)} reactions; of all |err|>20 ({sum(1 for r in rows if r['aerr']>20)}), {100*len(nb)/max(1,sum(1 for r in rows if r['aerr']>20)):.0f}% are WELL-SAMPLED")
print("Undersampled high error (max_sig>=3 or CAPPED, |err|>20):")
us=[r for r in rows if (r["max_sig"]>=3 or r["n_cap"]>0) and r["aerr"]>20]
print(f"   {len(us)} reactions")

# bucket each reaction to a dominant source
def bucket(r):
    if (r["dQ"] or 0)!=0 or (r["dHeavy"] or 0)>0: return "imbalance"
    if r["max_sig"]>=3 or r["n_cap"]>=2:          return "undersampled"
    if r["heavy"]>=40 and r["rot"]>=8:            return "floppy/large"
    return "electronic/speciation"
buc=collections.Counter()
buc_err=collections.defaultdict(list)
for r in rows:
    b=bucket(r); buc[b]+=1; buc_err[b].append(r["aerr"])
print("\nDominant-source buckets (all reactions):")
for b,n in buc.most_common():
    print(f"   {b:22s} n={n:3d}  MAE={np.mean(buc_err[b]):5.1f}  (tail>20: {sum(1 for a in buc_err[b] if a>20)})")

print("\nTAIL (>20) by bucket:")
tail=[r for r in rows if r["aerr"]>20]
tb=collections.Counter(bucket(r) for r in tail)
for b,n in tb.most_common(): print(f"   {b:22s} {n}")
