"""Group every reaction by MECHANISM (enzyme-name + SMARTS), report signed bias and sign-consistency.
A high |mean| with consistent sign = SYSTEMATIC (correctable). Near-zero mean w/ big std = scatter.
This is the test for 'is there a systematic pattern' the tail/mid analysis hinted at.
"""
import os, importlib.util, collections
import numpy as np
from rdkit import Chem

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, f))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
pfa = _load("pfa", "ph0_final_analysis.py")
meh = _load("meh", "make_error_histogram.py")
d = pfa.d

GUAN = Chem.MolFromSmarts("[NX3][CX3](=[NX2,NX3+])[NX3]")   # guanidinium (arg/creatine/taurocyamine)
COA  = Chem.MolFromSmarts("SCCNC(=O)CCNC(=O)")
THIOEST = Chem.MolFromSmarts("[#6]C(=O)[#16X2]")
NICO = Chem.MolFromSmarts("[n,N]1cccc(c1)C(=O)[NX3]")
AAC  = Chem.MolFromSmarts("[NX3;H2,H1][CX4][CX3](=O)[OX1-,OX2H]")  # alpha-amino-acid motif

def has(sp, pat):
    return any(Chem.MolFromSmiles(s) is not None and Chem.MolFromSmiles(s).HasSubstructMatch(pat)
               for c,q,s in sp.values())

def count_motif(sp, pat, side):
    tot=0
    for c,q,s in sp.values():
        if (side=="R" and c<0) or (side=="P" and c>0):
            m=Chem.MolFromSmiles(s)
            if m: tot += abs(c)*len(m.GetSubstructMatches(pat))
    return tot

def mechanism(rid):
    r=d[rid]; note=r["note"].lower(); sp=r["species"]
    g=has(sp,GUAN); coa=has(sp,COA); nico=has(sp,NICO); aac_created = count_motif(sp,AAC,"P")!=count_motif(sp,AAC,"R")
    if g and ("kinase" in note or "phospha" in note or "creatine" in note or "cyamine" in note):
        return "phosphagen-PN-kinase"
    if coa: return "CoA-thioester"
    if nico and aac_created: return "NAD-reductive-amination"   # C=O<->C-NH2 on the substrate + NAD
    if nico: return "NAD-redox-alcohol"                         # ordinary hydroxyl/keto NAD redox
    if "transaminase" in note or "aminotransferase" in note: return "transaminase"
    if "ammonia-lyase" in note: return "ammonia-lyase"
    if "phosphorylase" in note or "phosphoribosyl" in note or "nucleosidase" in note: return "glycosyl"
    if "phosphatase" in note: return "phosphatase"
    if "hydratase" in note or "dehydratase" in note: return "hydratase"
    if "isomerase" in note or "epimerase" in note or "mutase" in note or pfa.is_isomerization(sp): return "isomerase"
    if "aldolase" in note: return "aldolase"
    return "other"

rows=[]
for rid in d:
    u=meh.uma_dG(rid)
    if u is None or abs(u)>200: continue
    err=u-d[rid]["exp"][0]
    rows.append((mechanism(rid), err, rid))

by=collections.defaultdict(list)
for mech,err,rid in rows: by[mech].append((err,rid))

print(f"{'mechanism':24s} {'n':>3s} {'meanErr':>8s} {'|mean|':>7s} {'std':>6s} {'MAE':>6s} {'sign':>10s}")
order=sorted(by, key=lambda k:-abs(np.mean([e for e,_ in by[k]]))*len(by[k])**0.5)
for mech in order:
    errs=[e for e,_ in by[mech]]
    npos=sum(1 for e in errs if e>0); nneg=len(errs)-npos
    consist=max(npos,nneg)/len(errs)
    print(f"{mech:24s} {len(errs):3d} {np.mean(errs):+8.1f} {abs(np.mean(errs)):7.1f} "
          f"{np.std(errs):6.1f} {np.mean(np.abs(errs)):6.1f}  {npos:2d}+/{nneg:2d}- {consist:.0%}")

print("\n# Correctable candidates: |mean| large AND sign-consistent >=70%")
for mech in order:
    errs=[e for e,_ in by[mech]]
    npos=sum(1 for e in errs if e>0); consist=max(npos,len(errs)-npos)/len(errs)
    if abs(np.mean(errs))>12 and consist>=0.70 and len(errs)>=3:
        mae_before=np.mean(np.abs(errs))
        mae_after=np.mean(np.abs(np.array(errs)-np.mean(errs)))   # after removing class-mean offset
        print(f"  {mech:24s} n={len(errs)} meanErr={np.mean(errs):+.1f}  MAE {mae_before:.1f}->{mae_after:.1f} (offset-corrected)")
