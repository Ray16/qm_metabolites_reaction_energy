#!/usr/bin/env python
"""Route 1 diagnostic: is redox a QC niche? Cofactor-cancelling pairwise test.

Differences two reactions sharing a cofactor couple so the badly-solvated
cofactor cancels exactly, then stratifies the residual substrate-exchange error
by substrate charge. Runs on the committed G_aq_tecrdb_full scored output.
"""
import json, itertools, os, sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); PIPE=os.path.join(os.path.dirname(HERE),'pipeline')
sc=json.load(open(os.path.join(os.path.dirname(HERE),'results','benchmark','tecrdb_full_scored.json')))
rx=json.load(open(os.path.join(PIPE,'tecrdb_full_reactions.json')))
exp=json.load(open(os.path.join(PIPE,'tecrdb_full_experiment.json')))
spec=json.load(open(os.path.join(PIPE,'tecrdb_full_species.json')))
mets={m['id']:m for m in json.load(open(os.path.join(PIPE,'tecrdb_full_metabolites.json')))}
qc=sc['scored_kJ']
COUPLES={'NAD':('cpd00003','cpd00004'),'NADP':('cpd00006','cpd00005')}
COF=set(sum(COUPLES.values(),()))
def couple_of(r):
    st=rx[r]; f=[n for n,(ox,red) in COUPLES.items() if ox in st and red in st and st[ox]*st[red]<0]
    return f[0] if len(f)==1 else None
def subs(r): return [c for c in rx[r] if c not in COF and c!='cpd00001']
def maxz(r): return max([abs(int(spec[c]['charge'])) for c in subs(r)] or [0])
groups={}
for r in qc:
    c=couple_of(r)
    if c: groups.setdefault(c,[]).append(r)
def analyze(filt,label):
    errs=[]
    for name,rs in groups.items():
        ox,red=COUPLES[name]; rs2=[r for r in rs if filt(r)]
        for a,b in itertools.combinations(rs2,2):
            if (rx[a][ox],rx[a][red])!=(rx[b][ox],rx[b][red]): continue
            errs.append(abs((qc[a]-qc[b])-(exp[a]['dG_kJ']-exp[b]['dG_kJ'])))
    if errs: print(f"  {label:34s} pairs={len(errs):5d}  MAE {np.mean(errs):5.1f}  median {np.median(errs):5.1f}")
if __name__=='__main__':
    for z,lab in [(0,'neutral substrates'),(1,'substrate max|z|=1'),(2,'substrate max|z|>=2')]:
        analyze((lambda r,z=z: (maxz(r)==z if z<2 else maxz(r)>=2)), lab)
