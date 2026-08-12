#!/usr/bin/env python
"""Integration test: does correcting speciation improve reaction dG prediction?

Applies the carbonyl-hydration ensemble correction (-RT ln(1+Khyd) to each
hydrated species' effective formation energy) to the scored 367-reaction set and
measures the change vs experiment. Uses curated experimental Khyd
(pipeline/hydration_constants.json) so the test isolates 'does speciation
correctness help' from 'can QC predict speciation' (validated separately at
~10 kJ in experiments/speciation/). Stratifies by charge, because the phosphate
solvation wall dominates the multiply-charged reactions and swamps any small
speciation correction there.
"""
import json, math, os, sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); THERMO=os.path.dirname(HERE)
sc=json.load(open(os.path.join(THERMO,'results','benchmark','tecrdb_full_scored.json')))
rx=json.load(open(os.path.join(HERE,'tecrdb_full_reactions.json')))
exp=json.load(open(os.path.join(HERE,'tecrdb_full_experiment.json')))
spec=json.load(open(os.path.join(HERE,'tecrdb_full_species.json')))
khyd=json.load(open(os.path.join(HERE,'hydration_constants.json')))['compounds']
RT=8.314462618e-3*298.15; qc=sc['scored_kJ']
corr={c:-RT*math.log(1+d['Khyd']) for c,d in khyd.items()}
def cor(r): return qc[r]+sum(v*corr.get(c,0.0) for c,v in rx[r].items())
def maxz(r): return max(abs(int(spec[c]['charge'])) for c in rx[r])
def mae(f,rs): return np.mean([abs(f(r)-exp[r]['dG_kJ']) for r in rs])
aff=[r for r in qc if any(c in corr for c in rx[r])]
print(f"affected reactions: {len(aff)}   whole-set MAE {mae(lambda r:qc[r],list(qc)):.1f} -> {mae(cor,list(qc)):.1f}")
for lo,hi,lab in [(0,1,'low-charge |z|<=1 (no wall)'),(2,9,'|z|>=2 (wall)')]:
    s=[r for r in aff if lo<=maxz(r)<=hi]
    if s: print(f"  {lab:28s} n={len(s):2d}  MAE {mae(lambda r:qc[r],s):5.1f} -> {mae(cor,s):5.1f}  "
                f"improved {sum(abs(cor(r)-exp[r]['dG_kJ'])<abs(qc[r]-exp[r]['dG_kJ']) for r in s)}/{len(s)}")
if __name__=='__main__': pass
