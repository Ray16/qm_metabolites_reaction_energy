"""Regime-3 (open-shell/multireference) probe: p-benzoquinone + H2 -> hydroquinone.
Known ΔG ~ -135 kJ/mol (E0'=+0.70 V, 2e/2H). Relax with UMA, dump geoms + UMA electronic
ΔE for an AIMNet2 rescore. If UMA is far from -135 AND/OR UMA vs AIMNet2 disagree a lot, the
conjugated redox couple is electronically hard (the semiquinone RADICAL is the real wall).
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from batched_relax import load_uma, batched_energies, batched_fire
from step4e_targeted import pool_confs
EV2KJ=96.485
SP={"BQ":(-1,0,"O=C1C=CC(=O)C=C1"),"H2":(-1,0,"[H][H]"),"H2Q":(1,0,"Oc1ccc(O)cc1")}
OUT=os.path.join(os.path.dirname(__file__),"..","artifacts","quinone"); os.makedirs(OUT,exist_ok=True)

def relax_best(pu,smi,q,seeds=(1,2,3),keep=8,pool=60):
    best=None
    for s in seeds:
        cands=pool_confs(smi,q,s,pool)
        order=np.argsort(batched_energies(pu,cands))[:keep]
        rel,E,conv=batched_fire(pu,[cands[i] for i in order],fmax=0.05,steps=300,stop_frac=0.9,return_converged=True,label="q")
        for a,e,c in zip(rel,E,conv):
            if c and (best is None or e<best[0]): best=(float(e),a.get_chemical_symbols(),a.get_positions().copy())
    return best

def main():
    pu=load_uma(); rows={}
    for n,(c,q,smi) in SP.items():
        b=relax_best(pu,smi,q); E=b[0]*EV2KJ
        p=os.path.join(OUT,f"{n}.xyz")
        with open(p,"w") as f:
            f.write(f"{len(b[1])}\ncharge={q} coeff={c}\n")
            for s,(x,y,z) in zip(b[1],b[2]): f.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")
        rows[n]=dict(coeff=c,charge=q,uma_E_kj=round(E,3)); print(f"{n:4s} UMA_E {E:.1f}",flush=True)
    dE=sum(r["coeff"]*r["uma_E_kj"] for r in rows.values())
    print(f"\nUMA gas ΔE_elec(BQ+H2->H2Q) = {dE:+.1f} kJ/mol   vs exp ΔG ~ -135",flush=True)
    json.dump(rows,open(os.path.join(OUT,"uma.json"),"w"),indent=2)

if __name__=="__main__": main()
