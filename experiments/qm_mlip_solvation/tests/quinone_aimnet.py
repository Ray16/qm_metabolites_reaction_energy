import os,sys,json,glob
import numpy as np
ART=os.path.join(os.path.dirname(__file__),"..","artifacts","quinone"); EV2KJ=96.485
def rd(p):
    L=open(p).read().splitlines(); n=int(L[0]); m=dict(kv.split("=") for kv in L[1].split() if "=" in kv)
    q=int(m["charge"]); c=int(float(m["coeff"])); sym=[];pos=[]
    for ln in L[2:2+n]:
        a,x,y,z=ln.split(); sym.append(a); pos.append([float(x),float(y),float(z)])
    return sym,np.array(pos),q,c
def main():
    from aimnet2calc import AIMNet2ASE; from ase import Atoms
    rows={}
    for p in sorted(glob.glob(os.path.join(ART,"*.xyz"))):
        n=os.path.splitext(os.path.basename(p))[0]; sym,pos,q,c=rd(p)
        at=Atoms(symbols=sym,positions=pos); at.calc=AIMNet2ASE("aimnet2",charge=q)
        E=float(np.asarray(at.get_potential_energy()).reshape(-1)[0])*EV2KJ
        rows[n]=dict(coeff=c,aimnet_E_kj=round(E,3)); print(f"{n:4s} AIMNet2_E {E:.1f}",flush=True)
    dE=sum(r["coeff"]*r["aimnet_E_kj"] for r in rows.values())
    uma=json.load(open(os.path.join(ART,"uma.json"))); dEu=sum(r["coeff"]*r["uma_E_kj"] for r in uma.values())
    print(f"\n=== ΔE_elec(BQ+H2->H2Q): UMA {dEu:+.1f}  AIMNet2 {dE:+.1f}  diff {dE-dEu:+.1f}  (exp ΔG ~-135) ===",flush=True)
if __name__=="__main__": main()
