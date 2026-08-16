"""Regime-1 Step 2: Mg2+-phosphate BINDING via ligand substitution (keeps Mg 6-coordinate
both sides so the -1830 kJ Mg hydration cancels). Truncated ATP -> methyl triphosphate.

  Mg(H2O)6^2+  +  MePPP^3-   ->   [Mg(H2O)3.MePPP]^-  +  3 H2O
  dG_bind = G_aq(cluster) + 3 G_liq(H2O) - G_aq(Mg(H2O)6) - G_aq(MePPP)
Compare to experimental Mg-ATP: log K ~4 -> dG ~ -24 kJ/mol (adenosine truncated = spectator
for Mg, which binds the beta,gamma phosphates). Validation only -- nothing fitted.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "backup", "explicit_water"))
from batched_relax import load_uma, batched_energies, batched_fire
from step7b_charge_balanced_waters import bare_geom
from thermal_solv import uma_gibbs_corr, xtb_dgsolv
from ase import Atoms
EV2KJ = 96.485
RT_LN_C = 8.314e-3 * 298.15 * np.log(55.34)

def G_aq(pu, sym, pos, q, label):
    """E_UMA + UMA thermal + xtb COSMO solvation for a (relaxed) species."""
    E = float(batched_energies(pu, [Atoms(symbols=list(sym), positions=pos, info={"charge":int(q),"spin":1})])[0])*EV2KJ
    return E + uma_gibbs_corr(pu, list(sym), pos, q) + xtb_dgsolv(list(sym), pos, q, "cosmo")

def relax(pu, sym, pos, q, label, steps=500):
    rel,E,conv = batched_fire(pu,[Atoms(symbols=sym,positions=pos,info={"charge":int(q),"spin":1})],
                              fmax=0.05,steps=steps,stop_frac=1.0,return_converged=True,label=label)
    return rel[0].get_chemical_symbols(), rel[0].get_positions()

def octahedral_waters(center, d=2.09, avoid=None):
    out=[]
    for u in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
        u=np.array(u,float); O=center+d*u
        if avoid is not None and any(np.linalg.norm(O-a)<1.6 for a in avoid): continue
        perp=np.cross(u,[0.3,0.5,0.8]); perp/=np.linalg.norm(perp)
        out.append(("O",O)); out.append(("H",O+0.96*(0.6*u+0.8*perp))); out.append(("H",O+0.96*(0.6*u-0.8*perp)))
    return out

def main():
    pu=load_uma()
    Gw = (lambda s,c:(float(batched_energies(pu,[Atoms(symbols=list(s),positions=c,info={"charge":0,"spin":1})])[0])*EV2KJ
          + uma_gibbs_corr(pu,list(s),c,0)+xtb_dgsolv(list(s),c,0,"cosmo")+RT_LN_C))(*bare_geom(pu,0,"O"))
    print(f"G_liq(H2O) = {Gw:.1f}", flush=True)

    # Mg(H2O)6 2+
    ms,mp = bare_geom(pu,0,"O")  # dummy; build hexaaqua fresh
    mg_center=np.zeros(3)
    aq=[("Mg",mg_center)]+octahedral_waters(mg_center)
    s6=[a for a,_ in aq]; p6=np.array([x for _,x in aq])
    s6,p6=relax(pu,s6,p6,2,"MgW6")
    G_MgW6=G_aq(pu,s6,p6,2,"MgW6")
    print(f"G_aq[Mg(H2O)6]2+ = {G_MgW6:.1f}", flush=True)

    # MePPP 3-
    ps,pp=bare_geom(pu,-3,"COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-]")
    G_PPP=G_aq(pu,ps,pp,-3,"MePPP")
    print(f"G_aq[MePPP]3- = {G_PPP:.1f}", flush=True)

    # [Mg(H2O)3.MePPP]-  : place Mg near the two terminal (gamma) phosphate O-, + 3 waters
    Oidx=[i for i,a in enumerate(ps) if a=="O"]
    # terminal O = anionic O furthest from the methyl C (atom 0 region). pick 3 lowest-coordination O far from C
    cpos=pp[[i for i,a in enumerate(ps) if a=="C"][0]]
    farO=sorted(Oidx,key=lambda i:-np.linalg.norm(pp[i]-cpos))[:3]
    mgpos=pp[farO].mean(0);
    # push Mg 2.1 A off the centroid, away from the phosphate backbone
    out_dir=mgpos-pp.mean(0); out_dir/=np.linalg.norm(out_dir)+1e-9
    mgpos=mgpos+2.0*out_dir
    waters=octahedral_waters(mgpos, avoid=pp)  # skip water positions clashing with phosphate
    waters=waters[:9]  # up to 3 waters (3 atoms each)
    sc=list(ps)+["Mg"]+[a for a,_ in waters]
    pc=np.vstack([pp,[mgpos]]+[x for _,x in waters])
    sc,pc=relax(pu,sc,pc,-1,"MgPPP",steps=600)
    # Mg coordination check
    mgi=[i for i,a in enumerate(sc) if a=="Mg"][0]
    od=sorted(round(np.linalg.norm(pc[i]-pc[mgi]),2) for i,a in enumerate(sc) if a=="O")[:7]
    print(f"[Mg(H2O)3.MePPP]- Mg-O nearest: {od}", flush=True)
    nwat=(len(sc)-len(ps)-1)//3
    G_clus=G_aq(pu,sc,pc,-1,"MgPPP")
    print(f"G_aq[Mg(H2O){nwat}.MePPP]- = {G_clus:.1f}", flush=True)

    dG = G_clus + nwat*Gw - G_MgW6 - G_PPP + (6-nwat)*0  # waters released = 6 - nwat? approx via nwat
    # proper: Mg(H2O)6 + PPP -> Mg(H2O)_nwat.PPP + (6-nwat) H2O
    dG = G_clus + (6-nwat)*Gw - G_MgW6 - G_PPP
    print(f"\n=== Mg-phosphate binding dG = {dG:+.1f} kJ/mol   vs Mg-ATP exp ~ -24 ===", flush=True)
    print(f"    (nwat on Mg={nwat}, {6-nwat} waters released)", flush=True)

if __name__=="__main__":
    main()
