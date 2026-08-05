import os, re, shutil, subprocess, tempfile, json, sys
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
GFN2="/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/macepolar/bin/xtb"
GXTB="/nfs/lambda_stor_01/homes/rzhu/gxtb/xtb-6.7.1/bin/xtb"
H2KJ=2625.499639
SPECIES=[("cpd00009","Phosphate","O=P([O-])([O-])O",-2),
         ("cpd00012","PPi","O=P([O-])([O-])OP(=O)([O-])O",-3),
         ("cpd00089","Glucose-1-phosphate","O=P([O-])([O-])O[C@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O",-2),
         ("cpd00082","D-Fructose","OC[C@H]1OC(O)(CO)[C@@H](O)[C@@H]1O",0),
         ("cpd00076","Sucrose","OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O",0)]
def write_xyz(sym,pos,path,c=""):
    with open(path,'w') as fh:
        fh.write(f"{len(sym)}\n{c}\n"+"".join(f"{s} {x:.8f} {y:.8f} {z:.8f}\n" for s,(x,y,z) in zip(sym,pos)))
def read_xyz(p):
    L=open(p).read().splitlines(); n=int(L[0].split()[0])
    return [l.split()[0] for l in L[2:2+n]], np.array([[float(x) for x in l.split()[1:4]] for l in L[2:2+n]])
def run(binary,args,wd):
    return subprocess.run([binary,*args],cwd=wd,capture_output=True,text=True,
                          env={**os.environ,'OMP_NUM_THREADS':'8'},timeout=7200)
def energy(out,pat=r"TOTAL ENERGY\s+(-?\d+\.\d+)"):
    m=re.search(pat,out); return float(m.group(1)) if m else None
res={}
for cid,name,smi,chg in SPECIES:
    mol=Chem.AddHs(Chem.MolFromSmiles(smi))
    p=AllChem.ETKDGv3(); p.randomSeed=0xC0FFEE
    if AllChem.EmbedMolecule(mol,p)!=0: print(f"{cid}: embed failed"); continue
    AllChem.MMFFOptimizeMolecule(mol,maxIters=2000)
    sym=[a.GetSymbol() for a in mol.GetAtoms()]
    pos=mol.GetConformer().GetPositions()
    wd=tempfile.mkdtemp(prefix=f"ab_{cid}_"); write_xyz(sym,pos,f"{wd}/start.xyz")
    arms={}
    for arm,(binary,extra) in {"GFN2/ALPB":(GFN2,["--gfn","2","--alpb","water"]),
                               "g-xTB/ddCOSMO":(GXTB,["--gxtb","--cosmo","water"])}.items():
        d=f"{wd}/{arm.replace('/','_')}"; os.makedirs(d); shutil.copy(f"{wd}/start.xyz",f"{d}/in.xyz")
        import time; t0=time.time()
        r=run(binary,["in.xyz",*extra,"--opt","--chrg",str(chg),"--uhf","0"],d)
        dt=time.time()-t0
        conv="normal termination" in r.stdout
        opt=f"{d}/xtbopt.xyz"
        arms[arm]={"converged":conv,"seconds":dt,"geom":opt if os.path.exists(opt) else None}
        print(f"  {cid:10s} {arm:16s} conv={conv} {dt:7.1f}s")
    if not all(a["geom"] for a in arms.values()): res[cid]={"error":"opt failed"}; continue
    # heavy-atom RMSD between the two optimised geometries
    s1,g1=read_xyz(arms["GFN2/ALPB"]["geom"]); s2,g2=read_xyz(arms["g-xTB/ddCOSMO"]["geom"])
    hv=[i for i,s in enumerate(s1) if s!="H"]
    A,B=g1[hv],g2[hv]
    A=A-A.mean(0); B=B-B.mean(0)
    U,S,Vt=np.linalg.svd(A.T@B); d=np.sign(np.linalg.det(Vt.T@U.T))
    R=Vt.T@np.diag([1,1,d])@U.T
    rmsd=float(np.sqrt(((B-(R@A.T).T)**2).sum(1).mean()))
    # score both geometries with the SAME downstream terms: xtb-GFN2 dGsolv single points
    scored={}
    for arm,a in arms.items():
        d=f"{wd}/score_{arm.replace('/','_')}"; os.makedirs(d); shutil.copy(a["geom"],f"{d}/in.xyz")
        ea=energy(run(GFN2,["in.xyz","--gfn","2","--alpb","water","--chrg",str(chg),"--uhf","0"],d).stdout)
        dg=f"{d}/gas"; os.makedirs(dg); shutil.copy(a["geom"],f"{dg}/in.xyz")
        eg=energy(run(GFN2,["in.xyz","--gfn","2","--chrg",str(chg),"--uhf","0"],dg).stdout)
        scored[arm]={"dGsolv_kJ":(ea-eg)*H2KJ,"E_xtb_gas_Eh":eg}
    res[cid]={"name":name,"charge":chg,"heavy_rmsd_A":rmsd,
              "arms":{k:{**arms[k],"geom":None,**scored[k]} for k in arms}}
    print(f"  {cid:10s} q={chg:+d}  heavy RMSD={rmsd:.3f} A   "
          f"dGsolv: GFN2 {scored['GFN2/ALPB']['dGsolv_kJ']:9.1f}  gxtb-geom {scored['g-xTB/ddCOSMO']['dGsolv_kJ']:9.1f}  "
          f"delta {scored['g-xTB/ddCOSMO']['dGsolv_kJ']-scored['GFN2/ALPB']['dGsolv_kJ']:+7.1f} kJ")
    shutil.rmtree(wd,ignore_errors=True)
json.dump(res,open('/tmp/geo_ab.json','w'),indent=1)
