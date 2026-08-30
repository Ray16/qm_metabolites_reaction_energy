"""Diagnose the rxn00054 failure (4 o-aminophenol + 3 O2 -> 2 phenoxazinone + 6 H2O), all NEUTRAL so
the ~700 kJ error is ELECTRONIC or nothing. UMA vs DFT (PBE0/def2-TZVP) gas-phase ΔE localises it;
+ singlet-triplet gap of the phenoxazinone product tests diradical/multireference character (which a
single-reference-trained MLIP would get wrong)."""
import os, subprocess, tempfile, shutil
import numpy as np
from ase import Atoms
from rdkit import Chem; from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from metag.energetics.uma import load_uma, batched_fire, batched_energies
from metag.energetics.conformers import spin_multiplicity
EV=96.48533212; HA=2625.499639
XTB=f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb"; SCR=os.environ.get("QM_SCRATCH","/tmp/qmscr")
AP="Nc1ccccc1O"; O2="O=O"; PXO="Nc1cc2nc3ccccc3oc-2cc1=O"; W="O"
def xtbopt(smi,q,uhf=0):
    os.makedirs(SCR,exist_ok=True); wd=tempfile.mkdtemp(dir=SCR)
    try:
        m=Chem.AddHs(Chem.MolFromSmiles(smi)); AllChem.EmbedMolecule(m,randomSeed=1); AllChem.MMFFOptimizeMolecule(m)
        Chem.MolToXYZFile(m,f"{wd}/in.xyz")
        subprocess.run([XTB,"in.xyz","--gfn","2","--chrg",str(q),"--uhf",str(uhf),"--opt","tight"],cwd=wd,
                       env={**os.environ,"OMP_NUM_THREADS":"4"},capture_output=True,timeout=600)
        g=open(f"{wd}/xtbopt.xyz").read()
    finally: shutil.rmtree(wd,ignore_errors=True)
    s=[l.split()[0] for l in g.splitlines()[2:] if l.strip()]
    c=np.array([[float(x) for x in l.split()[1:4]] for l in g.splitlines()[2:] if l.strip()])
    return s,c
def dft(s,c,spin2):
    from pyscf import gto,dft as pdft
    mol=gto.M(atom=[(a,tuple(p)) for a,p in zip(s,c)],basis="def2-tzvp",spin=spin2,verbose=0)
    mf=(pdft.RKS(mol) if spin2==0 else pdft.UKS(mol)); mf.xc="PBE0"; return mf.kernel()*HA
def uma_e(pu,s,c,mult):
    a=Atoms(symbols=s,positions=c,info={"charge":0,"spin":int(mult)})
    rel,E=batched_fire(pu,[a],fmax=0.03,steps=400); return float(E[0])*EV, rel[0]
pu=load_uma("uma-s-1p2p1")
E={}  # species -> (uma, dft) at singlet-appropriate spin
for name,smi,mult in [("AP",AP,1),("O2",O2,3),("PXO",PXO,1),("W",W,1)]:
    s,c=xtbopt(smi,0,uhf=(mult-1))
    eu,rel=uma_e(pu,s,c,mult)
    ed=dft(rel.get_chemical_symbols(),rel.get_positions(),mult-1)
    E[name]=(eu,ed); print(f"  {name:4s} mult{mult}: UMA {eu:.1f}  DFT {ed:.1f}",flush=True)
dU=2*E["PXO"][0]+6*E["W"][0]-4*E["AP"][0]-3*E["O2"][0]
dD=2*E["PXO"][1]+6*E["W"][1]-4*E["AP"][1]-3*E["O2"][1]
print(f"\nrxn00054 gas ΔE:  UMA {dU:+.1f}   DFT {dD:+.1f}   UMA-DFT {dU-dD:+.1f} kJ/mol  (pipeline -1028)")
# diradical check: phenoxazinone triplet
s,c=xtbopt(PXO,0,uhf=2); euT,relT=uma_e(pu,s,c,3); edT=dft(relT.get_chemical_symbols(),relT.get_positions(),2)
print(f"PXO singlet-triplet gap: UMA {(E['PXO'][0]-euT):+.1f}  DFT {(E['PXO'][1]-edT):+.1f} kJ/mol "
      f"(large +ve = clean singlet; small/-ve = diradical/multireference -> single-ref UMA unreliable)")
