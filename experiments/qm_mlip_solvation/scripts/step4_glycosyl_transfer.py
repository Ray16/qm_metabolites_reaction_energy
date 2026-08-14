#!/usr/bin/env python
"""Step 4: INDEPENDENCE TEST — the same workflow on a glycosyl-transfer reaction,
a different class from the redox couple.

rxn00579:  UDP-glucose + D-fructose -> UDP + sucrose   (exp ΔG'° = -4.2 kJ/mol)

Truncate the uridine spectator -> methyl.  The truncated model reaction
  MeUDP-Glc + Fructose -> MeUDP + Sucrose
is atom- AND charge-balanced (verified) with NO net proton -> this test drops the
load-bearing CHE proton reference entirely.  The q-2 diphosphate anion sits on
BOTH sides, so xTB-ALPB anion-undersolvation largely cancels.  The one new risk
vs redox is sugar conformer flexibility -> we sample a large ensemble and report
the spread.

Full workflow (identical to the redox Step 3b): UMA conformer-opt -> UMA Hessian
(ase Vibrations + IdealGasThermo) -> G_gas(298); + xTB-ALPB(water) ΔGsolv.
  ΔG_aq = [G(MeUDP)+G(Suc)] - [G(MeUDPGlc)+G(Fru)] + ΔΔGsolv         (no proton)

Run (uma env):  CUDA_VISIBLE_DEVICES=1 python scripts/step4_glycosyl_transfer.py
"""
import json
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo

from fairchem.core import pretrained_mlip, FAIRChemCalculator

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "artifacts")
GEOM = os.path.join(OUT, "geom_glycosyl")
os.makedirs(GEOM, exist_ok=True)
EV2KJ = 96.485
HARTREE2KJ = 2625.4996
CM2EV = 1.23984e-4
T = 298.15

SPECIES = {
    "MeUDPGlc": (-2, "OC[C@H]1O[C@@H](OP(=O)([O-])OP(=O)([O-])OC)[C@H](O)[C@@H](O)[C@@H]1O"),
    "Fructose": (0,  "OC[C@H]1OC(O)(CO)[C@@H](O)[C@@H]1O"),
    "MeUDP":    (-2, "COP(=O)([O-])OP(=O)([O-])O"),
    "Suc":      (0,  "OC[C@H]1O[C@@H](O[C@]2(CO)O[C@H](CO)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O"),
}
EXP = -4.2   # kJ/mol, rxn00579


def uma_confs(smiles, q, calc, nconf=24):
    """UMA-opt a conformer ensemble; return (best Atoms, sorted E_kJ array)."""
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3(); p.randomSeed = 1; p.pruneRmsThresh = 0.3
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=nconf, params=p))
    if not cids:
        AllChem.EmbedMolecule(m, randomSeed=1, useRandomCoords=True); cids = [0]
    try:
        AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=300)
    except Exception:
        pass
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    Es, best = [], (1e18, None)
    for cid in cids:
        a = Atoms(symbols=syms, positions=m.GetConformer(cid).GetPositions())
        a.info = {"charge": int(q), "spin": 1}; a.calc = calc
        try:
            BFGS(a, logfile=None).run(fmax=0.03, steps=250)
            e = a.get_potential_energy() * EV2KJ
            Es.append(e)
            if e < best[0]:
                best = (e, a.copy())
        except Exception:
            continue
    return best[1], np.array(sorted(Es))


def gibbs_gas(atoms, q, calc, tag):
    atoms = atoms.copy(); atoms.info = {"charge": int(q), "spin": 1}; atoms.calc = calc
    e_elec = atoms.get_potential_energy()
    vdir = tempfile.mkdtemp(prefix=f"vib_{tag}_")
    try:
        vib = Vibrations(atoms, name=os.path.join(vdir, "vib")); vib.run()
        energies = vib.get_energies()
    finally:
        shutil.rmtree(vdir, ignore_errors=True)
    mags = np.sort(np.abs(energies.real))[6:]
    mags = np.where(mags < 50 * CM2EV, 50 * CM2EV, mags)
    thermo = IdealGasThermo(vib_energies=mags, potentialenergy=e_elec, atoms=atoms,
                            geometry="nonlinear", symmetrynumber=1, spin=0)
    G = thermo.get_gibbs_energy(temperature=T, pressure=101325.0, verbose=False)
    return float(G * EV2KJ), float(e_elec * EV2KJ)


def write_xyz(atoms, path):
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\n\n")
        for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            f.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")


def xtb_dgsolv(xyz, q):
    def e(solv):
        cmd = ["conda", "run", "-n", "xtb", "xtb", os.path.abspath(xyz),
               "--gfn", "2", "--chrg", str(int(q)), "--sp"]
        if solv:
            cmd += ["--alpb", solv]
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=400)
        m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", r.stdout)
        if not m:
            raise RuntimeError(f"xtb fail {xyz} solv={solv}: {r.stdout[-400:]}")
        return float(m.group(1))
    return (e("water") - e(None)) * HARTREE2KJ


def main():
    print("loading uma-s-1p2 ...", flush=True)
    pu = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda")
    calc = FAIRChemCalculator(pu, task_name="omol")

    G_gas, E_elec, dGsolv, spread = {}, {}, {}, {}
    for name, (q, smi) in SPECIES.items():
        best, Es = uma_confs(smi, q, calc)
        spread[name] = float(Es.max() - Es.min())
        xyz = os.path.join(GEOM, f"{name}.xyz"); write_xyz(best, xyz)
        g, e = gibbs_gas(best, q, calc, name)
        ds = xtb_dgsolv(xyz, q)
        G_gas[name], E_elec[name], dGsolv[name] = g, e, ds
        print(f"  {name:9s} q{q:+d}  Ggas {g:13.1f}  Gcorr {g-e:6.1f}  ΔGsolv {ds:7.1f}  "
              f"conf-spread {spread[name]:5.1f} kJ ({len(Es)} conf)", flush=True)

    # ΔG_aq = [MeUDP + Suc] - [MeUDPGlc + Fru] + ΔΔGsolv   (no proton)
    dG_gas = (G_gas["MeUDP"] + G_gas["Suc"]) - (G_gas["MeUDPGlc"] + G_gas["Fructose"])
    dG_elec = (E_elec["MeUDP"] + E_elec["Suc"]) - (E_elec["MeUDPGlc"] + E_elec["Fructose"])
    dSolv = (dGsolv["MeUDP"] + dGsolv["Suc"]) - (dGsolv["MeUDPGlc"] + dGsolv["Fructose"])
    dG_aq_elec = dG_elec + dSolv
    dG_aq = dG_gas + dSolv

    print(f"\n==== Step 4 glycosyl transfer (rxn00579), independence test ====")
    print(f"  max sugar conformer spread {max(spread.values()):.1f} kJ")
    print(f"  ΔG_elec(gas) {dG_elec:8.1f}   ΔG_gas(+thermal) {dG_gas:8.1f}  (thermal {dG_gas-dG_elec:+.1f})")
    print(f"  ΔΔGsolv (xTB-ALPB, q-2 diphosphate both sides -> cancels) {dSolv:8.1f}")
    print(f"  ΔG_aq  electronic-only {dG_aq_elec:7.1f}   WITH THERMAL {dG_aq:7.1f} kJ/mol")
    print(f"  experiment {EXP} kJ  ->  err(elec) {dG_aq_elec-EXP:+.1f}   err(thermal) {dG_aq-EXP:+.1f}")
    print(f"  (dGP-retrained on rxn00579: -46.9 kJ, off by ~43)")

    json.dump(dict(reaction="rxn00579", exp=EXP, G_gas=G_gas, E_elec=E_elec,
                   dGsolv=dGsolv, conf_spread=spread, dG_gas=dG_gas, dG_elec=dG_elec,
                   dSolv=dSolv, dG_aq_elec=dG_aq_elec, dG_aq_thermal=dG_aq,
                   err_thermal=dG_aq - EXP),
              open(os.path.join(OUT, "step4_glycosyl_transfer.json"), "w"), indent=2)
    print("wrote artifacts/step4_glycosyl_transfer.json")


if __name__ == "__main__":
    main()
