#!/usr/bin/env python
"""Step 3b: add THERMAL free-energy corrections (ZPE + thermal enthalpy + entropy)
to the matched-scaffold redox model, and test a faithful capped-cysteine thiol
model.  This is the one legitimate missing physics term from Step 3 (which used
electronic ΔE, not ΔG).

For each model species:
  UMA optimize -> UMA Hessian (ase.vibrations, finite diff on UMA forces)
  -> ase IdealGasThermo -> G_gas(298 K) = E_elec + ZPE + H_thermal - T S
Then  ΔG_aq = Σ G_gas(prod) - Σ G_gas(react) + ΔΔGsolv(xTB-ALPB) + G(H+,aq,pH7).

The reaction 2 RSH + MNA+ -> RSSR + MNAH + H+ conserves molecule count (3->3),
so the ill-defined translational/rotational entropy in solution largely CANCELS
-> the ideal-gas thermal treatment is reliable here (unlike Δn≠0 reactions).

Two thiol models: methanethiol (MeSH) and capped cysteine Ac-Cys-NHMe (faithful to
glutathione's peptide environment).  Run (uma env):
  CUDA_VISIBLE_DEVICES=1 python scripts/step3b_redox_thermal.py
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
GEOM = os.path.join(OUT, "geom_redox")
os.makedirs(GEOM, exist_ok=True)
EV2KJ = 96.485
HARTREE2KJ = 2625.4996
T = 298.15

MODELS = {
    "MNA+":   (+1, "C[n+]1cccc(C(N)=O)c1"),
    "MNAH":   (0,  "O=C(N)C1=CN(C)C=CC1"),
    "MeSH":   (0,  "CS"),
    "MeSSMe": (0,  "CSSC"),
    "CysSH":  (0,  "CC(=O)NC(CS)C(=O)NC"),                        # Ac-Cys-NHMe (capped cysteine)
    "CysSSCys": (0, "CC(=O)NC(CSSCC(NC(C)=O)C(=O)NC)C(=O)NC"),    # its disulfide
}
G_H_GAS, DGSOLV_H = -26.3, -1104.5
RT_LN10 = 2.303 * 8.314e-3 * T
def g_proton(pH):
    return G_H_GAS + DGSOLV_H - RT_LN10 * pH


def uma_opt_best(smiles, q, calc, nconf=16):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3(); p.randomSeed = 1; p.pruneRmsThresh = 0.3
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=nconf, params=p))
    if not cids:
        AllChem.EmbedMolecule(m, randomSeed=1, useRandomCoords=True); cids = [0]
    try:
        AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=200)
    except Exception:
        pass
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    best = (1e18, None)
    for cid in cids:
        a = Atoms(symbols=syms, positions=m.GetConformer(cid).GetPositions())
        a.info = {"charge": int(q), "spin": 1}; a.calc = calc
        try:
            BFGS(a, logfile=None).run(fmax=0.03, steps=200)
            e = a.get_potential_energy()
            if e * EV2KJ < best[0]:
                best = (e * EV2KJ, a.copy())
        except Exception:
            continue
    return best[1], best[0]


def gibbs_gas(atoms, q, calc, tag):
    """UMA Hessian -> IdealGasThermo G_gas(T) in kJ/mol. Drops the 6 lowest
    (trans+rot) modes; soft/imaginary vib modes floored to 50 cm^-1."""
    atoms = atoms.copy(); atoms.info = {"charge": int(q), "spin": 1}; atoms.calc = calc
    e_elec = atoms.get_potential_energy()                       # eV
    vdir = tempfile.mkdtemp(prefix=f"vib_{tag}_")
    try:
        vib = Vibrations(atoms, name=os.path.join(vdir, "vib"))
        vib.run()
        energies = vib.get_energies()                           # eV, complex
    finally:
        shutil.rmtree(vdir, ignore_errors=True)
    # real magnitudes, sort, drop 6 lowest (trans+rot), floor softs to 50 cm^-1
    CM2EV = 1.23984e-4
    mags = np.sort(np.abs(energies.real))[6:]                   # 3N-6 vibrational
    mags = np.where(mags < 50 * CM2EV, 50 * CM2EV, mags)
    thermo = IdealGasThermo(vib_energies=mags, potentialenergy=e_elec,
                            atoms=atoms, geometry="nonlinear",
                            symmetrynumber=1, spin=0)
    G = thermo.get_gibbs_energy(temperature=T, pressure=101325.0, verbose=False)  # eV
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
            r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=300)
        m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", r.stdout)
        return float(m.group(1))
    return (e("water") - e(None)) * HARTREE2KJ


def main():
    print("loading uma-s-1p2 ...", flush=True)
    pu = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda")
    calc = FAIRChemCalculator(pu, task_name="omol")

    G_gas, E_elec, dGsolv = {}, {}, {}
    for name, (q, smi) in MODELS.items():
        atoms, _ = uma_opt_best(smi, q, calc)
        xyz = os.path.join(GEOM, f"{name.replace('+','plus')}.xyz")
        write_xyz(atoms, xyz)
        g, e = gibbs_gas(atoms, q, calc, name.replace("+", "p"))
        ds = xtb_dgsolv(xyz, q)
        G_gas[name], E_elec[name], dGsolv[name] = g, e, ds
        print(f"  {name:9s} q{q:+d}  Eelec {e:13.1f}  Ggas {g:13.1f}  Gcorr {g-e:6.1f}  "
              f"ΔGsolv {ds:7.1f} kJ", flush=True)

    gH = g_proton(7.0)
    exp = {"NAD (rxn00070)": 18.0, "NADP (rxn00086)": 11.9}

    def reaction(thiol, disulf, label):
        # 2 thiol + MNA+ -> disulf + MNAH + H+
        dG_elec = (E_elec[disulf] + E_elec["MNAH"]) - (2 * E_elec[thiol] + E_elec["MNA+"])
        dG_gas = (G_gas[disulf] + G_gas["MNAH"]) - (2 * G_gas[thiol] + G_gas["MNA+"])
        dSolv = (dGsolv[disulf] + dGsolv["MNAH"]) - (2 * dGsolv[thiol] + dGsolv["MNA+"])
        dG_aq_elec = dG_elec + dSolv + gH          # Step-3 style (electronic, for comparison)
        dG_aq = dG_gas + dSolv + gH                # Step-3b (with thermal)
        print(f"\n  --- thiol model: {label} ---")
        print(f"    ΔG_elec(gas) {dG_elec:8.1f}   ΔG_gas(+thermal) {dG_gas:8.1f}   "
              f"(thermal shift {dG_gas-dG_elec:+.1f} kJ)")
        print(f"    ΔΔGsolv {dSolv:8.1f}   G(H+) {gH:8.1f}")
        print(f"    ΔG_aq  electronic-only {dG_aq_elec:7.1f}   WITH THERMAL {dG_aq:7.1f} kJ/mol")
        for k, v in exp.items():
            print(f"      vs {k}: err(elec) {dG_aq_elec-v:+6.1f}   err(thermal) {dG_aq-v:+6.1f}")
        return dict(model=label, dG_gas=dG_gas, dSolv=dSolv, gH=gH,
                    dG_aq_elec=dG_aq_elec, dG_aq_thermal=dG_aq,
                    thermal_shift=dG_gas - dG_elec)

    print(f"\n==== Step 3b: redox ΔG with thermal free-energy corrections ====")
    r_me = reaction("MeSH", "MeSSMe", "methanethiol")
    r_cys = reaction("CysSH", "CysSSCys", "capped cysteine (Ac-Cys-NHMe)")

    json.dump(dict(G_gas=G_gas, E_elec=E_elec, dGsolv=dGsolv, g_proton=gH,
                   exp=exp, methanethiol=r_me, cysteine=r_cys),
              open(os.path.join(OUT, "step3b_redox_thermal.json"), "w"), indent=2)
    print("\nwrote artifacts/step3b_redox_thermal.json")


if __name__ == "__main__":
    main()
