#!/usr/bin/env python
"""Step 5: glyoxalase rxn01834  (R)-S-Lactoylglutathione -> GSH + methylglyoxal (+H+).
EC 4.4.1.5 (lactoylglutathione lyase). exp ΔG'° = +23.5 kJ/mol (TECRDB, n=1).

GSH is CONSERVED (thioester on one side, free thiol on the other) -> truncate the
glutathione peptide to a small thiol cap (same trick as the Step-3b redox model).
The truncated reaction is charge-neutral and proton-balanced on its own:

    R-S-C(=O)-CH(OH)-CH3   ->   R-SH  +  CH3-CO-CHO
    (S-lactoyl thioester)       thiol    methylglyoxal

so NO explicit H+ term (the reaction's +H+ is a GSH-backbone pH-7 artifact that the
truncation removes -- the backbone carboxyls/amine are identical on both sides).

CAVEAT: Δn = +1 (1 molecule -> 2), so gas-phase trans/rot entropy does NOT cancel
(unlike the Δn=0 redox couple). The ideal-gas thermal term is therefore less reliable
here; we report electronic-only and with-thermal so the entropy sensitivity is visible.

Two thiol caps: methanethiol (MeSH) and capped cysteine (Ac-Cys-NHMe), faithful to GSH.
Run (uma env):  CUDA_VISIBLE_DEVICES=0 python scripts/step5_glyoxalase.py
"""
import json, os, re, shutil, subprocess, tempfile
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo
from fairchem.core import pretrained_mlip, FAIRChemCalculator

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "artifacts"); GEOM = os.path.join(OUT, "geom_glyoxalase")
os.makedirs(GEOM, exist_ok=True)
EV2KJ = 96.485; HARTREE2KJ = 2625.4996; T = 298.15

# name -> (charge, SMILES)
MODELS = {
    "MeSH":        (0, "CS"),
    "MeS_lactoyl": (0, "CSC(=O)C(O)C"),                         # CH3-S-CO-CH(OH)-CH3
    "MethylGlyox": (0, "CC(=O)C=O"),                            # methylglyoxal
    "CysSH":       (0, "CC(=O)NC(CS)C(=O)NC"),                  # Ac-Cys-NHMe
    "CysS_lactoyl":(0, "CC(=O)NC(CSC(=O)C(O)C)C(=O)NC"),        # its S-lactoyl thioester
}


def uma_opt_best(smiles, q, calc, nconf=24):
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
    return best[1]


def gibbs_gas(atoms, q, calc, tag):
    atoms = atoms.copy(); atoms.info = {"charge": int(q), "spin": 1}; atoms.calc = calc
    e_elec = atoms.get_potential_energy()
    vdir = tempfile.mkdtemp(prefix=f"vib_{tag}_")
    try:
        vib = Vibrations(atoms, name=os.path.join(vdir, "vib")); vib.run()
        energies = vib.get_energies()
    finally:
        shutil.rmtree(vdir, ignore_errors=True)
    CM2EV = 1.23984e-4
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
        atoms = uma_opt_best(smi, q, calc)
        xyz = os.path.join(GEOM, f"{name}.xyz"); write_xyz(atoms, xyz)
        g, e = gibbs_gas(atoms, q, calc, name)
        ds = xtb_dgsolv(xyz, q)
        G_gas[name], E_elec[name], dGsolv[name] = g, e, ds
        print(f"  {name:13s} Eelec {e:13.1f}  Ggas {g:13.1f}  Gcorr {g-e:6.1f}  ΔGsolv {ds:7.1f}", flush=True)

    exp = 23.48

    def reaction(thiol, thioester, label):
        # thioester -> thiol + methylglyoxal   (Δn = +1)
        dE = (E_elec[thiol] + E_elec["MethylGlyox"]) - E_elec[thioester]
        dG = (G_gas[thiol] + G_gas["MethylGlyox"]) - G_gas[thioester]
        dS = (dGsolv[thiol] + dGsolv["MethylGlyox"]) - dGsolv[thioester]
        dG_aq_elec = dE + dS
        dG_aq = dG + dS
        print(f"\n  --- thiol cap: {label} ---")
        print(f"    ΔE_elec(gas) {dE:8.1f}   ΔG_gas(+thermal) {dG:8.1f}   (thermal shift {dG-dE:+.1f})")
        print(f"    ΔΔGsolv {dS:8.1f}   (no H+ term)")
        print(f"    ΔG_aq electronic-only {dG_aq_elec:7.1f}   WITH THERMAL {dG_aq:7.1f}   exp {exp:+.1f}")
        print(f"      err(elec) {dG_aq_elec-exp:+6.1f}   err(thermal) {dG_aq-exp:+6.1f}")
        return dict(model=label, dE=dE, dG_gas=dG, dSolv=dS,
                    dG_aq_elec=dG_aq_elec, dG_aq_thermal=dG_aq, thermal_shift=dG-dE)

    print("\n==== Step 5: glyoxalase rxn01834 (truncated GSH) ====")
    r_me = reaction("MeSH", "MeS_lactoyl", "methanethiol")
    r_cys = reaction("CysSH", "CysS_lactoyl", "capped cysteine (Ac-Cys-NHMe)")

    json.dump(dict(reaction="rxn01834", exp=exp, G_gas=G_gas, E_elec=E_elec,
                   dGsolv=dGsolv, methanethiol=r_me, cysteine=r_cys,
                   caveat="Dn=+1 (1->2 molecules): gas trans/rot entropy does not cancel"),
              open(os.path.join(OUT, "step5_glyoxalase.json"), "w"), indent=2)
    print("\nwrote artifacts/step5_glyoxalase.json")


if __name__ == "__main__":
    main()
