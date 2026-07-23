#!/usr/bin/env python
"""Replace the xtb Hessian thermal step with an AIMNet2 Hessian (same PES as the
electronic energy, on GPU). For each figure compound: AIMNet2-optimise -> ASE
finite-difference Hessian -> IdealGasThermo G_RRHO. Compares G_RRHO and wall-time
against the xtb values, and re-scores the 5 reactions with thermal swapped in.
"""
from __future__ import annotations
import json, os, shutil, sys, time
import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo

THERMO = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc"
sys.path.insert(0, THERMO)
from qm_thermo import config
from qm_thermo.structures import load_metabolites
from qm_thermo.reactions import reaction_dG, species_info
from qm_thermo.references import reactions_within
sys.path.insert(0, os.path.join(THERMO, "aimnet2_workflow", "aimnet2"))
from aimnet2calc import AIMNet2ASE

EV_TO_KJ = 96.48533212
SCR = "/tmp/qm_thermo_scratch/aimnet_thermal"
os.makedirs(SCR, exist_ok=True)
RXNS = ["rxn00830", "rxn00283", "rxn00191", "rxn00260", "rxn00276"]

# reused: AIMNet2 gas electronic (kJ/mol), xtb-ALPB dGsolv (kJ/mol), xtb G_RRHO (kJ/mol)
E_AIMNET = {"cpd00020": -898157.6, "cpd00023": -1447904.6, "cpd00024": -1495190.1,
            "cpd00032": -1391824.8, "cpd00033": -747204.9, "cpd00035": -850506.1,
            "cpd00040": -794841.9, "cpd00041": -1344582.7, "cpd00113": -3693404.8,
            "cpd00117": -850506.2, "cpd00202": -3693398.4}
DGSOLV_XTB = {"cpd00020": -267.7, "cpd00023": -304.6, "cpd00024": -759.5, "cpd00032": -841.2,
              "cpd00033": -143.3, "cpd00035": -135.8, "cpd00040": -249.4, "cpd00041": -335.2,
              "cpd00113": -853.9, "cpd00117": -135.7, "cpd00202": -866.2}
GRRHO_XTB = {"cpd00020": 67.1, "cpd00023": 260.0, "cpd00024": 128.7, "cpd00032": 63.6,
             "cpd00033": 126.8, "cpd00035": 193.6, "cpd00040": -0.7, "cpd00041": 189.2,
             "cpd00113": 318.1, "cpd00117": 194.0, "cpd00202": 315.2}


def read_xyz(path):
    L = open(path).read().splitlines(); n = int(L[0]); syms, xyz = [], []
    for ln in L[2:2 + n]:
        p = ln.split(); syms.append(p[0]); xyz.append([float(x) for x in p[1:4]])
    return syms, xyz


def qrrho_dS(vib_energies_eV, T):
    """Grimme (2012) quasi-RRHO entropy minus harmonic entropy, summed over modes
    (J/mol/K). Low-frequency modes are interpolated toward a free rotor, damping
    the harmonic-entropy divergence that corrupts floppy/zwitterionic species."""
    h, kB, c, R = 6.62607015e-34, 1.380649e-23, 2.99792458e10, 8.314462618
    B_av, w0 = 1.0e-44, 100.0   # kg m^2 ; cm^-1
    dS = 0.0
    for E in vib_energies_eV:
        wn = float(E) * 8065.544   # eV -> cm^-1
        if wn <= 0:
            continue
        nu = wn * c                # Hz
        x = h * nu / (kB * T)
        S_HO = R * (x / (np.exp(x) - 1) - np.log(1 - np.exp(-x)))
        mu = h / (8 * np.pi ** 2 * nu)
        mu_p = mu * B_av / (mu + B_av)
        S_FR = R * (0.5 + np.log(np.sqrt(8 * np.pi ** 3 * mu_p * kB * T / h ** 2)))
        wgt = 1.0 / (1.0 + (w0 / wn) ** 4)
        dS += (wgt * S_HO + (1 - wgt) * S_FR) - S_HO
    return dS


def aimnet_grrho(syms, xyz, chg, wd):
    """AIMNet2 opt + Hessian -> G_RRHO (kJ/mol) and wall seconds."""
    if os.path.isdir(wd):
        shutil.rmtree(wd)
    os.makedirs(wd)
    t0 = time.perf_counter()
    atoms = Atoms(symbols=syms, positions=xyz)
    atoms.calc = AIMNet2ASE("aimnet2", charge=chg, mult=1)
    BFGS(atoms, logfile=os.path.join(wd, "opt.log")).run(fmax=0.02, steps=200)
    e_elec = atoms.get_potential_energy()                       # eV
    vib = Vibrations(atoms, name=os.path.join(wd, "vib"))
    vib.run()
    energies = vib.get_energies()                               # complex eV, 3N modes
    # keep real vibrational modes above ~30 cm^-1 (drops 6 trans/rot + tiny imaginary)
    cutoff_eV = 30 * 1.23984e-4
    vibs = sorted(e.real for e in energies if abs(e.imag) < 1e-6 and e.real > cutoff_eV)
    n_imag = sum(1 for e in energies if abs(e.imag) > 1e-6 and abs(e.imag) > cutoff_eV)
    thermo = IdealGasThermo(vib_energies=np.array(vibs), potentialenergy=e_elec,
                            atoms=atoms, geometry="nonlinear", symmetrynumber=1, spin=0)
    g = thermo.get_gibbs_energy(temperature=298.15, pressure=101325.0, verbose=False)
    g_harm = float((g - e_elec) * EV_TO_KJ)
    g_qrrho = g_harm - 298.15 * qrrho_dS(vibs, 298.15) / 1000.0
    return g_qrrho, g_harm, time.perf_counter() - t0, n_imag


def main():
    mets = {m.cpd_id: m for m in load_metabolites()}
    refs = reactions_within(set(mets))
    need = set()
    for rid in RXNS:
        need |= {c for c in refs[rid].reaction.compounds() if c != "cpd00067"}

    grrho, t_tot = {}, 0.0
    print(f"{'cpd':9s} {'qRRHO':>8s} {'harm':>8s} {'xtb':>8s} {'qRRHO-xtb':>10s} {'wall_s':>7s} {'imag':>5s}")
    for c in sorted(need):
        chg = mets[c].charge
        gdir = os.path.join(THERMO, "results", "geometries", c)
        geom = os.path.join(gdir, sorted(f for f in os.listdir(gdir) if f.endswith(".xyz"))[0])
        syms, xyz = read_xyz(geom)
        g, g_harm, dt, nim = aimnet_grrho(syms, xyz, chg, os.path.join(SCR, c))
        grrho[c] = g; t_tot += dt
        print(f"{c:9s} {g:8.1f} {g_harm:8.1f} {GRRHO_XTB[c]:8.1f} {g-GRRHO_XTB[c]:10.1f} {dt:7.1f} {nim:5d}")
    print(f"\nTotal AIMNet2 thermal wall: {t_tot:.1f} s for {len(grrho)} compounds "
          f"({t_tot/len(grrho):.1f} s/cpd)  vs xtb ~30 s/cpd")

    # re-score the 5 reactions with AIMNet2 thermal (electronic + xtb-ALPB solv unchanged)
    G = {c: E_AIMNET[c] + DGSOLV_XTB[c] + grrho[c] for c in grrho}
    species = {c: species_info(mets[c]) for c in G}
    import csv
    exp = {r["rxn_id"]: float(r["dG_exp_nearstd_kJ"])
           for r in csv.DictReader(open(os.path.join(THERMO, "results/benchmark/experimental_dG_TECRDB.csv")))
           if r["dG_exp_nearstd_kJ"]}
    v1 = json.load(open(os.path.join(THERMO, "results/benchmark/aimnet2_reaction_dG.json")))
    print(f"\n{'reaction':10s} {'exp':>6s} {'xtb-thermal':>11s} {'AIMNet2-thermal':>15s}")
    ae = []
    for rid in RXNS:
        new = reaction_dG(refs[rid].reaction, G, species, conditions=config.DEFAULT_CONDITIONS).dG_transformed_kJ
        e = exp.get(rid)
        if e is not None:
            ae.append(new - e)
        es = f"{e:6.1f}" if e is not None else "   n/a"
        print(f"{rid:10s} {es} {v1.get(rid, float('nan')):11.1f} {new:15.1f}")
    if ae:
        print(f"\nMAE vs exp (n={len(ae)}): AIMNet2-thermal = {sum(abs(x) for x in ae)/len(ae):.1f}  "
              f"(xtb-thermal was 10.8) kJ/mol  [solvation still xtb-ALPB]")


if __name__ == "__main__":
    main()
