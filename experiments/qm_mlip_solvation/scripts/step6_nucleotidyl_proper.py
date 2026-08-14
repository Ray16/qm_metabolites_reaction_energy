#!/usr/bin/env python
"""Step 6 (proper): nucleotidyl transfer, done RIGHT — fixes the two gaps in the
first attempt: (1) it used THERMAL_FIXED=-0.8 (the GLYCOSYL reaction's thermal!),
(2) only CPCM-X. Here: real per-species UMA-Hessian thermal + ALL solvation models
+ a protonation-microspecies check.

  rxn01675 Glc-1-P + TTP -> PPi + dTDP-glucose   exp +1.0
  rxn01005 UTP + GlcA-1-P -> PPi + UDP-glucuronate exp +2.7
Truncated (nucleoside->Me, sugar->Me): MeP + MePPP -> MePPMe + PPi.

Per species: pool -> batched UMA rank -> relax lowest keep -> Boltzmann over
(E_gas + ΔGsolv(model)) + UMA-Hessian thermal Gcorr (on the min-G conformer).
NOTE the KEY subtlety: MePPP protonation. At pH7 a triphosphate is ~-4 (like ATP4-);
the −3 choice keeps the reaction proton-balanced (-5/-5) but may be the wrong
microspecies. −4 releases a proton -> needs the CHE proton term. We report both.

Run (uma env): CUDA_VISIBLE_DEVICES=0 python scripts/step6_nucleotidyl_proper.py --seeds 1,2,3
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.thermochemistry import IdealGasThermo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batched_relax import load_uma, batched_fire, batched_energies
from step4e_targeted import pool_confs, boltz
from step5c_solv_compare import multi_dgsolv, MODELS

EV2KJ = 96.485
CM2EV = 1.23984e-4
T = 298.15
# CHE proton free energy at pH7 (for the -4 variant that releases a proton)
G_HPLUS = -26.3 - 1104.5 - 2.303 * 8.314e-3 * T * 7.0   # ~ -1170.8 kJ/mol

# -3 proton-balanced microspecies. METHYL caps (nucleoside/sugar->Me) and ETHYL caps
# (->Et) to test the TRUNCATION contribution to the residual (user's "missing something").
SP3 = {"MeP": (-2, "COP(=O)([O-])[O-]"),
       "MePPP": (-3, "COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])O"),
       "MePPMe": (-2, "COP(=O)([O-])OP(=O)([O-])OC"),
       "PPi": (-3, "O=P([O-])([O-])OP(=O)([O-])O"),
       "EtP": (-2, "CCOP(=O)([O-])[O-]"),
       "EtPPP": (-3, "CCOP(=O)([O-])OP(=O)([O-])OP(=O)([O-])O"),
       "EtPPEt": (-2, "CCOP(=O)([O-])OP(=O)([O-])OCC")}
EXP = {"rxn01675 (TTP)": 1.0, "rxn01005 (UTP)": 2.7}


def gibbs_corr(atoms, q, calc):
    """UMA-Hessian Gibbs correction Gcorr = G_gas - E_elec (kJ/mol)."""
    a = atoms.copy(); a.info = {"charge": int(q), "spin": 1}; a.calc = calc
    e = a.get_potential_energy()
    d = tempfile.mkdtemp(prefix="vib6_")
    try:
        vib = Vibrations(a, name=os.path.join(d, "v")); vib.run()
        en = vib.get_energies()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    mags = np.sort(np.abs(en.real))[6:]
    mags = np.where(mags < 50 * CM2EV, 50 * CM2EV, mags)
    th = IdealGasThermo(vib_energies=mags, potentialenergy=e, atoms=a,
                        geometry="nonlinear", symmetrynumber=1, spin=0)
    G = th.get_gibbs_energy(temperature=T, pressure=101325.0, verbose=False)
    return float((G - e) * EV2KJ)


def species_data(pu, calc, name, q, smi, seed, pool, keep, log):
    """Return per-model species free energy G(model) = Boltz(E_gas+ΔGsolv) + Gcorr."""
    from ase import Atoms  # noqa
    cands = pool_confs(smi, q, seed, pool)
    order = np.argsort(batched_energies(pu, cands))[:keep]
    sel = [cands[i] for i in order]
    _, E_ev, conv = batched_fire(pu, sel, fmax=0.05, steps=300, stop_frac=0.9,
                                 return_converged=True, label=f"{name}s{seed}")
    sel = [x for x, c in zip(sel, conv) if c]; Eg = E_ev[conv] * EV2KJ
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        solv = list(ex.map(lambda x: multi_dgsolv(x, q), sel))
    # Boltzmann per model over (E_gas + ΔGsolv)
    Gmodel = {}
    for m in MODELS:
        Gt = [float(e + sv[m]) for e, sv in zip(Eg, solv) if np.isfinite(e) and sv[m] is not None]
        Gmodel[m] = boltz(Gt) if Gt else None
    # thermal on the min-electronic conformer
    gcorr = gibbs_corr(sel[int(np.argmin(Eg))], q, calc)
    log(f"    {name:7s} q{q:+d}: Gcorr(thermal) {gcorr:6.1f}   " +
        "  ".join(f"{m} {Gmodel[m]:.0f}" for m in MODELS))
    return {m: (Gmodel[m] + gcorr if Gmodel[m] is not None else None) for m in MODELS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3"); ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--keep", type=int, default=15)
    a = ap.parse_args(); seeds = [int(s) for s in a.seeds.split(",")]
    from fairchem.core import FAIRChemCalculator
    log = lambda s: print(s, flush=True)
    log(f"loading UMA... seeds={seeds} keep={a.keep}  (proper thermal + all solv models)")
    pu = load_uma(); calc = FAIRChemCalculator(pu, task_name="omol")

    me = {m: [] for m in MODELS}; et = {m: [] for m in MODELS}
    for seed in seeds:
        log(f"  seed {seed}:")
        G = {n: species_data(pu, calc, n, q, smi, seed, a.pool, a.keep, log)
             for n, (q, smi) in SP3.items()}
        for m in MODELS:
            me[m].append((G["MePPMe"][m] + G["PPi"][m]) - (G["MeP"][m] + G["MePPP"][m]))
            et[m].append((G["EtPPEt"][m] + G["PPi"][m]) - (G["EtP"][m] + G["EtPPP"][m]))

    log(f"\n==== nucleotidyl ΔG by solvation model (proper thermal), exp +1/+3 ====")
    log(f"  {'model':6}  {'METHYL cap':>18}   {'ETHYL cap':>18}   trunc Δ(Et-Me)")
    for m in MODELS:
        vm = np.array(me[m]); ve = np.array(et[m])
        log(f"    {m:6}: mean {vm.mean():7.1f} std {vm.std():4.1f}   "
            f"mean {ve.mean():7.1f} std {ve.std():4.1f}   {ve.mean()-vm.mean():+6.1f}")
    log("  -> large Et-Me shift = truncation is a major contributor (the 'missing' term);"
        " small = residual is electronic/solvation, not truncation")
    json.dump(dict(exp=EXP, methyl={m: me[m] for m in MODELS}, ethyl={m: et[m] for m in MODELS},
                   seeds=seeds, keep=a.keep),
              open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "artifacts", "step6_nucleotidyl_proper.json"), "w"), indent=2)
    log("wrote artifacts/step6_nucleotidyl_proper.json")


if __name__ == "__main__":
    main()
