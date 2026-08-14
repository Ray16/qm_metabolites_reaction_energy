#!/usr/bin/env python
"""Step 2: matched-scaffold (truncated-model) redox ΔG with UMA — the real test
of whether CANCELLATION tames the conformer noise that sank Step 1.

The two hard redox reactions (both: 2 GSH + NAD(P)+ -> NAD(P)H + GSSG + H+) have
large SPECTATOR moieties identical on both sides:
  - NAD(P)+ / NAD(P)H differ only by a hydride on the nicotinamide ring; the whole
    adenine-ribose-(2'phospho)-pyrophosphate tail is a spectator.
  - GSH / GSSG differ only at the cysteine thiol (S-H -> S-S); the gamma-Glu and
    Gly residues are spectators.
Truncating the spectators to small caps (1-methylnicotinamide; a thiol) computes
the SAME reaction energy (spectators cancel) while making conformer sampling
trivial -- the literature-standard trick (Jinich).  NADP vs NAD differ only by a
spectator 2'-phosphate -> the model is identical for both, predicting their small
(~6 kJ) experimental difference as ~0 (a feature, not a bug).

Half-reactions vs the computational hydrogen electrode (H+ + e- = 1/2 H2):
  thiol ox :  2 RSH        -> RSSR  + H2                    (clean, neutral, no ion)
  nic red  :  MNA+  + H2   -> MNAH  + H+                    (one released proton)
  sum      :  2 RSH + MNA+ -> RSSR + MNAH + H+   (= the biochemical reaction)

ΔG = [E(RSSR)+E(MNAH)] - [2 E(RSH)+E(MNA+)] + G(H+,aq,pH7) + Σ ΔGsolv
Step 2a (this script): GAS-PHASE electronic energies + conformer-noise diagnostic
+ the CHE proton term.  Solvation (Σ ΔGsolv) is Step 3 -- but note the charged
species here are a CATION (MNA+) + a proton, NOT polyanions, so implicit
solvation is far more reliable than the anion regime that broke the old composite.

Run (uma env):  CUDA_VISIBLE_DEVICES=1 python scripts/step2_redox_matched.py
"""
import json
import os
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.optimize import BFGS

from fairchem.core import pretrained_mlip, FAIRChemCalculator

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "artifacts")
os.makedirs(OUT, exist_ok=True)
EV2KJ = 96.485
KT_KJ = 2.4789

# --- model species (small caps; spectators truncated) ---
# charge, SMILES
MODELS = {
    "MNA+":   (+1, "C[n+]1cccc(C(N)=O)c1"),        # 1-methylnicotinamide cation (NAD+ model)
    "MNAH":   (0,  "O=C(N)C1=CN(C)C=CC1"),         # 1-methyl-1,4-dihydronicotinamide (NADH model)
    "MeSH":   (0,  "CS"),                          # methanethiol (glutathione thiol model)
    "MeSSMe": (0,  "CSSC"),                        # dimethyl disulfide (GSSG model)
}

# G(H+, aq) reference, kJ/mol.  Standard absolute aqueous proton free energy:
#   G_gas(H+) (= H - TS translational, ~ -26.3 kJ/mol at 298K)  +  ΔGsolv(H+)
#   ΔGsolv(H+) = -1104.5 kJ/mol (Tissandier/CCCBDB consensus, 1M)
#   1M->1atm and standard-state terms folded into the consensus value below.
# pH 7 correction: + RT ln(10^-7) = -2.303*RT*7.
G_H_GAS = -26.3
DGSOLV_H = -1104.5
RT_LN10 = 2.303 * 8.314e-3 * 298.15   # kJ/mol per pH unit
def g_proton(pH):
    return G_H_GAS + DGSOLV_H - RT_LN10 * pH   # released proton: +G(H+); pH lowers it


def conf_energies(smiles, q, calc, nconf):
    """UMA-opt conformers; return sorted absolute energies (kJ/mol) + H2 handled
    separately.  Small models -> low conformer spread (the point)."""
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    params = AllChem.ETKDGv3(); params.randomSeed = 1; params.pruneRmsThresh = 0.3
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=nconf, params=params))
    if not cids:
        AllChem.EmbedMolecule(m, randomSeed=1, useRandomCoords=True); cids = [0]
    try:
        AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=200)
    except Exception:
        pass
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    Es = []
    for cid in cids:
        a = Atoms(symbols=syms, positions=m.GetConformer(cid).GetPositions())
        a.info = {"charge": int(q), "spin": 1}; a.calc = calc
        try:
            BFGS(a, logfile=None).run(fmax=0.05, steps=150)
            e = a.get_potential_energy() * EV2KJ
            if np.isfinite(e):
                Es.append(e)
        except Exception:
            continue
    return np.array(sorted(Es))


def h2_energy(calc):
    a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    a.info = {"charge": 0, "spin": 1}; a.calc = calc
    BFGS(a, logfile=None).run(fmax=0.02, steps=50)
    return a.get_potential_energy() * EV2KJ


def main():
    nconf = 16
    print("loading uma-s-1p2 ...", flush=True)
    pu = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda")
    calc = FAIRChemCalculator(pu, task_name="omol")

    t0 = time.time()
    E, spread = {}, {}
    for name, (q, smi) in MODELS.items():
        Es = conf_energies(smi, q, calc, nconf)
        E[name] = float(Es.min())
        spread[name] = float(Es.max() - Es.min())
        print(f"  {name:7s} q{q:+d}  Emin {Es.min():14.1f} kJ  conf-spread {spread[name]:5.1f} kJ  "
              f"({len(Es)} conf)", flush=True)
    E["H2"] = h2_energy(calc)
    print(f"  H2         Emin {E['H2']:14.1f} kJ", flush=True)

    # --- assemble the redox reaction (electronic + CHE proton) ---
    # 2 RSH + MNA+ -> RSSR + MNAH + H+
    dE_elec = (E["MeSSMe"] + E["MNAH"]) - (2 * E["MeSH"] + E["MNA+"])
    gH = g_proton(7.0)
    dG_gas_che = dE_elec + gH        # gas-phase electronic + aqueous proton, NO other solvation yet

    # diagnostic: also report the two clean half-reaction energies
    dE_thiol = E["MeSSMe"] + E["H2"] - 2 * E["MeSH"]      # 2RSH -> RSSR + H2 (clean)
    dE_nic = E["MNAH"] + gH - E["MNA+"] - E["H2"]         # MNA+ + H2 -> MNAH + H+

    exp = {"rxn00070_NAD": 18.0, "rxn00086_NADP": 11.9}   # both -> same model
    max_spread = max(spread.values())
    print(f"\n==== matched-scaffold (truncated-model) redox, UMA gas + CHE proton ====")
    print(f"  conformer noise: max model spread {max_spread:.1f} kJ  "
          f"(vs 49 kJ median for full-molecule Step 1 -> cancellation/truncation WORKS)"
          if max_spread < 20 else
          f"  conformer noise: max model spread {max_spread:.1f} kJ (still high)")
    print(f"  half-reaction  2 RSH -> RSSR + H2 :  ΔE_elec {dE_thiol:8.1f} kJ  (thiol oxidation, gas)")
    print(f"  half-reaction  MNA+ +H2-> MNAH +H+:  ΔG      {dE_nic:8.1f} kJ  (incl. CHE proton, no solv)")
    print(f"  FULL reaction 2 RSH + MNA+ -> RSSR + MNAH + H+:")
    print(f"     ΔG (gas elec + CHE proton, NO model solvation yet) = {dG_gas_che:8.1f} kJ/mol")
    print(f"     experiment: rxn00070(NAD) {exp['rxn00070_NAD']}, rxn00086(NADP) {exp['rxn00086_NADP']} kJ")
    print(f"  NOTE: model solvation of MNA+ (cation) + MeSSMe/MNAH (neutral) still to add (Step 3);")
    print(f"        no polyanions here, so implicit solvation should be reliable.")
    print(f"  total {time.time()-t0:.0f}s")

    json.dump(dict(E_kJ=E, conf_spread_kJ=spread, max_spread=max_spread,
                   dE_elec=dE_elec, g_proton_pH7=gH, dG_gas_che=dG_gas_che,
                   dE_thiol_half=dE_thiol, dE_nic_half=dE_nic, exp=exp,
                   note="gas-phase electronic + CHE proton; model solvation is Step 3"),
              open(os.path.join(OUT, "step2_redox_matched.json"), "w"), indent=2)
    print("wrote artifacts/step2_redox_matched.json")


if __name__ == "__main__":
    main()
