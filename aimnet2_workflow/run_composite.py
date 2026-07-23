#!/usr/bin/env python
"""Composite free-energy workflow prototype (kept OUT of the qm_thermo package):

    xtb (sample/optimise geometry)
      -> AIMNet2  (gas-phase electronic energy, wB97M-D3 quality)
      -> xtb ALPB (implicit-solvation free energy, thermodynamic-cycle dGsolv)
      -> xtb       (thermal/entropy G_RRHO from a Hessian)

  G_aq(compound) = E_AIMNet2(gas) + dGsolv(ALPB) + G_RRHO

Reaction Delta_rG'^o is then formed with the SAME qm_thermo transform used by the
DFT/xtb bars, so the result is directly comparable. Validated on the 5 TECRDB
reactions in figures/qm_vs_experiment.png (11 compounds).

All work is on the xtb-optimised geometry (the "xtb samples geometry" step), so
electronic energy, solvation, and thermal corrections are mutually consistent.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import torch

THERMO = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc"
sys.path.insert(0, THERMO)
from qm_thermo import config                                  # noqa: E402
from qm_thermo.structures import load_metabolites             # noqa: E402
from qm_thermo.reactions import reaction_dG, species_info     # noqa: E402
from qm_thermo.references import reactions_within             # noqa: E402

XTB = "/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/xtb/bin/xtb"
HARTREE_TO_KJ = 2625.499639
EV_TO_KJ = 96.48533212
WORK = os.path.join(THERMO, "aimnet2_workflow", "scratch")
os.makedirs(WORK, exist_ok=True)

# 5 TECRDB figure reactions and the 11 compounds they touch.
RXNS = ["rxn00830", "rxn00283", "rxn00191", "rxn00260", "rxn00276"]


def _xtb(args, cwd):
    return subprocess.run([XTB, *args], cwd=cwd, capture_output=True, text=True,
                          env={**os.environ, "OMP_NUM_THREADS": "4"})


def xtb_optimise_and_thermal(geom_abs, chg, wd):
    """xtb --ohess in ALPB water. Returns (xtbopt_xyz, E_alpb_Eh, G_RRHO_Eh)."""
    os.makedirs(wd, exist_ok=True)
    r = _xtb([geom_abs, "--gfn", "2", "--alpb", "water", "--ohess",
              "--chrg", str(chg), "--uhf", "0"], wd)
    e = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", r.stdout)
    g = re.search(r"TOTAL FREE ENERGY\s+(-?\d+\.\d+)", r.stdout)
    e_alpb, g_tot = float(e.group(1)), float(g.group(1))
    return os.path.join(wd, "xtbopt.xyz"), e_alpb, g_tot - e_alpb


def xtb_sp(geom_abs, chg, wd, solvent):
    """Single-point TOTAL ENERGY (Eh). solvent='' for gas, 'water' for ALPB."""
    os.makedirs(wd, exist_ok=True)
    flags = ["--alpb", "water"] if solvent else []
    r = _xtb([geom_abs, "--gfn", "2", *flags, "--chrg", str(chg), "--uhf", "0"], wd)
    return float(re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", r.stdout).group(1))


_PT = {"H": 1, "C": 6, "N": 7, "O": 8, "P": 15, "S": 16}


def read_xyz(path):
    lines = open(path).read().splitlines()
    n = int(lines[0])
    syms, xyz = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        syms.append(_PT[p[0]])
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return torch.tensor(xyz), torch.tensor(syms)


def main():
    mets = {m.cpd_id: m for m in load_metabolites()}
    refs = reactions_within(set(mets))
    need = set()
    for rid in RXNS:
        need |= {c for c in refs[rid].reaction.compounds() if c != "cpd00067"}

    from aimnet2calc import AIMNet2Calculator
    calc = AIMNet2Calculator("aimnet2")

    G_aq = {}
    for c in sorted(need):
        chg = mets[c].charge
        geom = os.path.join(THERMO, "results", "geometries", c)
        conf = sorted(f for f in os.listdir(geom) if f.endswith(".xyz"))[0]
        geom_abs = os.path.join(geom, conf)
        wd = os.path.join(WORK, c)

        xtbopt, e_alpb, g_rrho = xtb_optimise_and_thermal(geom_abs, chg, wd)
        e_gas_xtb = xtb_sp(xtbopt, chg, os.path.join(wd, "gas"), "")
        dgsolv = e_alpb - e_gas_xtb                              # ALPB solvation, Eh

        xyz, numbers = read_xyz(xtbopt)
        out = calc({"coord": xyz, "numbers": numbers,
                    "charge": torch.tensor([float(chg)])}, forces=False)
        e_aimnet_kJ = float(out["energy"]) * EV_TO_KJ           # gas electronic

        g_kJ = e_aimnet_kJ + dgsolv * HARTREE_TO_KJ + g_rrho * HARTREE_TO_KJ
        G_aq[c] = g_kJ
        print(f"{c} q={chg:+d}: E_AIMNet2={e_aimnet_kJ:12.1f}  "
              f"dGsolv={dgsolv*HARTREE_TO_KJ:7.1f}  G_RRHO={g_rrho*HARTREE_TO_KJ:6.1f}"
              f"  -> G_aq={g_kJ:12.1f} kJ/mol")

    json.dump(G_aq, open(os.path.join(THERMO, "aimnet2_workflow",
                                      "G_aq_composite.json"), "w"), indent=2)

    # Reaction Delta_rG'^o via the standard transform; compare to DFT + experiment.
    species = {c: species_info(mets[c]) for c in G_aq}
    dft = {}
    for c in G_aq:
        d = json.load(open(os.path.join(THERMO, "results", "compounds", f"{c}.json")))
        dft[c] = d["gibbs_highlevel_kJ"] or d["gibbs_kJ"]
    import csv
    exp = {}
    for r in csv.DictReader(open(os.path.join(THERMO, "results", "benchmark",
                                              "experimental_dG_TECRDB.csv"))):
        if r["dG_exp_nearstd_kJ"]:
            exp[r["rxn_id"]] = float(r["dG_exp_nearstd_kJ"])

    print(f"\n{'reaction':28s} {'exp':>6s} {'DFT':>7s} {'AIMNet2-comp':>13s}")
    out_rows = {}
    aerr, derr = [], []
    for rid in RXNS:
        ref = refs[rid].reaction
        comp = reaction_dG(ref, G_aq, species, conditions=config.DEFAULT_CONDITIONS)
        dftv = reaction_dG(ref, dft, species, conditions=config.DEFAULT_CONDITIONS)
        out_rows[rid] = comp.dG_transformed_kJ
        e = exp.get(rid)
        tag = ""
        if e is not None:
            aerr.append(comp.dG_transformed_kJ - e)
            derr.append(dftv.dG_transformed_kJ - e)
        es = f"{e:6.1f}" if e is not None else "   n/a"
        print(f"{rid:28s} {es} {dftv.dG_transformed_kJ:7.1f} {comp.dG_transformed_kJ:13.1f}")
    json.dump(out_rows, open(os.path.join(THERMO, "results", "benchmark",
                                          "aimnet2_reaction_dG.json"), "w"), indent=2)
    mae = lambda v: sum(abs(x) for x in v) / len(v)
    print(f"\nMAE vs experiment (n={len(aerr)}):  DFT={mae(derr):.1f}   "
          f"AIMNet2-composite={mae(aerr):.1f}   (xtb-only was 18.1) kJ/mol")


if __name__ == "__main__":
    main()
