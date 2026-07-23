#!/usr/bin/env python
"""Solvation upgrade for the AIMNet2 composite (separate-folder prototype).

The decomposition showed the composite's error is dominated by the xtb-ALPB
solvation term, NOT the (wB97M-D3-quality) AIMNet2 electronic energy. This swaps
the solvation model for DFT-SMD while keeping AIMNet2 electronic + xtb thermal:

    G_aq = E_AIMNet2(gas)  +  dGsolv(DFT-SMD)  +  G_RRHO(xtb)
                                 ^^^^^^^^^^^^  upgraded from xtb-ALPB

dGsolv(DFT-SMD) = E_SMD(r2SCAN-3c) - E_gas(r2SCAN-3c), a single-point thermodynamic
cycle on the DFT-optimised geometry. Compared on the 5 TECRDB figure reactions.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import torch

THERMO = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc"
ORCA = "/nfs/lambda_stor_01/homes/rzhu/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca"
XTB = "/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/xtb/bin/xtb"
sys.path.insert(0, THERMO)
from qm_thermo import config                                  # noqa: E402
from qm_thermo.structures import load_metabolites             # noqa: E402
from qm_thermo.reactions import reaction_dG, species_info     # noqa: E402
from qm_thermo.references import reactions_within             # noqa: E402

HARTREE_TO_KJ = 2625.499639
EV_TO_KJ = 96.48533212
SCR = "/tmp/qm_thermo_scratch/improve_solv"
os.makedirs(SCR, exist_ok=True)
RXNS = ["rxn00830", "rxn00283", "rxn00191", "rxn00260", "rxn00276"]
_PT = {"H": 1, "C": 6, "N": 7, "O": 8, "P": 15, "S": 16}
_SYM = {v: k for k, v in _PT.items()}


def read_xyz(path):
    L = open(path).read().splitlines()
    n = int(L[0])
    syms, xyz = [], []
    for ln in L[2:2 + n]:
        p = ln.split()
        syms.append(p[0]); xyz.append([float(p[1]), float(p[2]), float(p[3])])
    return syms, xyz


def orca_sp(syms, xyz, chg, smd, wd):
    """r2SCAN-3c single point; smd=True adds SMD(water). Returns energy (Eh)."""
    os.makedirs(wd, exist_ok=True)
    lines = ["! r2SCAN-3c TightSCF", "%maxcore 3000", "%pal nprocs 8 end"]
    if smd:
        lines += ['%cpcm', '  smd true', '  SMDsolvent "water"', 'end']
    lines.append(f"* xyz {chg} 1")
    for s, c in zip(syms, xyz):
        lines.append(f"  {s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
    lines.append("*")
    inp = os.path.join(wd, "sp.inp")
    open(inp, "w").write("\n".join(lines) + "\n")
    out = subprocess.run([ORCA, inp], cwd=wd, capture_output=True, text=True)
    m = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", out.stdout)
    return float(m[-1]) if m else None


def xtb_thermal(geom_abs, chg, wd):
    """xtb --ohess in ALPB water -> G_RRHO (Eh) = G_total - E_total."""
    os.makedirs(wd, exist_ok=True)
    r = subprocess.run([XTB, geom_abs, "--gfn", "2", "--alpb", "water", "--ohess",
                        "--chrg", str(chg), "--uhf", "0"], cwd=wd,
                       capture_output=True, text=True,
                       env={**os.environ, "OMP_NUM_THREADS": "4"})
    e = float(re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", r.stdout).group(1))
    g = float(re.search(r"TOTAL FREE ENERGY\s+(-?\d+\.\d+)", r.stdout).group(1))
    return g - e


def main():
    mets = {m.cpd_id: m for m in load_metabolites()}
    refs = reactions_within(set(mets))
    need = set()
    for rid in RXNS:
        need |= {c for c in refs[rid].reaction.compounds() if c != "cpd00067"}

    from aimnet2calc import AIMNet2Calculator
    calc = AIMNet2Calculator("aimnet2")

    G = {}
    for c in sorted(need):
        chg = mets[c].charge
        gdir = os.path.join(THERMO, "results", "geometries", c)
        geom = os.path.join(gdir, sorted(f for f in os.listdir(gdir) if f.endswith(".xyz"))[0])
        syms, xyz = read_xyz(geom)
        wd = os.path.join(SCR, c)

        e_gas = orca_sp(syms, xyz, chg, False, os.path.join(wd, "gas"))
        e_smd = orca_sp(syms, xyz, chg, True, os.path.join(wd, "smd"))
        dgsolv = (e_smd - e_gas) * HARTREE_TO_KJ                 # DFT-SMD solvation
        grrho = xtb_thermal(geom, chg, os.path.join(wd, "thermal")) * HARTREE_TO_KJ
        nums = torch.tensor([_PT[s] for s in syms])
        out = calc({"coord": torch.tensor(xyz), "numbers": nums,
                    "charge": torch.tensor([float(chg)])}, forces=False)
        e_elec = float(out["energy"]) * EV_TO_KJ

        G[c] = e_elec + dgsolv + grrho
        print(f"{c} q={chg:+d}: dGsolv(SMD)={dgsolv:7.1f} (xtb-ALPB ref differs)  "
              f"G_RRHO={grrho:6.1f}  -> G_aq={G[c]:12.1f}")

    species = {c: species_info(mets[c]) for c in G}
    dft = {}
    for c in G:
        d = json.load(open(os.path.join(THERMO, "results", "compounds", f"{c}.json")))
        dft[c] = d["gibbs_highlevel_kJ"] or d["gibbs_kJ"]
    import csv
    exp = {}
    for r in csv.DictReader(open(os.path.join(THERMO, "results", "benchmark",
                                              "experimental_dG_TECRDB.csv"))):
        if r["dG_exp_nearstd_kJ"]:
            exp[r["rxn_id"]] = float(r["dG_exp_nearstd_kJ"])
    v1 = json.load(open(os.path.join(THERMO, "results", "benchmark",
                                     "aimnet2_reaction_dG.json")))

    print(f"\n{'reaction':12s} {'exp':>6s} {'DFT':>7s} {'AIMNet2+xtbsolv':>16s} {'AIMNet2+SMDsolv':>16s}")
    out_rows, ae = {}, []
    for rid in RXNS:
        ref = refs[rid].reaction
        new = reaction_dG(ref, G, species, conditions=config.DEFAULT_CONDITIONS).dG_transformed_kJ
        dftv = reaction_dG(ref, dft, species, conditions=config.DEFAULT_CONDITIONS).dG_transformed_kJ
        out_rows[rid] = new
        e = exp.get(rid)
        if e is not None:
            ae.append(new - e)
        es = f"{e:6.1f}" if e is not None else "   n/a"
        print(f"{rid:12s} {es} {dftv:7.1f} {v1.get(rid, float('nan')):16.1f} {new:16.1f}")
    json.dump(out_rows, open(os.path.join(THERMO, "results", "benchmark",
                                          "aimnet2_smdsolv_reaction_dG.json"), "w"), indent=2)
    if ae:
        print(f"\nMAE vs exp (n={len(ae)}): AIMNet2+SMD-solv = "
              f"{sum(abs(x) for x in ae)/len(ae):.1f}  (was 10.8 with xtb-ALPB; DFT 5.5) kJ/mol")


if __name__ == "__main__":
    main()
