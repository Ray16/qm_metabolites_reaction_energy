#!/usr/bin/env python
"""UMA composite free-energy workflow (parallel to aimnet2_workflow/run_composite.py).

Swaps ONLY the gas-phase electronic-energy method (AIMNet2 -> Meta UMA / OMol25)
while keeping the solvation and thermal terms identical, so the reaction-energy
difference vs the AIMNet2 / GFN2-xTB bars isolates the electronic-energy quality:

    xtb (sample/optimise geometry, ALPB water, --ohess)
      -> UMA       (gas-phase electronic energy, OMol25 head, uma-s-1p2)
      -> xtb ALPB  (implicit-solvation dGsolv via thermodynamic cycle)
      -> xtb       (thermal/entropy G_RRHO from the Hessian)

  G_aq(compound) = E_UMA(gas) + dGsolv(xtb-ALPB) + G_RRHO(xtb)

Reaction Delta_rG'^o is then formed with the SAME qm_thermo transform used by the
DFT / xtb / AIMNet2 bars, so the result is directly comparable. Validated on the
5 TECRDB figure reactions (11 compounds).

Run inside the `uma` env (needs a free GPU):
    CUDA_VISIBLE_DEVICES=2 /homes/rzhu/miniforge3/envs/uma/bin/python run_uma_composite.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

from ase import Atoms

# qm_thermo lives on the NFS project; uma_helper lives on the local home.
THERMO = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc"
sys.path.insert(0, THERMO)
sys.path.insert(0, "/homes/rzhu/uma_tools")
from qm_thermo import config                                  # noqa: E402
from qm_thermo.structures import load_metabolites             # noqa: E402
from qm_thermo.reactions import reaction_dG, species_info     # noqa: E402
from qm_thermo.references import reactions_within             # noqa: E402
import uma_helper                                             # noqa: E402

XTB = config.XTB_BIN
HARTREE_TO_KJ = 2625.499639
EV_TO_KJ = 96.48533212
UMA_MODEL = os.environ.get("UMA_MODEL", "uma-s-1p2")
WORK = os.path.join(THERMO, "uma_workflow", "scratch")
os.makedirs(WORK, exist_ok=True)

# Same 5 TECRDB figure reactions / 11 compounds as the AIMNet2 composite.
RXNS = ["rxn00830", "rxn00283", "rxn00191", "rxn00260", "rxn00276"]

_PT = {"H": 1, "C": 6, "N": 7, "O": 8, "P": 15, "S": 16}


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
    if e is None or g is None:
        raise RuntimeError(f"xtb --ohess failed in {wd}\n{r.stdout[-1500:]}\n{r.stderr[-500:]}")
    e_alpb, g_tot = float(e.group(1)), float(g.group(1))
    return os.path.join(wd, "xtbopt.xyz"), e_alpb, g_tot - e_alpb


def xtb_sp(geom_abs, chg, wd, solvent):
    """Single-point TOTAL ENERGY (Eh). solvent='' for gas, 'water' for ALPB."""
    os.makedirs(wd, exist_ok=True)
    flags = ["--alpb", "water"] if solvent else []
    r = _xtb([geom_abs, "--gfn", "2", *flags, "--chrg", str(chg), "--uhf", "0"], wd)
    m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", r.stdout)
    if m is None:
        raise RuntimeError(f"xtb SP failed in {wd}\n{r.stdout[-1500:]}")
    return float(m.group(1))


def read_xyz_atoms(path):
    """Read an xyz file into an ASE Atoms object."""
    lines = open(path).read().splitlines()
    n = int(lines[0])
    syms, pos = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        syms.append(p[0])
        pos.append([float(p[1]), float(p[2]), float(p[3])])
    return Atoms(symbols=syms, positions=pos)


def uma_gas_energy_kJ(xyz_path, chg):
    """UMA OMol25 gas-phase electronic energy (kJ/mol). spin=1 (closed shell)."""
    atoms = read_xyz_atoms(xyz_path)
    res = uma_helper.single_point(atoms, charge=int(chg), spin=1,
                                  model_name=UMA_MODEL)
    return float(res["energy"]) * EV_TO_KJ


def main():
    mets = {m.cpd_id: m for m in load_metabolites()}
    refs = reactions_within(set(mets))
    need = set()
    for rid in RXNS:
        need |= {c for c in refs[rid].reaction.compounds() if c != "cpd00067"}

    print(f"=== UMA composite | model={UMA_MODEL} | {len(need)} compounds ===")
    G_aq, breakdown = {}, {}
    for c in sorted(need):
        chg = mets[c].charge
        geom = os.path.join(THERMO, "results", "geometries", c)
        conf = sorted(f for f in os.listdir(geom) if f.endswith(".xyz"))[0]
        geom_abs = os.path.join(geom, conf)
        wd = os.path.join(WORK, c)

        xtbopt, e_alpb, g_rrho = xtb_optimise_and_thermal(geom_abs, chg, wd)
        e_gas_xtb = xtb_sp(xtbopt, chg, os.path.join(wd, "gas"), "")
        dgsolv = e_alpb - e_gas_xtb                          # ALPB solvation, Eh

        e_uma_kJ = uma_gas_energy_kJ(xtbopt, chg)            # UMA gas electronic
        dgsolv_kJ = dgsolv * HARTREE_TO_KJ
        grrho_kJ = g_rrho * HARTREE_TO_KJ
        g_kJ = e_uma_kJ + dgsolv_kJ + grrho_kJ
        G_aq[c] = g_kJ
        breakdown[c] = dict(charge=chg, E_UMA_kJ=e_uma_kJ,
                            dGsolv_xtbALPB_kJ=dgsolv_kJ, G_RRHO_kJ=grrho_kJ,
                            G_aq_kJ=g_kJ)
        print(f"{c} q={chg:+d}: E_UMA={e_uma_kJ:12.1f}  dGsolv={dgsolv_kJ:7.1f}  "
              f"G_RRHO={grrho_kJ:6.1f}  -> G_aq={g_kJ:12.1f} kJ/mol")

    os.makedirs(os.path.join(THERMO, "uma_workflow"), exist_ok=True)
    json.dump(breakdown, open(os.path.join(THERMO, "uma_workflow",
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

    bench_dir = os.path.join(THERMO, "results", "benchmark")
    xtb_v = json.load(open(os.path.join(bench_dir, "xtb_reaction_dG.json")))
    aim_v = json.load(open(os.path.join(bench_dir, "aimnet2_reaction_dG.json")))

    print(f"\n{'reaction':10s} {'exp':>6s} {'DFT':>7s} {'xTB':>7s} "
          f"{'AIMNet2':>8s} {'UMA':>8s}")
    out_rows = {}
    uerr, derr, aerr, xerr = [], [], [], []
    for rid in RXNS:
        ref = refs[rid].reaction
        uma = reaction_dG(ref, G_aq, species, conditions=config.DEFAULT_CONDITIONS)
        dftv = reaction_dG(ref, dft, species, conditions=config.DEFAULT_CONDITIONS)
        out_rows[rid] = uma.dG_transformed_kJ
        e = exp.get(rid)
        if e is not None:
            uerr.append(uma.dG_transformed_kJ - e)
            derr.append(dftv.dG_transformed_kJ - e)
            aerr.append(aim_v[rid] - e)
            xerr.append(xtb_v[rid] - e)
        es = f"{e:6.1f}" if e is not None else "   n/a"
        print(f"{rid:10s} {es} {dftv.dG_transformed_kJ:7.1f} {xtb_v[rid]:7.1f} "
              f"{aim_v[rid]:8.1f} {uma.dG_transformed_kJ:8.1f}")

    json.dump(out_rows, open(os.path.join(bench_dir, "uma_reaction_dG.json"),
                             "w"), indent=2)

    mae = lambda v: sum(abs(x) for x in v) / len(v)
    print(f"\nMAE vs experiment (n={len(uerr)}, same xtb-ALPB solvation+thermal):")
    print(f"   DFT(self-consistent) = {mae(derr):.1f}")
    print(f"   GFN2-xTB             = {mae(xerr):.1f}")
    print(f"   AIMNet2-composite    = {mae(aerr):.1f}")
    print(f"   UMA-composite        = {mae(uerr):.1f}   kJ/mol")
    print("\nwrote", os.path.join(bench_dir, "uma_reaction_dG.json"))


if __name__ == "__main__":
    main()
