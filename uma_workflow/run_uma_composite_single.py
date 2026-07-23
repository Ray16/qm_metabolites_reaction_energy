#!/usr/bin/env python
"""UMA composite Delta_rG'^o for the collaborator's dGPredictor-vs-TECRDB top-10 disagreements.

Same physics as run_uma_composite.py -- only the metabolite set and reaction list
are swapped for the collaborator's (built by large_dGPredictor_error/build_inputs.py):

    xtb --ohess (ALPB water) -> optimized geom + E_alpb + G_RRHO(thermal)
    xtb SP gas               -> dGsolv = E_alpb - E_gas
    UMA (uma-s-1p2, OMol25)  -> E_gas electronic
    G_aq = E_UMA + dGsolv(xtb-ALPB) + G_RRHO(xtb)

Reaction Delta_rG'^o via the standard qm_thermo Alberty + Debye-Huckel transform,
directly comparable to the dGPredictor and TECRDB values in the collaborator's reactions CSV.

Handles the large cofactors ORCA DFT could not (NAD/NADP/NADPH/GSSG, 40-48 heavy
atoms): UMA + xtb scale to these sizes in minutes.

Run inside the `uma` env with a free GPU (this is the heavy job -- submit it on a
compute node):

    CUDA_VISIBLE_DEVICES=<n> /homes/rzhu/miniforge3/envs/uma/bin/python run_uma_composite_single.py

Writes:
    uma_workflow/G_aq.json                     (per-compound breakdown)
    results/benchmark/qm_reaction_dG.json      (rxn_id -> Delta_rG'^o kJ/mol)
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys

from ase import Atoms

THERMO = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc"
BENCH = os.path.join(THERMO, "large_dGPredictor_error")
sys.path.insert(0, THERMO)
sys.path.insert(0, "/homes/rzhu/uma_tools")
from qm_thermo import config                                  # noqa: E402
# NB: qm_thermo.reactions is now rdkit-free; the `uma` env has no rdkit, so per-species
# data (charge + H-count) is read from species.json instead of load_metabolites.
from qm_thermo.reactions import Reaction, reaction_dG, SpeciesInfo  # noqa: E402
import uma_helper                                             # noqa: E402

XTB = config.XTB_BIN
HARTREE_TO_KJ = 2625.499639
EV_TO_KJ = 96.48533212
UMA_MODEL = os.environ.get("UMA_MODEL", "uma-s-1p2")

SPECIES_JSON = os.path.join(BENCH, "species.json")
RXN_JSON = os.path.join(BENCH, "reactions.json")
RXN_CSV = os.path.join(BENCH, "top10_reactions_stereo_significant.csv")
GEOM_DIR = os.path.join(BENCH, "geometries")
WORK = os.path.join(THERMO, "uma_workflow", "scratch_bench")
os.makedirs(WORK, exist_ok=True)


def _xtb(args, cwd):
    return subprocess.run([XTB, *args], cwd=cwd, capture_output=True, text=True,
                          env={**os.environ, "OMP_NUM_THREADS": "8"})


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
    lines = open(path).read().splitlines()
    n = int(lines[0])
    syms, pos = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        syms.append(p[0])
        pos.append([float(p[1]), float(p[2]), float(p[3])])
    return Atoms(symbols=syms, positions=pos)


_CALC = None


def _uma_calc():
    """UMA OMol25 ASE calculator, loaded once and cached (get_calculator caches too)."""
    global _CALC
    if _CALC is None:
        _CALC = uma_helper.get_calculator("omol", UMA_MODEL)
    return _CALC


def uma_gas_energy_kJ(xyz_path, chg):
    """UMA OMol25 gas-phase electronic energy (kJ/mol). spin=1 (closed-shell singlet).

    FAIRChem reads net charge and spin multiplicity from atoms.info; energy is eV.
    """
    atoms = read_xyz_atoms(xyz_path)
    atoms.info["charge"] = int(chg)
    atoms.info["spin"] = 1
    atoms.calc = _uma_calc()
    return float(atoms.get_potential_energy()) * EV_TO_KJ


def load_reactions() -> dict[str, Reaction]:
    raw = json.load(open(RXN_JSON))
    return {rid: Reaction(rid, {c: float(v) for c, v in st.items()})
            for rid, st in raw.items()}


def load_reference_table():
    """dGPredictor + TECRDB per rxn from the collaborator's reactions CSV (kJ/mol)."""
    dgp, exp, name = {}, {}, {}
    for row in csv.DictReader(open(RXN_CSV)):
        rid = row["modelseed_rxn"]
        name[rid] = row["name"]
        dgp[rid] = float(row["dGpredictor_modelseed_dG_kJ"])
        exp[rid] = float(row["tecrdb_dG_kJ"])
    return dgp, exp, name


def main():
    spec = json.load(open(SPECIES_JSON))   # {cpd: {name, charge, n_hydrogens}}
    reactions = load_reactions()
    dgp, exp, name = load_reference_table()

    need = set()
    for rxn in reactions.values():
        need |= {c for c in rxn.compounds() if c != "cpd00067"}
    missing_rec = need - set(spec)
    if missing_rec:
        raise SystemExit(f"reactions reference compounds with no record: {sorted(missing_rec)}")

    print(f"=== UMA composite (benchmark set) | model={UMA_MODEL} | {len(need)} compounds ===")
    G_aq, breakdown = {}, {}
    for c in sorted(need):
        chg = int(spec[c]["charge"])
        geom_abs = os.path.join(GEOM_DIR, c, "conf_000.xyz")
        wd = os.path.join(WORK, c)

        xtbopt, e_alpb, g_rrho = xtb_optimise_and_thermal(geom_abs, chg, wd)
        e_gas_xtb = xtb_sp(xtbopt, chg, os.path.join(wd, "gas"), "")
        dgsolv = e_alpb - e_gas_xtb                          # ALPB solvation, Eh

        e_uma_kJ = uma_gas_energy_kJ(xtbopt, chg)            # UMA gas electronic
        dgsolv_kJ = dgsolv * HARTREE_TO_KJ
        grrho_kJ = g_rrho * HARTREE_TO_KJ
        g_kJ = e_uma_kJ + dgsolv_kJ + grrho_kJ
        G_aq[c] = g_kJ
        breakdown[c] = dict(name=spec[c]["name"], charge=chg, E_UMA_kJ=e_uma_kJ,
                            dGsolv_xtbALPB_kJ=dgsolv_kJ, G_RRHO_kJ=grrho_kJ,
                            G_aq_kJ=g_kJ)
        print(f"{c} q={chg:+d}: E_UMA={e_uma_kJ:12.1f}  dGsolv={dgsolv_kJ:7.1f}  "
              f"G_RRHO={grrho_kJ:6.1f}  -> G_aq={g_kJ:12.1f} kJ/mol  ({spec[c]['name']})")

    json.dump(breakdown, open(os.path.join(THERMO, "uma_workflow",
                                           "G_aq.json"), "w"), indent=2)

    species = {c: SpeciesInfo(c, n_hydrogens=int(spec[c]["n_hydrogens"]),
                              charge=int(spec[c]["charge"])) for c in G_aq}
    bench = os.path.join(THERMO, "results", "benchmark")
    out_rows = {}
    print(f"\n{'rxn':10s} {'exp':>8s} {'dGPredictor':>12s} {'QM(UMA)':>9s} "
          f"{'|QM-exp|':>9s} {'|dGP-exp|':>10s}  name")
    qm_ae, dgp_ae = [], []
    for rid, rxn in reactions.items():
        qm = reaction_dG(rxn, G_aq, species,
                         conditions=config.DEFAULT_CONDITIONS).dG_transformed_kJ
        out_rows[rid] = qm
        e = exp[rid]
        qm_ae.append(abs(qm - e)); dgp_ae.append(abs(dgp[rid] - e))
        print(f"{rid:10s} {e:8.1f} {dgp[rid]:12.1f} {qm:9.1f} "
              f"{abs(qm - e):9.1f} {abs(dgp[rid] - e):10.1f}  {name[rid]}")

    json.dump(out_rows, open(os.path.join(bench, "qm_reaction_dG.json"),
                             "w"), indent=2)

    mae = lambda v: sum(v) / len(v)
    print(f"\nMAE vs TECRDB experiment (n={len(qm_ae)}):")
    print(f"   dGPredictor (fine-tuned) = {mae(dgp_ae):6.1f}")
    print(f"   QM (UMA composite)       = {mae(qm_ae):6.1f}   kJ/mol")
    print("\nwrote", os.path.join(bench, "qm_reaction_dG.json"))
    print("wrote", os.path.join(THERMO, "uma_workflow", "G_aq.json"))


if __name__ == "__main__":
    main()
