#!/usr/bin/env python
"""Best-implicit-solvation test: openCOSMO-RS (native ORCA) for the composite.

Composite = AIMNet2 electronic (gas, wB97M) + openCOSMO-RS dGsolv + xtb thermal,
parallel to run_composite.py (xtb-ALPB) and improve_solvation.py (DFT-SMD), so the
three solvation models are directly comparable on the 5 TECRDB figure reactions.

AIMNet2 electronic energies and xtb G_RRHO are reused (computed earlier) to keep
this run to just the openCOSMO-RS calls.
"""
from __future__ import annotations
import json, os, re, subprocess, sys
THERMO = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc"
ORCA = "/nfs/lambda_stor_01/homes/rzhu/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca"
sys.path.insert(0, THERMO)
from qm_thermo import config
from qm_thermo.structures import load_metabolites
from qm_thermo.reactions import reaction_dG, species_info
from qm_thermo.references import reactions_within
HARTREE_TO_KJ = 2625.499639
SCR = "/tmp/qm_thermo_scratch/cosmors"
os.makedirs(SCR, exist_ok=True)
RXNS = ["rxn00830", "rxn00283", "rxn00191", "rxn00260", "rxn00276"]

# reused per-compound components (kJ/mol): AIMNet2 gas electronic, xtb G_RRHO
E_AIMNET = {"cpd00020": -898157.6, "cpd00023": -1447904.6, "cpd00024": -1495190.1,
            "cpd00032": -1391824.8, "cpd00033": -747204.9, "cpd00035": -850506.1,
            "cpd00040": -794841.9, "cpd00041": -1344582.7, "cpd00113": -3693404.8,
            "cpd00117": -850506.2, "cpd00202": -3693398.4}
G_RRHO = {"cpd00020": 67.1, "cpd00023": 260.0, "cpd00024": 128.7, "cpd00032": 63.6,
          "cpd00033": 126.8, "cpd00035": 193.6, "cpd00040": -0.7, "cpd00041": 189.2,
          "cpd00113": 318.1, "cpd00117": 194.0, "cpd00202": 315.2}


def cosmors_dgsolv(syms, xyz, chg, wd):
    os.makedirs(wd, exist_ok=True)
    lines = ["! r2SCAN-3c", "%maxcore 3000", "%pal nprocs 8 end",
             '%cosmors', '  solvent "water"', 'end', f"* xyz {chg} 1"]
    for s, c in zip(syms, xyz):
        lines.append(f"  {s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
    lines.append("*")
    open(os.path.join(wd, "c.inp"), "w").write("\n".join(lines) + "\n")
    out = subprocess.run([ORCA, os.path.join(wd, "c.inp")], cwd=wd,
                         capture_output=True, text=True)
    m = re.search(r"Free energy of solvation \(dGsolv\)\s*:\s*(-?\d+\.\d+)\s*Eh", out.stdout)
    return float(m.group(1)) if m else None


def read_xyz(path):
    L = open(path).read().splitlines(); n = int(L[0]); syms, xyz = [], []
    for ln in L[2:2 + n]:
        p = ln.split(); syms.append(p[0]); xyz.append([float(x) for x in p[1:4]])
    return syms, xyz


def main():
    mets = {m.cpd_id: m for m in load_metabolites()}
    refs = reactions_within(set(mets))
    need = set()
    for rid in RXNS:
        need |= {c for c in refs[rid].reaction.compounds() if c != "cpd00067"}

    G = {}
    for c in sorted(need):
        chg = mets[c].charge
        gdir = os.path.join(THERMO, "results", "geometries", c)
        geom = os.path.join(gdir, sorted(f for f in os.listdir(gdir) if f.endswith(".xyz"))[0])
        syms, xyz = read_xyz(geom)
        dg = cosmors_dgsolv(syms, xyz, chg, os.path.join(SCR, c))
        if dg is None:
            print(f"{c} q={chg:+d}: openCOSMO-RS FAILED"); continue
        G[c] = E_AIMNET[c] + dg * HARTREE_TO_KJ + G_RRHO[c]
        print(f"{c} q={chg:+d}: dGsolv(COSMO-RS)={dg*HARTREE_TO_KJ:7.1f}  -> G_aq={G[c]:12.1f}")

    species = {c: species_info(mets[c]) for c in G}
    import csv
    exp = {r["rxn_id"]: float(r["dG_exp_nearstd_kJ"])
           for r in csv.DictReader(open(os.path.join(THERMO, "results/benchmark/experimental_dG_TECRDB.csv")))
           if r["dG_exp_nearstd_kJ"]}
    v1 = json.load(open(os.path.join(THERMO, "results/benchmark/aimnet2_reaction_dG.json")))
    v2 = json.load(open(os.path.join(THERMO, "results/benchmark/aimnet2_smdsolv_reaction_dG.json")))
    print(f"\n{'reaction':10s} {'exp':>6s} {'xtb-ALPB':>9s} {'DFT-SMD':>8s} {'COSMO-RS':>9s}")
    ae = []
    for rid in RXNS:
        if not all(c in G for c in refs[rid].reaction.compounds() if c != "cpd00067"):
            continue
        new = reaction_dG(refs[rid].reaction, G, species, conditions=config.DEFAULT_CONDITIONS).dG_transformed_kJ
        e = exp.get(rid)
        if e is not None:
            ae.append(new - e)
        es = f"{e:6.1f}" if e is not None else "   n/a"
        print(f"{rid:10s} {es} {v1.get(rid, float('nan')):9.1f} {v2.get(rid, float('nan')):8.1f} {new:9.1f}")
    if ae:
        print(f"\nMAE vs exp (n={len(ae)}): COSMO-RS = {sum(abs(x) for x in ae)/len(ae):.1f}  "
              f"(xtb-ALPB 10.8, DFT-SMD 9.5, self-consistent DFT 5.5) kJ/mol")


if __name__ == "__main__":
    main()
