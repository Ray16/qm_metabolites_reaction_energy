#!/usr/bin/env python
"""Step 5b: solvation-model sensitivity + COST on the REDOX reaction (rxn00070/86)
-- the regression test for COSMO. Redox was accurate with ALPB (cation+neutrals);
does COSMO keep it good (=> universal upgrade) or break it (=> per-charge-class)?

Reuses the step3b UMA geometries (artifacts/geom_redox/*.xyz) and gas free energies
(G_gas, includes thermal), varying ONLY the xtb solvation model on the same
geometries. Reaction (cysteine model): 2 CysSH + MNA+ -> CysSSCys + MNAH + H+.
  ΔG_aq = ΔG_gas + ΔΔGsolv(model) + G(H+,aq,pH7)
Also times ALPB vs COSMO (both are implicit-continuum SCF in xtb -- NOT COSMO-RS).

Run:  python scripts/step5b_redox_solv.py
"""
import json
import os
import re
import subprocess
import tempfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "artifacts")
GEOM = os.path.join(OUT, "geom_redox")
HARTREE2KJ = 2625.4996
XTB_BIN = os.environ.get("XTB_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb")
XTB_ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

# species: (xyz basename, charge)
SP = {"MNA+": ("MNAplus.xyz", 1), "MNAH": ("MNAH.xyz", 0),
      "CysSH": ("CysSH.xyz", 0), "CysSSCys": ("CysSSCys.xyz", 0)}
EXP = {"NAD": 18.0, "NADP": 11.9}


def xtb_E(xyz, q, model=None, timeit=False):
    cmd = [XTB_BIN, os.path.abspath(xyz), "--gfn", "2", "--chrg", str(int(q)), "--sp"]
    if model:
        cmd += ([f"--{model}", "water"] if model in ("alpb", "cosmo") else ["--gbsa", "water"])
    with tempfile.TemporaryDirectory() as d:
        t = time.time()
        r = subprocess.run(cmd, cwd=d, env=XTB_ENV, capture_output=True, text=True, timeout=180)
        dt = time.time() - t
    m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", r.stdout)
    E = float(m.group(1)) * HARTREE2KJ if m else None
    return (E, dt) if timeit else E


def main():
    d3b = json.load(open(os.path.join(OUT, "step3b_redox_thermal.json")))
    Gg = d3b["G_gas"]; gH = d3b["g_proton"]

    # gas free-energy part (fixed across solvation models)
    dG_gas = (Gg["CysSSCys"] + Gg["MNAH"]) - (2 * Gg["CysSH"] + Gg["MNA+"])

    print("=== redox ΔΔGsolv & ΔG by solvation model (cysteine model, thermal+CHE fixed) ===")
    print(f"  ΔG_gas {dG_gas:.1f}   G(H+) {gH:.1f}   exp NAD {EXP['NAD']} / NADP {EXP['NADP']}")
    for model in ("alpb", "gbsa", "cosmo"):
        solv = {}
        for n, (xyz, q) in SP.items():
            eg = xtb_E(os.path.join(GEOM, xyz), q, None)
            es = xtb_E(os.path.join(GEOM, xyz), q, model)
            solv[n] = es - eg
        dSolv = (solv["CysSSCys"] + solv["MNAH"]) - (2 * solv["CysSH"] + solv["MNA+"])
        dG = dG_gas + dSolv + gH
        print(f"  {model:6}: ΔΔGsolv {dSolv:7.1f}  ΔG {dG:6.1f}  "
              f"(err NAD {dG-EXP['NAD']:+.1f}, NADP {dG-EXP['NADP']:+.1f})   "
              f"solv[MNA+ {solv['MNA+']:.0f}, MNAH {solv['MNAH']:.0f}, "
              f"CysSH {solv['CysSH']:.0f}, CysSSCys {solv['CysSSCys']:.0f}]")

    print("\n=== COST: ALPB vs COSMO single point (both implicit SCF, not COSMO-RS) ===")
    for n in ("MNA+", "CysSSCys"):
        xyz, q = SP[n]
        _, t_a = xtb_E(os.path.join(GEOM, xyz), q, "alpb", timeit=True)
        _, t_c = xtb_E(os.path.join(GEOM, xyz), q, "cosmo", timeit=True)
        _, t_g = xtb_E(os.path.join(GEOM, xyz), q, None, timeit=True)
        print(f"  {n:9s}: gas {t_g:.2f}s   alpb {t_a:.2f}s   cosmo {t_c:.2f}s")


if __name__ == "__main__":
    main()
