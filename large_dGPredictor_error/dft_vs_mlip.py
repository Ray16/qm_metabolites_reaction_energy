#!/usr/bin/env python
"""Decisive test: is the electronic term the error source?

UMA and MACE-POLAR agree to r=0.98, but both are trained on OMol25, so that
measured architecture sensitivity, not accuracy. The electronic term is the only
component of the composite never compared against anything outside its own
training distribution -- and MLIP error is per-MOLECULE total energy, which
cancels between similar molecules but NOT across an atom-balanced set of
dissimilar ones. Four independent per-molecule errors of ~17 kJ/mol would give
sqrt(4)*17 ~ 34 kJ/mol of reaction scatter, which is what we observe.

Design: for each compound take the Boltzmann-DOMINANT conformer only, and form
reaction energies twice from the identical geometry, dGsolv and G_RRHO, swapping
only E_elec (MACE-POLAR vs r2SCAN-3c). Everything else cancels exactly, so the
difference is attributable to the electronic term alone.

Run:  /homes/rzhu/miniforge3/envs/palm/bin/python dft_vs_mlip.py [--nprocs 4] [--workers 7]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
sys.path.insert(0, THERMO)

ORCA = os.environ.get("ORCA_BIN",
    "/nfs/lambda_stor_01/homes/rzhu/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca")
MPI = os.environ.get("OPENMPI_ROOT", "/nfs/lambda_stor_01/homes/rzhu/openmpi-4.1.8-install")
HARTREE_TO_KJ = 2625.499639
# r2SCAN-3c: composite designed for exactly this -- reaction energies of medium
# organic molecules at a fraction of hybrid-DFT cost, with its own basis and
# dispersion/BSSE corrections built in.
KEYWORDS = "! r2SCAN-3c TightSCF"


def read_xyz(path):
    lines = open(path).read().splitlines()
    n = int(lines[0].split()[0])
    return [ln.split() for ln in lines[2:2 + n]]


def orca_energy_kJ(xyz_path, charge, nprocs):
    wd = tempfile.mkdtemp(prefix="orca_")
    try:
        atoms = read_xyz(xyz_path)
        with open(os.path.join(wd, "in.inp"), "w") as fh:
            fh.write(f"{KEYWORDS}\n%pal nprocs {nprocs} end\n"
                     f"%maxcore 3000\n* xyz {int(charge)} 1\n")
            for a in atoms:
                fh.write(f"{a[0]:<3s} {a[1]:>16s} {a[2]:>16s} {a[3]:>16s}\n")
            fh.write("*\n")
        env = {**os.environ,
               "PATH": f"{MPI}/bin:" + os.environ.get("PATH", ""),
               "LD_LIBRARY_PATH": f"{MPI}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")}
        r = subprocess.run([ORCA, "in.inp"], cwd=wd, capture_output=True,
                           text=True, env=env)
        m = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", r.stdout)
        return float(m[-1]) * HARTREE_TO_KJ if m else None
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nprocs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()

    sel = json.load(open(os.path.join(HERE, "dft_test_set.json")))
    bd = json.load(open(os.path.join(THERMO, "uma_workflow", "G_aq_bench226.json")))
    ens = json.load(open(os.path.join(HERE, "bench226_xtb.json")))
    spec = json.load(open(os.path.join(HERE, "bench226_species.json")))

    # Boltzmann-dominant conformer per compound
    dom = {}
    for c in sel["compounds"]:
        confs = bd[c]["conformers"]
        k = max(range(len(confs)), key=lambda i: confs[i]["weight"])
        dom[c] = dict(idx=k, xyz=ens[c][k]["xyz"], charge=spec[c]["charge"],
                      E_mlip=confs[k]["E_elec_kJ"],
                      dGsolv=confs[k]["dGsolv_kJ"], G_RRHO=confs[k]["G_RRHO_kJ"])

    print(f"=== r2SCAN-3c single points on {len(dom)} dominant conformers "
          f"| {args.workers} workers x {args.nprocs} procs ===", flush=True)
    out_path = os.path.join(HERE, "dft_energies.json")
    done = json.load(open(out_path)) if os.path.isfile(out_path) else {}
    todo = [c for c in dom if c not in done]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(orca_energy_kJ, dom[c]["xyz"], dom[c]["charge"],
                          args.nprocs): c for c in todo}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                done[c] = fut.result()
            except Exception as e:
                done[c] = None
                print(f"  [fail] {c}: {e}", flush=True)
            print(f"  {c} q={dom[c]['charge']:+d}  E_DFT={done[c]}", flush=True)
            json.dump(done, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}  ({sum(1 for v in done.values() if v)} / {len(dom)} ok)")


if __name__ == "__main__":
    main()
