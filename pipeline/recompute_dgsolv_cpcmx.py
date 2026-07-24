#!/usr/bin/env python
"""Replace ALPB dGsolv with CPCM-X, per conformer, in parallel.

Why CPCM-X: it is parameterised for the GFN2-xTB Hamiltonian and is far better
for ions than ALPB, which was fitted predominantly on neutrals. Measured here on
acetate, CPCM-X stabilises the anion by an extra 40 kJ/mol relative to ALPB and
shifts the acetic-acid pKa by -8.4 units -- against a measured ALPB error of
+7.1 pKa units. So it removes the anion-solvation error from physics rather than
from a fitted per-group constant.

Why PER CONFORMER: the earlier openCOSMO-RS attempt failed not because COSMO-RS
is wrong but because it was affordable only on one dominant conformer while the
gas term stayed ensemble-averaged. That inconsistency injected +-50-150 kJ/mol of
conformer noise. CPCM-X costs ~6.6x an ALPB call (65 s vs 9.8 s on 74-atom
NADPH), which is cheap enough to run on every conformer and keep the ensemble
self-consistent. Embarrassingly parallel: one independent xtb process each.

The geometry is NOT re-optimised -- dGsolv is evaluated at the same ALPB-optimised
structure the gas energy and G_RRHO use, preserving geometry consistency across
all three terms of the composite.

Run:  /homes/rzhu/miniforge3/envs/palm/bin/python recompute_dgsolv_cpcmx.py \
          --ens ensemble_deep_xtb.json --out ensemble_deep_cpcmx.json
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
sys.path.insert(0, THERMO)

HARTREE_TO_KJ = 2625.499639
# build that actually has the CPCM-X library compiled in (the default xtb env
# advertises --cpcmx in --help but errors with "library was not included")
XTB = os.environ.get("XTB_CPCMX", "/homes/rzhu/miniforge3/envs/xtbcpx/bin/xtb")
SCRATCH = os.path.join("/tmp", "qm_thermo_scratch", "cpcmx")
_LOCK = threading.Lock()


def cpcmx_dgsolv_kJ(xyz_path, chg, omp=2):
    """CPCM-X solvation free energy at a fixed geometry, in kJ/mol.

    xtb prints the CPCM-X result to fort.6 in the working directory, not to
    stdout, so it must be read from there.
    """
    wd = tempfile.mkdtemp(prefix="cpx_", dir=SCRATCH)
    try:
        shutil.copy(xyz_path, os.path.join(wd, "in.xyz"))
        env = {**os.environ, "OMP_NUM_THREADS": str(omp), "OMP_STACKSIZE": "4G",
               "OPENBLAS_NUM_THREADS": "1"}
        subprocess.run([XTB, "in.xyz", "--gfn", "2", "--chrg", str(int(chg)),
                        "--uhf", "0", "--cpcmx", "water"],
                       cwd=wd, capture_output=True, text=True, env=env)
        f6 = os.path.join(wd, "fort.6")
        if not os.path.isfile(f6):
            return None
        m = re.search(r"solvation free energy \(dG_solv\):\s+(-?\d+\.\d+E[+-]\d+)",
                      open(f6, errors="replace").read())
        return float(m.group(1)) * HARTREE_TO_KJ if m else None
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens", default="ensemble_deep_xtb.json")
    ap.add_argument("--out", default="ensemble_deep_cpcmx.json")
    ap.add_argument("--species", default="species.json")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 16) // 2))
    ap.add_argument("--omp", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(SCRATCH, exist_ok=True)
    ens = json.load(open(os.path.join(HERE, args.ens)))
    spec = json.load(open(os.path.join(HERE, args.species)))
    out_path = os.path.join(HERE, args.out)
    done = json.load(open(out_path)) if os.path.isfile(out_path) else {}

    jobs = []
    for c, confs in ens.items():
        if c in done:
            continue
        chg = int(spec[c]["charge"])
        for i, cf in enumerate(confs):
            p = cf["xyz"] if os.path.isabs(cf["xyz"]) else os.path.join(HERE, cf["xyz"])
            jobs.append((c, i, p, chg))

    total = sum(len(v) for v in ens.values())
    print(f"=== CPCM-X dGsolv | {len(ens)} compounds / {total} conformers "
          f"| {len(jobs)} to run | {args.workers} workers x {args.omp} threads ===",
          flush=True)

    results = {c: [None] * len(v) for c, v in ens.items()}
    n_done = [0]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(cpcmx_dgsolv_kJ, p, q, args.omp): (c, i)
                for c, i, p, q in jobs}
        for fut in as_completed(futs):
            c, i = futs[fut]
            try:
                results[c][i] = fut.result()
            except Exception:
                results[c][i] = None
            with _LOCK:
                n_done[0] += 1
                if n_done[0] % 50 == 0:
                    print(f"  {n_done[0]}/{len(jobs)}", flush=True)

    new = {}
    for c, confs in ens.items():
        recs, nfail = [], 0
        for cf, dg in zip(confs, results[c]):
            r = dict(cf)
            if dg is None:                       # keep ALPB, flag it
                nfail += 1
                r["dGsolv_source"] = "ALPB (CPCM-X failed)"
            else:
                r["dGsolv_alpb_kJ"] = cf["dGsolv_kJ"]
                r["dGsolv_kJ"] = dg
                r["dGsolv_source"] = "CPCM-X"
            recs.append(r)
        new[c] = recs
        ok = [r for r in recs if r.get("dGsolv_source") == "CPCM-X"]
        if ok:
            d = sum(r["dGsolv_kJ"] - r["dGsolv_alpb_kJ"] for r in ok) / len(ok)
            print(f"{c:10} q={spec[c]['charge']:+d} {len(ok):3d}/{len(recs):3d} ok  "
                  f"mean(CPCM-X - ALPB) = {d:+8.1f} kJ/mol  ({spec[c]['name']})")
        if nfail:
            print(f"           [warn] {nfail} conformer(s) fell back to ALPB")

    json.dump(new, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
