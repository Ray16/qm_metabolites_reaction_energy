"""SMD solvation free energy via gpu4pyscf, callable from the pipeline (uma env, no gpu4pyscf).

The pipeline runs in the `uma` env which lacks gpu4pyscf, so — exactly as it already shells out to the
`xtb` binary for COSMO — this shells out to the `redox` env python (gpu4pyscf 1.8.1) for the SMD DFT
SCF. ΔGsolv = E_SMD(water) − E_gas at the SAME geometry. Content-hash cached on NFS so a species is
computed once across the fleet.

WHY SMD (not more xtb-COSMO): measured — xtb-COSMO under-solvates a CREATED compact polar/charged group
(e.g. a hydratase −OH: −3 kJ COSMO vs −17 SMD≈exp), which is the solvation-wall bias. SMD is calibrated
to neutral solvation. Used as a SOLUTE-ONLY correction ΔGsolv_SMD−ΔGsolv_COSMO (water stays on the
pipeline's water_ref_G), gated to reactions that create/destroy such a group -- see routing.solv_gate.

Run as CLI (in redox env):  python -m metag.energetics.smd_solv --xyz f.xyz --charge Q [--basis def2-svp]
"""
import argparse
import hashlib
import json
import os
import subprocess
import tempfile

HARTREE_KJ = 2625.499639
REDOX_PY = os.environ.get("SMD_PY", "/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/redox/bin/python")
_CACHE_DIR = os.environ.get("SMD_CACHE", os.path.join(os.path.dirname(__file__), "..", "..",
                                                      "analysis", "smd_measure_out", "smd_cache"))
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _key(symbols, coords, q, basis):
    h = hashlib.md5()
    h.update(f"{q}|{basis}".encode())
    for s, (x, y, z) in zip(symbols, coords):
        h.update(f"{s}{x:.3f}{y:.3f}{z:.3f}".encode())
    return h.hexdigest()


def _compute_smd(symbols, coords, q, basis, xc="PBE0"):
    """In-process SMD (only works where gpu4pyscf is importable, i.e. the CLI in the redox env)."""
    from pyscf import gto
    mol = gto.M(atom=[(s, tuple(map(float, p))) for s, p in zip(symbols, coords)],
                basis=basis, charge=int(q), spin=0, verbose=0)
    from gpu4pyscf import dft
    mf = dft.RKS(mol); mf.xc = xc
    e_gas = float(mf.kernel())
    ms = mf.SMD(); ms.with_solvent.solvent = "water"
    e_sol = float(ms.kernel())
    return (e_sol - e_gas) * HARTREE_KJ


def smd_dgsolv(symbols, coords, q, basis="def2-svp"):
    """ΔGsolv (kJ/mol) at the given geometry, cached. Returns None if the SMD subprocess fails
    (caller then skips the correction rather than crashing)."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    ck = os.path.join(_CACHE_DIR, _key(symbols, coords, q, basis) + ".json")
    if os.path.exists(ck):
        try:
            return json.load(open(ck))["dgsolv"]
        except Exception:
            pass
    # try in-process (redox env), else shell out to the redox python
    try:
        import gpu4pyscf  # noqa: F401
        val = _compute_smd(symbols, coords, q, basis)
    except Exception:
        with tempfile.TemporaryDirectory() as d:
            xyz = os.path.join(d, "m.xyz")
            with open(xyz, "w") as fh:
                fh.write(f"{len(symbols)}\n\n")
                for s, (x, y, z) in zip(symbols, coords):
                    fh.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")
            try:
                r = subprocess.run([REDOX_PY, "-m", "metag.energetics.smd_solv",
                                    "--xyz", xyz, "--charge", str(int(q)), "--basis", basis],
                                   cwd=_REPO, env={**os.environ, "PYTHONPATH": _REPO},
                                   capture_output=True, text=True, timeout=900)
            except Exception:
                return None
            val = None
            for line in r.stdout.splitlines():
                if line.startswith("DGSOLV "):
                    val = float(line.split()[1])
            if val is None:
                return None
    json.dump({"dgsolv": val}, open(ck, "w"))
    return val


def _read_xyz(path):
    lines = [l for l in open(path).read().splitlines()[2:] if l.strip()]
    sym = [l.split()[0] for l in lines]
    crd = [[float(x) for x in l.split()[1:4]] for l in lines]
    return sym, crd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--basis", default="def2-svp")
    a = ap.parse_args()
    sym, crd = _read_xyz(a.xyz)
    print(f"DGSOLV {_compute_smd(sym, crd, a.charge, a.basis):.6f}")


if __name__ == "__main__":
    main()
