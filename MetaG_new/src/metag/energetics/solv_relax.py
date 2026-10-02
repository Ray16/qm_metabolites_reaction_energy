"""Relaxation on the AQUEOUS free-energy surface: G_aq(r) = E_UMA(r) + ΔG_solv,xtb(r).

The default species path relaxes conformers in the GAS phase and adds a vertical continuum solvation
single point. Gas relaxation collapses polar groups onto each other (carboxylate/phosphate O onto O-H,
P-OH...O=P networks in the neutral pH-0 polyphosphates) into compact, intramolecularly H-bonded
geometries that are under-solvated in water; the vertical solvation then cannot recover the geometry
the solute adopts in solution. Minimising E_UMA + ΔG_solv instead gives the solution-phase minimum.

The solvation gradient is the difference of two GFN2-xTB gradients at the same geometry,
∇ΔG_solv = ∇E_xtb(model) - ∇E_xtb(gas), consistent with how ΔG_solv itself is evaluated (thermal.dgsolv:
E_xtb(model) - E_xtb(gas)). Only ALPB/COSMO/GBSA have analytic gradients in xtb; CPCM-X does not."""
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from metag.energetics.thermal import XTB, ENV, _write_xyz

HB2EVA = 27.211386245988 / 0.529177210903      # Hartree/Bohr -> eV/Å


def _xtb_grad(d, xyz, q, uhf, flag):
    gpath = os.path.join(d, "gradient")
    if os.path.exists(gpath):
        os.remove(gpath)
    cmd = [XTB, xyz, "--gfn", "2", "--chrg", str(int(q)), "--uhf", str(int(uhf)), "--grad"] + flag
    try:
        r = subprocess.run(cmd, cwd=d, env=ENV, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0 or not os.path.exists(gpath):
        return None
    lines = open(gpath).read().split("\n")
    n = int(open(xyz).readline())
    g = [[float(x.replace("D", "E")) for x in ln.split()[:3]] for ln in lines[2 + n:2 + 2 * n]]
    return np.asarray(g, float) if len(g) == n else None


def solvation_forces(symbols, coords, q, mult=1, model="alpb"):
    """-∇ΔG_solv (eV/Å, shape (n,3)) or None on xtb failure."""
    flag = ["--gbsa", "water"] if model == "gbsa" else [f"--{model}", "water"]
    with tempfile.TemporaryDirectory() as d:
        xyz = os.path.join(d, "m.xyz")
        _write_xyz(xyz, symbols, coords)
        gg = _xtb_grad(d, xyz, q, mult - 1, [])
        gs = _xtb_grad(d, xyz, q, mult - 1, flag)
    if gg is None or gs is None:
        return None
    return -(gs - gg) * HB2EVA


def make_extra_forces(q, mult, model, workers=8):
    """Callable for uma.batched_fire(extra_forces=...): per-step solvation forces for the active structures.
    A structure whose xtb gradient fails gets zero extra force for that step (and is flagged)."""
    failed = set()

    def extra(atoms_list, done):
        def one(i):
            if done[i]:
                return np.zeros((len(atoms_list[i]), 3))
            f = solvation_forces(atoms_list[i].get_chemical_symbols(), atoms_list[i].get_positions(), q, mult, model)
            if f is None:
                failed.add(i)
                return np.zeros((len(atoms_list[i]), 3))
            return f
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return np.concatenate(list(ex.map(one, range(len(atoms_list)))))
    extra.failed = failed
    return extra
