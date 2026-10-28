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
from pathlib import Path
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
    try:
        lines = Path(gpath).read_text().splitlines()
        n = int(Path(xyz).read_text().splitlines()[0])
        if n <= 0:
            return None
        g = [[float(x.replace("D", "E").replace("d", "e")) for x in ln.split()[:3]]
             for ln in lines[2 + n:2 + 2 * n]]
        gradient = np.asarray(g, dtype=float)
    except (OSError, ValueError, IndexError):
        return None
    return gradient if gradient.shape == (n, 3) and np.isfinite(gradient).all() else None


def solvation_forces(symbols, coords, q, mult=1, model="alpb"):
    """-∇ΔG_solv (eV/Å, shape (n,3)) or None on xtb failure."""
    if model not in {"alpb", "cosmo", "gbsa"}:
        raise ValueError(f"No implemented xTB solvation gradient for {model!r}")
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
    A failed gradient is flagged and excluded by the caller. A frozen geometry
    reuses its last solvent force, so convergence is checked on the total force
    rather than accidentally reverting to the gas-only force.
    """
    failed = set()
    failures = {}
    cached = {}

    def extra(atoms_list, done):
        def one(i):
            atoms = atoms_list[i]
            symbols = tuple(atoms.get_chemical_symbols())
            positions = atoms.get_positions()
            previous = cached.get(i)
            if done[i] and previous is not None and previous[0] == symbols and np.array_equal(previous[1], positions):
                return previous[2].copy()
            f = solvation_forces(symbols, positions, q, mult, model)
            try:
                force = None if f is None else np.asarray(f, dtype=float)
            except (TypeError, ValueError):
                force = None
            if force is None or force.shape != (len(atoms), 3) or not np.isfinite(force).all():
                failed.add(i)
                failures[i] = "missing, malformed or nonfinite solvent gradient"
                cached.pop(i, None)
                return np.zeros((len(atoms), 3))
            cached[i] = (symbols, positions.copy(), force.copy())
            return force
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return np.concatenate(list(ex.map(one, range(len(atoms_list)))))
    extra.failed = failed
    extra.failures = failures
    return extra
