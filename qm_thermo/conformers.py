"""Conformer ensemble generation and cheap (xtb) screening.

Pipeline stage 2: take a `Metabolite`, embed an ETKDG ensemble, MMFF-prune it,
optimise every conformer with GFN2-xTB in implicit water (ALPB), then return the
lowest-energy conformers within an energy window. Only these go on to expensive
DFT, which is what makes the rigorous protocol affordable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

from rdkit.Chem import AllChem

from . import config
from .geometry import Geometry, from_rdkit_conformer, read_xyz, write_xyz
from .structures import Metabolite

HARTREE_TO_KJ = 2625.499639


@dataclass(frozen=True)
class ScreenedConformer:
    """An xtb-optimised conformer with its GFN2 energy."""

    geometry: Geometry
    xtb_energy_hartree: float

    @property
    def rel_energy_kJ(self) -> float:  # filled in relative to the set minimum
        return self._rel_kJ

    _rel_kJ: float = 0.0


def _embed_ensemble(meta: Metabolite, settings: config.ConformerSettings) -> "AllChem.Mol":
    """ETKDG-embed and MMFF-minimise an ensemble; return the RDKit mol with confs."""
    mol = AllChem.Mol(meta.mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = settings.random_seed
    params.pruneRmsThresh = settings.rmsd_prune
    params.numThreads = 0  # use all available RDKit threads
    AllChem.EmbedMultipleConfs(mol, numConfs=settings.n_confs, params=params)
    if mol.GetNumConformers() == 0:
        # Fallback: random-coordinate embedding for awkward/strained systems.
        params.useRandomCoords = True
        AllChem.EmbedMultipleConfs(mol, numConfs=settings.n_confs, params=params)
    # MMFF pre-optimisation removes gross clashes before the (costlier) xtb step.
    AllChem.MMFFOptimizeMoleculeConfs(
        mol, maxIters=settings.mmff_max_iters, numThreads=0
    )
    return mol


def _run_xtb_opt(
    geom: Geometry, workdir: str, threads: int
) -> tuple[Geometry, float]:
    """Optimise a single geometry with GFN2-xTB in ALPB water. Returns (geom, E_h)."""
    os.makedirs(workdir, exist_ok=True)
    inp = os.path.join(workdir, "in.xyz")
    write_xyz(geom, inp)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_STACKSIZE"] = "4G"
    cmd = [
        config.XTB_BIN, "in.xyz",
        "--opt", "tight",
        "--gfn", "2",
        "--alpb", "water",
        "--chrg", str(geom.charge),
        "--uhf", str(geom.multiplicity - 1),
    ]
    with open(os.path.join(workdir, "xtb.out"), "w") as out:
        proc = subprocess.run(
            cmd, cwd=workdir, env=env, stdout=out, stderr=subprocess.STDOUT
        )
    if proc.returncode != 0:
        raise RuntimeError(f"xtb failed in {workdir} (rc={proc.returncode})")

    opt_xyz = os.path.join(workdir, "xtbopt.xyz")
    opt_geom = read_xyz(opt_xyz, geom.charge, geom.multiplicity)
    energy = _parse_xtb_energy(opt_xyz)
    return opt_geom, energy


def _parse_xtb_energy(xtbopt_xyz: str) -> float:
    """xtb writes the total energy in the comment line of xtbopt.xyz."""
    with open(xtbopt_xyz) as fh:
        comment = fh.read().splitlines()[1]
    # Format: " energy: -XX.XXXXXX gnorm: ... xtb: ..."
    tokens = comment.split()
    for i, tok in enumerate(tokens):
        if tok.lower().startswith("energy"):
            return float(tokens[i + 1])
    # Fallback: first float on the line.
    for tok in tokens:
        try:
            return float(tok)
        except ValueError:
            continue
    raise RuntimeError(f"could not parse xtb energy from {xtbopt_xyz}")


def generate_screened_conformers(
    meta: Metabolite,
    *,
    conf_settings: config.ConformerSettings = config.DEFAULT_CONFORMERS,
    parallel: config.ParallelSettings = config.DEFAULT_PARALLEL,
    scratch_root: str = config.SCRATCH_ROOT,
    keep_scratch: bool = False,
) -> list[ScreenedConformer]:
    """Return the lowest xtb-optimised conformers within the energy window.

    The number returned is capped at `conf_settings.max_qm_confs`.
    """
    mol = _embed_ensemble(meta, conf_settings)
    n_confs = mol.GetNumConformers()
    if n_confs == 0:
        raise RuntimeError(f"{meta.cpd_id}: RDKit failed to embed any conformer")

    os.makedirs(scratch_root, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"conf_{meta.cpd_id}_", dir=scratch_root)
    screened: list[ScreenedConformer] = []
    try:
        for conf_id in range(n_confs):
            geom = from_rdkit_conformer(
                mol, conf_id, meta.charge, meta.spin_multiplicity
            )
            cdir = os.path.join(work, f"conf{conf_id:03d}")
            try:
                opt_geom, energy = _run_xtb_opt(geom, cdir, parallel.xtb_threads)
            except RuntimeError as err:
                print(f"[conformers] {meta.cpd_id} conf {conf_id} xtb failed: {err}")
                continue
            screened.append(ScreenedConformer(opt_geom, energy))
    finally:
        if not keep_scratch:
            shutil.rmtree(work, ignore_errors=True)

    if not screened:
        raise RuntimeError(f"{meta.cpd_id}: all conformers failed xtb optimisation")

    return _select_window(screened, conf_settings)


def _select_window(
    screened: list[ScreenedConformer], settings: config.ConformerSettings
) -> list[ScreenedConformer]:
    """Sort, deduplicate near-degenerate energies, and apply the energy window."""
    screened.sort(key=lambda c: c.xtb_energy_hartree)
    e_min = screened[0].xtb_energy_hartree

    selected: list[ScreenedConformer] = []
    seen_energies: list[float] = []
    for conf in screened:
        rel_kJ = (conf.xtb_energy_hartree - e_min) * HARTREE_TO_KJ
        if rel_kJ > settings.energy_window_kJ:
            break
        # Drop conformers whose energy is within 0.1 kJ/mol of one already kept
        # (almost certainly the same minimum reached from a different seed).
        if any(abs(rel_kJ - s) < 0.1 for s in seen_energies):
            continue
        seen_energies.append(rel_kJ)
        selected.append(
            ScreenedConformer(conf.geometry, conf.xtb_energy_hartree, _rel_kJ=rel_kJ)
        )
        if len(selected) >= settings.max_qm_confs:
            break
    return selected
