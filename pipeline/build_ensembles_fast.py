#!/usr/bin/env python
"""Fast, DB-SCALABLE conformer ensemble (RDKit ETKDG + xtb), palm env, CPU only.

The scatter we are killing came from picking ONE arbitrary local minimum, not from
"no CREST" -- averaging over any reasonable ensemble fixes it. This builds that
ensemble ~100x cheaper than CREST so it scales to the whole ModelSEED DB, and it
is APPLES-TO-APPLES with the CREST pipeline (same composite: xtb-optimized
conformers + UMA electronic + xtb solvation/thermal) so Stage B and the CREST
result compare directly -- only the conformer SEARCH differs.

Two scaling levers vs the CREST pipeline:
  1. Conformer search = RDKit ETKDG (instant) instead of GFN2 metadynamics (min-h).
  2. Thermal G_RRHO computed ONCE per compound (lowest conformer) and shared --
     thermal corrections vary little across conformers and largely cancel in
     reactions, so per-conformer Hessians are wasted at scale.

Per compound:
  ETKDG(v3) embed + MMFF -> keep N lowest unique
    -> xtb --opt (GFN2, ALPB water) per conformer  [fast, no Hessian]
    -> dedupe by aqueous electronic energy, keep within window
    -> xtb --ohess on the lowest kept conformer -> G_RRHO (shared)
    -> xtb gas SP per conformer -> dGsolv

Writes (SAME schema as the CREST stage, so run_uma_ensemble.py reads it via
ENS_JSON=pipeline/ensemble_fast_xtb.json):
  geometries_ensemble_fast/{cpd}/conf_XXX.xyz
  ensemble_fast_xtb.json  {cpd:[{conf,xyz,dGsolv_kJ,G_RRHO_kJ,g_tot_kJ,
                                        rel_kJ,n_imag}]}

Resumable/checkpointed. Tunables via env: FAST_EMBED(48), FAST_NSTART(16),
FAST_WINDOW_KJ(20), FAST_JOBS(auto), FAST_XTB_OMP(4), FAST_OPT_WORKERS(auto).
Run (palm env):  python build_ensembles_fast.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
sys.path.insert(0, THERMO)
from qm_thermo import config                                  # noqa: E402
from qm_thermo.structures import load_metabolites             # noqa: E402

XTB = config.XTB_BIN
HARTREE_TO_KJ = 2625.499639
# Overridable so an extra species (a reference-reaction compound, a resolved
# anomer) can be built without editing the benchmark's own metabolite list.
MET_JSON = os.environ.get("FAST_MET_JSON", os.path.join(HERE, "metabolites.json"))
ENS_DIR = os.environ.get("FAST_ENS_DIR", os.path.join(HERE, "geometries_ensemble_fast"))
ENS_JSON = os.environ.get("FAST_ENS_JSON",
                          os.path.join(HERE, "ensemble_fast_xtb.json"))
SCRATCH = os.path.join("/tmp", "qm_thermo_scratch", "bench_ensembles_fast")

_NCPU = os.cpu_count() or 16
EMBED = int(os.environ.get("FAST_EMBED", "48"))
NSTART = int(os.environ.get("FAST_NSTART", "16"))
WINDOW_KJ = float(os.environ.get("FAST_WINDOW_KJ", "20"))
XTB_OMP = int(os.environ.get("FAST_XTB_OMP", "4"))
JOBS = int(os.environ.get("FAST_JOBS", str(max(1, (_NCPU - 8) // (XTB_OMP * 4)))))
OPT_WORKERS = int(os.environ.get("FAST_OPT_WORKERS", "4"))
REDO = os.environ.get("FAST_REDO", "0") == "1"
DEDUP_KJ = float(os.environ.get("FAST_DEDUP_KJ", "0.5"))
# Two conformers count as duplicates only if they are close in BOTH energy and
# geometry. Energy alone discards distinct-but-isoenergetic rotamers, which is
# exactly the undersampling that hurts the floppy phosphorylated metabolites.
DEDUP_RMSD = float(os.environ.get("FAST_DEDUP_RMSD", "0.5"))   # Angstrom, heavy atoms

_THREAD_ENV = {"OMP_STACKSIZE": "4G", "OPENBLAS_NUM_THREADS": "1"}
_JSON_LOCK = threading.Lock()
_PRINT_LOCK = threading.Lock()


def embed_mmff(meta):
    mol = Chem.Mol(meta.mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    params.pruneRmsThresh = 0.5
    params.numThreads = 0
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=EMBED, params=params))
    if not cids:
        params.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=EMBED, params=params))
    if not cids:
        raise RuntimeError(f"{meta.cpd_id}: RDKit embedded no conformers")
    if AllChem.MMFFHasAllMoleculeParams(mol):
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000, numThreads=0)
    else:
        res = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000, numThreads=0)
    order = sorted(((e, cid) for cid, (_ok, e) in zip(cids, res)), key=lambda t: t[0])
    return mol, [cid for _e, cid in order[:NSTART]]


def conf_to_atoms(mol, cid):
    conf = mol.GetConformer(cid)
    return [(a.GetSymbol(), *(lambda p: (p.x, p.y, p.z))(conf.GetAtomPosition(i)))
            for i, a in enumerate(mol.GetAtoms())]


def write_xyz(atoms, path, comment=""):
    lines = [str(len(atoms)), comment]
    for s, x, y, z in atoms:
        lines.append(f"{s:<3s} {x:>18.10f} {y:>18.10f} {z:>18.10f}")
    open(path, "w").write("\n".join(lines) + "\n")


def read_xyz_heavy(path):
    """Heavy-atom coordinates of an xyz file, in file order (xtb preserves it)."""
    lines = open(path).read().splitlines()
    xs = []
    for ln in lines[2:2 + int(lines[0].split()[0])]:
        p = ln.split()
        if p and p[0] != "H":
            xs.append([float(p[1]), float(p[2]), float(p[3])])
    return np.asarray(xs)


def kabsch_rmsd(a, b):
    """Minimal RMSD between two same-ordered coordinate sets (Angstrom)."""
    if a.shape != b.shape or a.size == 0:
        return float("inf")
    a = a - a.mean(0)
    b = b - b.mean(0)
    v, _s, wt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(v @ wt))
    r = v @ np.diag([1.0, 1.0, d]) @ wt
    return float(np.sqrt(((a @ r - b) ** 2).sum() / len(a)))


def _xtb(args, cwd):
    env = {**os.environ, **_THREAD_ENV, "OMP_NUM_THREADS": str(XTB_OMP)}
    return subprocess.run([XTB, *args], cwd=cwd, capture_output=True, text=True, env=env)


def _imag_modes(cwd):
    """(count of modes < -1 cm-1, most negative frequency in cm-1).

    The magnitude matters: a hindered methyl/OH rotor shows up at a few negative
    wavenumbers and is harmless, whereas a genuine saddle point sits at hundreds.
    Counting both alike rejects perfectly good structures, so callers screen on
    the magnitude, not the count.
    """
    vs = os.path.join(cwd, "vibspectrum")
    if not os.path.isfile(vs):
        return -1, 0.0
    n, worst = 0, 0.0
    for ln in open(vs):
        p = ln.split()
        for i, tok in enumerate(p):
            if tok == "a" and i + 1 < len(p):
                try:
                    f = float(p[i + 1])
                except ValueError:
                    break
                if f < -1.0:
                    n += 1
                    worst = min(worst, f)
                break
    return n, worst


def xtb_opt(atoms, chg, wd):
    """xtb --opt (ALPB) + gas SP. Returns dict with optimized geom, no Hessian."""
    os.makedirs(wd, exist_ok=True)
    write_xyz(atoms, os.path.join(wd, "in.xyz"), "etkdg conf")
    r = _xtb(["in.xyz", "--gfn", "2", "--alpb", "water", "--opt", "tight",
              "--chrg", str(chg), "--uhf", "0"], wd)
    e = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", r.stdout)
    if e is None or not os.path.isfile(os.path.join(wd, "xtbopt.xyz")):
        return None
    e_alpb = float(e.group(1))
    gas = os.path.join(wd, "gas")
    os.makedirs(gas, exist_ok=True)
    shutil.copy(os.path.join(wd, "xtbopt.xyz"), os.path.join(gas, "in.xyz"))
    rg = _xtb(["in.xyz", "--gfn", "2", "--chrg", str(chg), "--uhf", "0"], gas)
    eg = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", rg.stdout)
    if eg is None:
        return None
    return dict(xtbopt=os.path.join(wd, "xtbopt.xyz"), e_alpb_Eh=e_alpb,
                dGsolv_kJ=(e_alpb - float(eg.group(1))) * HARTREE_TO_KJ)


def xtb_ohess_thermal(xtbopt_xyz, chg, wd):
    """xtb --ohess on ONE conformer -> (G_RRHO_kJ, n_imag, worst_imag_cm)."""
    os.makedirs(wd, exist_ok=True)
    shutil.copy(xtbopt_xyz, os.path.join(wd, "in.xyz"))
    r = _xtb(["in.xyz", "--gfn", "2", "--alpb", "water", "--ohess",
              "--chrg", str(chg), "--uhf", "0"], wd)
    e = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", r.stdout)
    g = re.search(r"TOTAL FREE ENERGY\s+(-?\d+\.\d+)", r.stdout)
    if e is None or g is None:
        return 0.0, -1, 0.0
    n_imag, imag_cm = _imag_modes(wd)
    return (float(g.group(1)) - float(e.group(1))) * HARTREE_TO_KJ, n_imag, imag_cm


def process_compound(meta):
    cdir = os.path.join(ENS_DIR, meta.cpd_id)
    os.makedirs(cdir, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"fast_{meta.cpd_id}_", dir=SCRATCH)
    try:
        mol, cids = embed_mmff(meta)
        opted = []
        with ThreadPoolExecutor(max_workers=OPT_WORKERS) as ex:
            futs = [ex.submit(xtb_opt, conf_to_atoms(mol, cid), meta.charge,
                              os.path.join(work, f"c{k:03d}"))
                    for k, cid in enumerate(cids)]
            for fut in as_completed(futs):
                res = fut.result()
                if res is not None:
                    opted.append(res)
        if not opted:
            raise RuntimeError(f"{meta.cpd_id}: all conformers failed xtb --opt")

        opted.sort(key=lambda r: r["e_alpb_Eh"])
        emin = opted[0]["e_alpb_Eh"]
        kept = []
        for r in opted:
            rel = (r["e_alpb_Eh"] - emin) * HARTREE_TO_KJ
            if rel > WINDOW_KJ:
                break
            geom = read_xyz_heavy(r["xtbopt"])
            if any(abs(rel - k["rel_kJ"]) < DEDUP_KJ
                   and kabsch_rmsd(geom, k["geom"]) < DEDUP_RMSD for k in kept):
                continue
            r["rel_kJ"] = rel
            r["geom"] = geom
            kept.append(r)

        # One Hessian, on the lowest kept conformer; shared as G_RRHO for all.
        grrho, n_imag, imag_cm = xtb_ohess_thermal(kept[0]["xtbopt"], meta.charge,
                                                   os.path.join(work, "hess"))
        records = []
        for idx, r in enumerate(kept):
            dest = os.path.join(cdir, f"conf_{idx:03d}.xyz")
            shutil.copy(r["xtbopt"], dest)
            records.append(dict(conf=idx, xyz=dest, dGsolv_kJ=r["dGsolv_kJ"],
                                G_RRHO_kJ=grrho, g_tot_kJ=None, rel_kJ=r["rel_kJ"],
                                n_imag=n_imag if idx == 0 else -1,
                                imag_cm=imag_cm if idx == 0 else 0.0))
        return records
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _checkpoint(ensemble, cpd, kept):
    with _JSON_LOCK:
        latest = json.load(open(ENS_JSON)) if os.path.isfile(ENS_JSON) else {}
        latest[cpd] = kept
        ensemble[cpd] = kept
        tmp = ENS_JSON + ".tmp"
        json.dump(latest, open(tmp, "w"), indent=1)
        os.replace(tmp, ENS_JSON)


def _run_one(meta, ensemble):
    kept = process_compound(meta)
    _checkpoint(ensemble, meta.cpd_id, kept)
    with _PRINT_LOCK:
        print(f"  {meta.cpd_id:9s} {meta.name[:26]:26s} kept={len(kept):2d} conf "
              f"(spread {kept[-1]['rel_kJ']:.1f} kJ, G_RRHO shared, "
              f"lowest imag={kept[0]['imag_cm']:.0f}cm) [done]", flush=True)


def main():
    os.makedirs(ENS_DIR, exist_ok=True)
    os.makedirs(SCRATCH, exist_ok=True)
    mets = load_metabolites(MET_JSON)
    only = set(sys.argv[1:])
    if only:
        mets = [m for m in mets if m.cpd_id in only]
    ensemble = json.load(open(ENS_JSON)) if os.path.isfile(ENS_JSON) else {}
    todo = [m for m in mets if REDO or not ensemble.get(m.cpd_id)]
    print(f"=== FAST ETKDG+xtb ensembles | embed={EMBED} nstart={NSTART} "
          f"window={WINDOW_KJ} | {JOBS} compounds x ({OPT_WORKERS}x{XTB_OMP}thr) "
          f"on {_NCPU} cores | {len(todo)} to run, {len(mets)-len(todo)} cached ===",
          flush=True)
    todo.sort(key=lambda m: m.mol.GetNumAtoms(), reverse=True)

    if JOBS <= 1:
        for meta in todo:
            _run_one(meta, ensemble)
    else:
        with ThreadPoolExecutor(max_workers=JOBS) as ex:
            futs = {ex.submit(_run_one, meta, ensemble): meta for meta in todo}
            for fut in as_completed(futs):
                if fut.exception() is not None:
                    with _PRINT_LOCK:
                        print(f"  [ERROR] {futs[fut].cpd_id}: {fut.exception()}", flush=True)

    ensemble = json.load(open(ENS_JSON)) if os.path.isfile(ENS_JSON) else ensemble
    total = sum(len(v) for v in ensemble.values())
    print(f"\nwrote {ENS_JSON} ({len(ensemble)} compounds, {total} conformers total)")


if __name__ == "__main__":
    main()
