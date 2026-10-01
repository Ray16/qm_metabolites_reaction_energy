#!/usr/bin/env python
"""Conformer sampling: energy-targeted batched Boltzmann free energy over an ETKDG pool.

Answers "which conformers?" and "how many?": instead of a random ETKDG set, we
  1. generate a large ETKDG pool (--pool, default 128), MMFF pre-tidy,
  2. rank ALL of them by a batched UMA single-point (seconds; avoids MMFF's
     unreliable charged-phosphate energies),
  3. UMA-RELAX only the lowest --keep (default 24) — the Boltzmann-relevant set,
  4. per-conformer xTB-ALPB solvation (threaded),
  5. Boltzmann ensemble free energy.
Straggler-robust relaxation (stop_frac/straggler_fmax) so one slow conformer can't
stall the batch. Reports min vs Boltzmann ΔG per seed + reproducibility, and a
convergence sweep over --keep so we can see how few conformers actually suffice.

Run (uma env):
  CUDA_VISIBLE_DEVICES=0 python scripts/step4e_targeted.py --seeds 1,2,3,4,5 \
      --pool 128 --keep 24
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms

from metag.energetics.uma import load_uma, batched_fire, batched_energies

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "artifacts")
CACHE = os.path.join(OUT, "cache_conf")        # legacy step4e script only (created in main())
EV2KJ = 96.485
HARTREE2KJ = 2625.4996
# Energy half of the unique-minimum test (kJ). UMA total energies are float32, whose step at |E| ~ 1e5 eV
# (ATP / NAD / CoA) is 0.0078 eV = 0.75 kJ: the old 0.5 kJ tolerance was BELOW the resolution, so two
# relaxations into the same basin merged only if their energies rounded identically, and every unmerged
# copy added a spurious -RT ln N. The tolerance is now max(DEDUP_E_TOL, 2 float32 steps of E); the RMSD
# test remains the real identity criterion.
DEDUP_E_TOL = 1.5
KT = 2.4789
THERMAL_FIXED = -0.8
EXP = -4.2

SPECIES = {
    "MeUDPGlc": (-2, "OC[C@H]1O[C@@H](OP(=O)([O-])OP(=O)([O-])OC)[C@H](O)[C@@H](O)[C@@H]1O"),
    "Fructose": (0,  "OC[C@H]1OC(O)(CO)[C@@H](O)[C@@H]1O"),
    "MeUDP":    (-2, "COP(=O)([O-])OP(=O)([O-])O"),
    "Suc":      (0,  "OC[C@H]1O[C@@H](O[C@]2(CO)O[C@H](CO)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O"),
}


def boltz(Gs):
    """-RT ln Σ exp(-G_i/RT). Each entry is ONE thermodynamic state, so Gs must be UNIQUE minima
    (see UniqueMinima): N copies of the same basin lower the result by RT ln N (6.9 kJ at N=16)."""
    Gs = np.asarray(Gs); ref = Gs.min()
    return float(ref - KT * np.log(np.exp(-(Gs - ref) / KT).sum()))


def principal_moments(atoms):
    """Sorted principal moments of inertia (amu Å²). Invariant to rotation, translation AND atom
    permutation, so it identifies symmetry-equivalent copies of a minimum without an atom mapping."""
    m = atoms.get_masses()
    x = atoms.get_positions() - atoms.get_center_of_mass()
    I = (np.eye(3) * (m[:, None] * x * x).sum() - np.einsum("i,ij,ik->jk", m, x, x))
    return np.sort(np.linalg.eigvalsh(I))


class UniqueMinima:
    """Relaxed conformers deduplicated to UNIQUE minima before Boltzmann weighting.

    Different ETKDG seeds (and conformers within one pool) routinely relax into the SAME basin; counting
    each copy as a separate state adds a spurious degeneracy -RT ln(N_copies) that depends on how many
    samples were drawn, so it does not cancel across a reaction whose species stop at different sample
    counts.

    Criterion: same minimum iff |ΔE_UMA| < e_tol AND the all-atom, symmetry-aware RMSD after optimal
    PROPER rotation is < rmsd_tol (RDKit GetBestRMS over graph automorphisms: permuted methyl / phosphate
    / carboxylate atoms and rotated copies merge). Proper rotations only: mirror-image conformers of an
    achiral molecule (gauche+/gauche-) are distinct states and stay separate -- the rotational-constant
    shortcut used first merged them (34 pairs for citric acid), undercounting conformational entropy for
    achiral species only, a bias that does not cancel against chiral partners.

    `template` = the RDKit molecule (with H) whose atom order matches the Atoms (pool_confs builds from
    Chem.AddHs(MolFromSmiles(smi))). Without a template (explicit water clusters) it falls back to
    energy + principal moments (can merge mirror images; that path is off by default)."""

    def __init__(self, e_tol=DEDUP_E_TOL, rmsd_tol=0.25, template=None, rot_tol=0.01, max_matches=20000):
        self.e_tol, self.rmsd_tol, self.rot_tol = e_tol, rmsd_tol, rot_tol
        self.template, self.max_matches = template, max_matches
        self.E, self.G, self.ref = [], [], []
        self.n_seen = 0

    def _e_tol(self, e):
        """Energy tolerance at energy e (kJ): never below 2 float32 steps of the UMA energy in eV.
        A negative e_tol (legacy A/B: never merge) is returned unchanged."""
        if self.e_tol < 0:
            return self.e_tol
        return max(self.e_tol, 2.0 * float(np.spacing(np.float32(abs(e) / EV2KJ))) * EV2KJ)

    def _mol_at(self, atoms):
        from rdkit import Chem as _C
        from rdkit.Geometry import Point3D
        m = _C.Mol(self.template)
        m.RemoveAllConformers()
        conf = _C.Conformer(m.GetNumAtoms())
        for i, (x, y, z) in enumerate(atoms.get_positions()):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        m.AddConformer(conf, assignId=True)
        return m

    def _same(self, e, rep, j):
        if abs(e - self.E[j]) >= self._e_tol(e):
            return False
        if self.template is None:
            ref = np.maximum(np.maximum(np.abs(rep), np.abs(self.ref[j])), 1e-3)
            return bool(np.all(np.abs(rep - self.ref[j]) / ref < self.rot_tol))
        from rdkit.Chem import rdMolAlign
        return rdMolAlign.GetBestRMS(rep, self.ref[j], maxMatches=self.max_matches) < self.rmsd_tol

    def add(self, atoms, E_kJ, G_kJ):
        """Register a relaxed conformer; returns True if it is a NEW minimum. A duplicate keeps the
        representative with the lower G (its solvation may differ slightly by geometry noise)."""
        self.n_seen += 1
        rep = principal_moments(atoms) if self.template is None else self._mol_at(atoms)
        for j in range(len(self.E)):
            if self._same(E_kJ, rep, j):
                if G_kJ < self.G[j]:                      # the representative IS the structure whose E/G is
                    self.G[j] = float(G_kJ)               # used: geometry, E and G are replaced together
                    self.E[j] = float(E_kJ)
                    self.ref[j] = rep
                return False
        self.E.append(float(E_kJ)); self.G.append(float(G_kJ)); self.ref.append(rep)
        return True

    def drop(self, indices):
        """Remove the minima at `indices` (e.g. saddle points found by the thermal validation)."""
        keep = [j for j in range(len(self.G)) if j not in set(indices)]
        self.E = [self.E[j] for j in keep]; self.G = [self.G[j] for j in keep]; self.ref = [self.ref[j] for j in keep]
        return keep

    def __len__(self):
        return len(self.G)


def bond_graph(template):
    """Set of bonded atom-index pairs of an RDKit molecule WITH explicit H (atom order = pool_confs)."""
    return {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in template.GetBonds()}


def perceived_graph(atoms, scale=1.25):
    """Bonded pairs perceived from 3D geometry: d_ij < scale * (r_cov,i + r_cov,j). With scale 1.25 an
    O-H covalent bond (0.97 A) is < 1.21 A but a hydrogen bond (>= 1.5 A) is not; P-O (1.6 A) < 2.16 A."""
    pt = Chem.GetPeriodicTable()
    r = np.array([pt.GetRcovalent(int(z)) for z in atoms.get_atomic_numbers()])
    x = atoms.get_positions()
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    i, j = np.where(np.triu(d < scale * (r[:, None] + r[None, :]), k=1))
    return set(zip(i.tolist(), j.tolist()))


def same_connectivity(atoms, ref_graph):
    """False if relaxation changed the bonding (e.g. a zwitterion's N-H proton moved to its O-, or a
    phosphate proton hopped): the relaxed structure is then a DIFFERENT species from the SMILES scored."""
    return perceived_graph(atoms) == ref_graph


TRANSITION_METALS = set(range(21, 31)) | set(range(39, 49)) | set(range(57, 81))


def spin_multiplicity(smiles, q):
    """Ground-state spin multiplicity (2S+1) for the UMA `spin` field. Metabolites are
    overwhelmingly closed-shell singlets (1), but a few open-shell species must be set
    explicitly or UMA computes the wrong electronic state:
      * O2 -- ground state is a TRIPLET (mult 3), not the singlet a bare `O=O` SMILES implies.
        Computed as singlet it is ~115 kJ/mol too high (measured), driving every O2-consuming
        oxygenase/oxidase reaction that-many-times too negative. (The common, high-impact case.)
      * Any odd-electron species -> doublet (mult 2): NO, superoxide O2*-, organic radicals.
    Deterministic in (smiles, q), so it needs no separate cache key for singlets."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return 1
    m = Chem.AddHs(m)
    # O2: two oxygens, no H, neutral -> triplet ground state
    zs = [a.GetAtomicNum() for a in m.GetAtoms()]
    if any(z in TRANSITION_METALS for z in zs):
        raise ValueError(f"{smiles}: transition-metal species -- spin state cannot be inferred from the SMILES "
                         "and UMA/xtb states are not validated here; not supported")
    if int(q) == 0 and sorted(zs) == [8, 8]:
        return 3
    n_rad = sum(a.GetNumRadicalElectrons() for a in m.GetAtoms())
    if n_rad:                                             # SMILES-specified radicals ([CH2], [O][O], ...):
        return n_rad + 1                                  # high-spin (all unpaired electrons parallel)
    n_elec = sum(zs) - int(q)
    return 2 if (n_elec % 2) else 1                       # odd electrons -> doublet, else singlet


def pool_confs(smiles, q, seed, pool, spin=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3(); p.randomSeed = seed; p.pruneRmsThresh = 0.3
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=pool, params=p))
    if not cids:
        AllChem.EmbedMolecule(m, randomSeed=seed, useRandomCoords=True); cids = [0]
    try:
        AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=200)
    except Exception:
        pass
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    return [Atoms(symbols=syms, positions=m.GetConformer(c).GetPositions(),
                  info={"charge": int(q), "spin": int(spin)}) for c in cids]


# DIRECT xtb binary — NEVER `conda run` in a loop (~30 s overhead/call vs 0.57 s).
# See ../../CLAUDE.md. OMP=1 avoids CPU oversubscription when threaded/fanned-out.
# Solvation: COSMO by default (Step 5/5b: ALPB/GBSA under-solvate polyanions by
# ~24 kJ; COSMO fixes glycosyl +28→+0.4 AND preserves redox → universal upgrade).
XTB_BIN = os.environ.get("XTB_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb")
XTBCPX_BIN = os.environ.get("XTBCPX_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtbcpx/bin/xtb")
XTB_ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "OMP_STACKSIZE": "4G"}
SOLV_MODEL = os.environ.get("SOLV_MODEL", "cpcmx")   # cpcmx | cosmo | alpb | gbsa


def _cpcmx_dgsolv(atoms, q, d):
    """CPCM-X dG_solv (kJ) in one call via the xtbcpx build; reads fort.6."""
    xyz = os.path.join(d, "in.xyz")
    with open(xyz, "w") as f:
        f.write(f"{len(atoms)}\n\n")
        for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            f.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")
    subprocess.run([XTBCPX_BIN, "in.xyz", "--gfn", "2", "--chrg", str(int(q)),
                    "--uhf", "0", "--cpcmx", "water"],
                   cwd=d, env=XTB_ENV, capture_output=True, text=True, timeout=600)
    f6 = os.path.join(d, "fort.6")
    if not os.path.isfile(f6):
        return None
    m = re.search(r"solvation free energy \(dG_solv\):\s+(-?\d+\.\d+E[+-]\d+)",
                  open(f6, errors="replace").read())
    return float(m.group(1)) * HARTREE2KJ if m else None


def xtb_dgsolv(atoms, q, model=SOLV_MODEL):
    with tempfile.TemporaryDirectory() as d:
        if model == "cpcmx":
            return _cpcmx_dgsolv(atoms, q, d)
        xyz = os.path.join(d, "m.xyz")
        with open(xyz, "w") as f:
            f.write(f"{len(atoms)}\n\n")
            for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
                f.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")

        def e(solvated):
            cmd = [XTB_BIN, xyz, "--gfn", "2", "--chrg", str(int(q)), "--sp"]
            if solvated:
                cmd += (["--gbsa", "water"] if model == "gbsa" else [f"--{model}", "water"])
            r = subprocess.run(cmd, cwd=d, env=XTB_ENV, capture_output=True,
                               text=True, timeout=120)
            m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", r.stdout)
            return float(m.group(1)) if m else None
        eg, ew = e(False), e(True)
        return (ew - eg) * HARTREE2KJ if (eg is not None and ew is not None) else None


def species_conformers(pu, name, q, smi, seed, pool, keep, log):
    """pool -> UMA single-point rank -> relax lowest `keep` -> per-conformer (Eg,Gt)."""
    cands = pool_confs(smi, q, seed, pool)
    Esp = batched_energies(pu, cands)                     # rank by UMA single-point
    order = np.argsort(Esp)[:keep]
    sel = [cands[i] for i in order]
    _, E_ev, conv = batched_fire(pu, sel, fmax=0.05, steps=300, stop_frac=0.9,
                                 return_converged=True, verbose=True, log_every=100,
                                 label=f"{name}s{seed}")
    # DROP unconverged stragglers — a conformer that won't relax is a bad geometry
    # with an unreliable energy; better excluded than averaged into the Boltzmann sum.
    sel = [a for a, c in zip(sel, conv) if c]
    Eg = (E_ev[conv] * EV2KJ)
    ndrop = int((~conv).sum())
    with ThreadPoolExecutor(max_workers=8) as ex:
        ds = list(ex.map(lambda a: xtb_dgsolv(a, q), sel))
    Egv, Gt = [], []
    for e, d in zip(Eg, ds):
        if d is None or not np.isfinite(e):
            continue
        Egv.append(float(e)); Gt.append(float(e + d))
    log(f"    {name}: pool {len(cands)} -> relaxed {keep} -> dropped {ndrop} straggler(s) "
        f"-> {len(Gt)} solvated; gas-spread {max(Egv)-min(Egv):.0f} kJ")
    return dict(Eg=Egv, Gt=Gt)


def dG(agg, kind):
    def val(n):
        Gt = np.array(agg[n]["Gt"]); return float(Gt.min()) if kind == "min" else boltz(Gt)
    return (val("MeUDP") + val("Suc")) - (val("MeUDPGlc") + val("Fructose")) + THERMAL_FIXED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--pool", type=int, default=128)
    ap.add_argument("--keep", type=int, default=24)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    os.makedirs(CACHE, exist_ok=True)

    def log(m): print(m, flush=True)
    log(f"loading UMA...  seeds={seeds} pool={a.pool} keep={a.keep}")
    pu = load_uma()

    rows = []
    for seed in seeds:
        log(f"  seed {seed}:")
        agg = {n: species_conformers(pu, n, q, smi, seed, a.pool, a.keep, log)
               for n, (q, smi) in SPECIES.items()}
        json.dump(agg, open(os.path.join(CACHE, f"e_seed{seed}_k{a.keep}.json"), "w"))
        # convergence sub-sweep: Boltzmann using lowest-k subset of the kept set
        subs = {}
        for k in (10, 15, 20, a.keep):
            if k > a.keep:
                continue
            sub = {n: {"Gt": sorted(agg[n]["Gt"])[:k]} for n in SPECIES}
            subs[k] = dG(sub, "boltz")
        r = dict(seed=seed, dG_min=dG(agg, "min"), dG_boltz=dG(agg, "boltz"), boltz_by_k=subs)
        rows.append(r)
        log(f"  seed {seed}:  ΔG_min {r['dG_min']:7.1f}  ΔG_boltz {r['dG_boltz']:7.1f} kJ  "
            f"(by keep-k {subs})  exp {EXP}")

    mn = np.array([r["dG_min"] for r in rows]); bz = np.array([r["dG_boltz"] for r in rows])
    log(f"\n==== energy-TARGETED batched Boltzmann, rxn00579, {len(seeds)} seeds, "
        f"pool {a.pool} keep {a.keep} ====")
    log(f"  ΔG_min    mean {mn.mean():6.1f}  std {mn.std():5.1f}  range {mn.max()-mn.min():5.1f}  err {mn.mean()-EXP:+.1f}")
    log(f"  ΔG_boltz  mean {bz.mean():6.1f}  std {bz.std():5.1f}  range {bz.max()-bz.min():5.1f}  err {bz.mean()-EXP:+.1f}")
    log(f"  (baseline: Step-4b random-24 min-only std 12.5, range 37)")
    # convergence vs k (mean over seeds)
    for k in (10, 15, 20, a.keep):
        vals = [r["boltz_by_k"].get(str(k), r["boltz_by_k"].get(k)) for r in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            log(f"  Boltzmann@k={k:2d}: mean {np.mean(vals):6.1f}  std {np.std(vals):5.1f}")
    json.dump(dict(seeds=seeds, pool=a.pool, keep=a.keep, rows=rows, exp=EXP,
                   boltz=dict(mean=float(bz.mean()), std=float(bz.std()))),
              open(os.path.join(OUT, f"step4e_targeted_k{a.keep}.json"), "w"), indent=2)
    log(f"wrote artifacts/step4e_targeted_k{a.keep}.json")


if __name__ == "__main__":
    main()
