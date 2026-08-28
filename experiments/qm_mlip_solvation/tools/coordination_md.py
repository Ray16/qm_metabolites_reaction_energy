"""Measure the solvation coordination number per site from explicit-water MLIP-MD — so QCT's inner-shell n
is DETERMINED per case, not guessed. Different chemistries coordinate differently, so we auto-detect sites
and use the chemistry-appropriate metric:
  - CATION site (Mg, metal, ammonium N+):  # water OXYGENS within the first Mg/N-O shell (acceptor coord)
  - ANIONIC O site (formal charge<0 O):    # water HYDROGENS within H-bond distance (waters DONATE to it)
Reports the mean coordination per site + its trajectory stability. Charged solute handled correctly
(unlike the neutral-only batched_replica_md). Short droplet MD (coordination is structural -> converges
fast, unlike a free energy).
"""
import os, sys, subprocess, tempfile, shutil, math, json
sys.path.insert(0, "scripts"); sys.path.insert(0, os.path.join("..", "..", "backup", "explicit_water"))
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

XTB = os.environ.get("XTB_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb")
SCRATCH = os.environ.get("QM_SCRATCH", "/nfs/lambda_stor_01/homes/rzhu/qm_scratch")
EV_KJ = 96.48533212; KB = 8.617333262e-5; FS = 0.09822694788; T = 300.0
NWAT = int(os.environ.get("NWAT", "30")); NREP = int(os.environ.get("NREP", "24"))  # batch: GPU not saturated at 6
EQ_PS = float(os.environ.get("EQ_PS", "1")); PROD_PS = float(os.environ.get("PROD_PS", "2"))
DT = float(os.environ.get("DT_FS", "2.0"))
# PHYSICAL SOUNDNESS: size the droplet wall to LIQUID density (~0.85 packing of 30 A^3/water + solute),
# else the waters spread into a gas-like droplet and the ion is under-solvated (WALL_R=8.5/28wat = 0.35!).
WALL_R = float(os.environ.get("WALL_R", "0")) or (3 * (NWAT * 30 + 60) / (4 * math.pi * 0.85)) ** (1.0 / 3.0)
CAT_OCUT = 2.9        # Å: metal/cation ... water-O first shell
HB_CUT = 2.5          # Å: anionic O ... water-H hydrogen bond
if os.environ.get("CASE_SMI"):                             # single-case via simple env (avoids JSON-over-ssh quoting)
    CASES = [[os.environ["CASE_NAME"], os.environ["CASE_SMI"], int(os.environ["CASE_Q"])]]
else:
    CASES = json.loads(os.environ.get("CASES", '[["Mg2+","[Mg+2]",2],["methylphosphate2-","COP(=O)([O-])[O-]",-2]]'))


def xtb_opt(smi, q):
    wd = tempfile.mkdtemp(dir=SCRATCH)
    try:
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        if m.GetNumAtoms() > 1:
            AllChem.EmbedMolecule(m, randomSeed=1); AllChem.MMFFOptimizeMolecule(m)
            Chem.MolToXYZFile(m, f"{wd}/in.xyz")
            subprocess.run([XTB, "in.xyz", "--gfn", "2", "--chrg", str(q), "--opt", "tight"], cwd=wd,
                           env={**os.environ, "OMP_NUM_THREADS": "1"}, capture_output=True, text=True, timeout=400)
            g = open(f"{wd}/xtbopt.xyz").read()
            sym = [l.split()[0] for l in g.splitlines()[2:] if l.strip()]
            crd = np.array([[float(x) for x in l.split()[1:4]] for l in g.splitlines()[2:] if l.strip()])
        else:
            sym = [m.GetAtomWithIdx(0).GetSymbol()]; crd = np.zeros((1, 3))
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    return sym, crd


def sites(smi, q):
    """Coordinating sites as (label, kind, [idxs]). Chemistry-correct detection (rdkit AddHs order == xtb):
      'metal'    -> count water O near the metal          (Mg/Ca/... acceptor coordination)
      'anionO'   -> count water H near this O             (one per TERMINAL O of a -COO/-SO3/-PO3 group,
                    incl. the resonance =O, not just the formal [O-]; excludes ester O and OH)
      'cationNH' -> count DISTINCT water O near the cation's N-H hydrogens (ammonium/guanidinium DONORS)"""
    m = Chem.AddHs(Chem.MolFromSmiles(smi)); out = []
    metals = {"Mg", "Ca", "Zn", "Mn", "Fe", "Na", "K", "Li"}
    for a in m.GetAtoms():
        if a.GetSymbol() in metals and a.GetFormalCharge() > 0:
            out.append((f"{a.GetSymbol()}#{a.GetIdx()}", "metal", [a.GetIdx()]))
    # anionic acceptor oxygens: every terminal O on a C/S/P center that bears a formal [O-] (captures all
    # resonance-equivalent oxygens of carboxylate/sulfonate/phosphate, the bug the panel exposed)
    centers = set()
    for a in m.GetAtoms():
        if a.GetSymbol() == "O" and a.GetFormalCharge() < 0:
            for nb in a.GetNeighbors():
                if nb.GetSymbol() in ("C", "S", "P"):
                    centers.add(nb.GetIdx())
    for ci in centers:
        for nb in m.GetAtomWithIdx(ci).GetNeighbors():
            if nb.GetSymbol() != "O":
                continue
            heavy = [x for x in nb.GetNeighbors() if x.GetSymbol() != "H"]
            hasH = any(x.GetSymbol() == "H" for x in nb.GetNeighbors())
            if len(heavy) == 1 and not hasH:               # terminal O (=O or O-), not ester/OH
                out.append((f"O#{nb.GetIdx()}", "anionO", [nb.GetIdx()]))
    # organic cation (net +, no metal): waters accept H-bonds from ALL N-H donors (ammonium/guanidinium)
    if q > 0 and not any(k == "metal" for _, k, _ in out):
        donorH = [nb.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "N"
                  for nb in a.GetNeighbors() if nb.GetSymbol() == "H"]
        if donorH:
            out.append(("N-H donors", "cationNH", donorH))
    if not out:                                            # bare monatomic ion
        out = [(f"{m.GetAtomWithIdx(0).GetSymbol()}#0", "metal" if q > 0 else "anionO", [0])]
    # dedup
    seen = set(); dd = []
    for lab, k, idx in out:
        key = (k, tuple(idx))
        if key not in seen:
            seen.add(key); dd.append((lab, k, idx))
    return dd


def masses(syms):
    from ase.data import atomic_masses, atomic_numbers
    mm = np.array([atomic_masses[atomic_numbers[s]] for s in syms])
    return np.where(mm < 2.0, 3.0, mm)


def build_droplet(sym0, crd0, nwat, wall_r, rng):
    """CLASH-FREE droplet: random-sequential-addition of water O's throughout the sphere with min O-O
    2.7 A and min O-solute 2.5 A (avoids the overlap that blew up the naive first-shell seeding). H's
    placed tetrahedrally; the relaxation + MD sort out orientation."""
    sol = crd0 - crd0.mean(0)
    heavy = [i for i in range(len(sym0)) if sym0[i] != "H"]
    Opos = []
    tries = 0
    while len(Opos) < nwat and tries < nwat * 4000:
        tries += 1
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        p = u * (wall_r * rng.random() ** (1.0 / 3.0))            # uniform in volume
        if any(np.linalg.norm(p - o) < 2.7 for o in Opos):
            continue
        if any(np.linalg.norm(p - sol[i]) < 2.5 for i in heavy):
            continue
        Opos.append(p)
    sym = list(sym0); crd = [c for c in sol]
    for p in Opos:
        a = rng.normal(size=3); a /= np.linalg.norm(a)
        b = np.cross(a, rng.normal(size=3)); b /= (np.linalg.norm(b) or 1.0)
        sym += ["O", "H", "H"]
        crd += [p, p + 0.96 * a, p + 0.96 * (-0.33 * a + 0.94 * b)]
    return sym, np.array(crd), len(Opos)


def run(pu, name, smi, q):
    from ase import Atoms
    from batched_relax import batched_fire, _predict
    import grand_canonical_clusters as gc
    sym0, crd0 = xtb_opt(smi, q); nsol = len(sym0)
    st = sites(smi, q)
    clus = []; nplaced = NWAT
    for r in range(NREP):
        cs, cc, npl = build_droplet(list(sym0), crd0, NWAT, WALL_R, np.random.default_rng(r + 1))
        nplaced = min(nplaced, npl)
        clus.append(Atoms(symbols=cs, positions=cc, info={"charge": int(q), "spin": 1}))
    syms = clus[0].get_chemical_symbols(); nat = len(syms)
    Oi = [i for i in range(nsol, nat) if syms[i] == "O"]; Hi = [i for i in range(nsol, nat) if syms[i] == "H"]
    if nplaced < NWAT:
        print(f"  [{name}] packed {nplaced}/{NWAT} waters", flush=True)
    # TIGHT relaxation so initial forces are small (loose relax + clashes was the blowup cause)
    rel, _, _ = batched_fire(pu, clus, fmax=0.05, steps=400, return_converged=True)
    P = np.stack([a.get_positions() for a in rel]); m = masses(syms)[None, :, None]
    rng = np.random.default_rng(0); V = rng.normal(size=P.shape) * np.sqrt(KB * T / m)
    dt = DT * FS; gamma = 0.1 / FS; c1 = math.exp(-gamma * dt); c2 = math.sqrt(1 - c1 * c1)  # strong thermostat

    # pre-build Atoms objects ONCE; per step only update positions (cuts per-step object/overhead churn)
    _A = [Atoms(symbols=syms, positions=P[i], info={"charge": int(q), "spin": 1}) for i in range(NREP)]

    def forces(P):
        for i in range(NREP):
            _A[i].set_positions(P[i])
        _, F, bi = _predict(pu, _A); F = F.detach().cpu().numpy(); bi = bi.detach().cpu().numpy()
        return np.stack([F[bi == i] for i in range(NREP)])

    F = forces(P); neq = int(EQ_PS * 1000 / DT); nprod = int(PROD_PS * 1000 / DT)
    dens = (NWAT * 30 + 60) / (4 / 3 * math.pi * WALL_R ** 3)
    print(f"  [{name}] MD {neq+nprod} steps x {DT}fs, {NREP} rep x {nat} atoms, {len(st)} sites; "
          f"WALL_R={WALL_R:.1f}A density={dens:.2f} ({'~liquid' if dens>0.7 else 'UNDER-DENSE'})", flush=True)
    coord = {lab: [] for lab, _, _ in st}
    import time as _time; _t0 = _time.time()
    for s in range(neq + nprod):
        V = V + 0.5 * dt * F / m; P = P + 0.5 * dt * V
        V = c1 * V + c2 * np.sqrt(KB * T / m) * rng.normal(size=V.shape); P = P + 0.5 * dt * V
        d = P - P.mean(1, keepdims=True); rr = np.linalg.norm(d, axis=2, keepdims=True)
        P = np.where(rr > WALL_R, P - 0.1 * (rr - WALL_R) * d / (rr + 1e-6), P)
        F = forces(P); V = V + 0.5 * dt * F / m
        if s % 100 == 0:                                   # SOUNDNESS monitor: temperature + blowup guard
            if not np.isfinite(P).all() or not np.isfinite(F).all():
                print(f"  [{name}] !! non-finite at step {s} -> UNSOUND", flush=True); break
            KE = 0.5 * np.sum(m[0] * V ** 2, axis=(1, 2))  # sum over atoms+xyz -> per-replica KE (eV; KB eV/K)
            Tinst = float(np.mean(2 * KE / (3 * nat * KB)))
            rate = s / (_time.time() - _t0 + 1e-9)
            print(f"    [{name}] step {s}/{neq+nprod}  T={Tinst:.0f}K  {rate:.1f} steps/s "
                  f"({rate*NREP*DT/1000:.1f} ps/s aggregate over {NREP} rep)", flush=True)
            if Tinst > 600:
                print(f"  [{name}] !! T runaway ({Tinst:.0f}K) -> UNSOUND (timestep too large)", flush=True); break
        if s >= neq and s % 10 == 0:
            for lab, kind, idxs in st:
                cc = 0
                for i in range(NREP):
                    if kind == "metal":
                        cc += int(np.sum(np.linalg.norm(P[i][Oi] - P[i][idxs[0]], axis=1) < CAT_OCUT))
                    elif kind == "anionO":
                        cc += int(np.sum(np.linalg.norm(P[i][Hi] - P[i][idxs[0]], axis=1) < HB_CUT))
                    elif kind == "cationNH":            # distinct water O H-bonded to any N-H donor
                        wset = set()
                        for h in idxs:
                            near = np.where(np.linalg.norm(P[i][Oi] - P[i][h], axis=1) < HB_CUT)[0]
                            wset.update(near.tolist())
                        cc += len(wset)
                coord[lab].append(cc / NREP)
    return name, st, syms, {lab: (float(np.mean(v)) if v else float("nan"),
                                  float(np.std(v)) if v else float("nan")) for lab, v in coord.items()}


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    from batched_relax import _ensure_registered, load_uma
    _ensure_registered("uma-s-1p2p1"); pu = load_uma("uma-s-1p2p1")
    print(f"{'case':22s} {'site':>14s} {'coordination (n)':>18s}", flush=True)
    out = {}
    for name, smi, q in CASES:
        nm, st, syms, res = run(pu, name, smi, q)
        tot = 0.0
        for (lab, kind, idxs) in st:
            mean, sd = res[lab]; tot += (mean if mean == mean else 0.0)
            print(f"{nm:22s} {lab+'('+kind+')':>18s} {mean:8.1f} ± {sd:.1f}")
        print(f"{'':22s} {'TOTAL inner-shell waters':>18s} {tot:8.1f}\n")
        out[nm] = {"sites": [(lab, kind, res[lab][0]) for lab, kind, idxs in st], "total": tot}
    json.dump(out, open("artifacts/coordination_md.json", "w"), indent=1)
    print("=> QCT inner-shell n is now MEASURED per case (cation O-coord vs anion H-bond coord), not guessed.")


if __name__ == "__main__":
    main()
