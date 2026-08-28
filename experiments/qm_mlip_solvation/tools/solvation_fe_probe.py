"""FEP go/no-go CONVERGENCE PROBE — the one test that decides whether the anion-wall FEP is worth building.

The prior explicit-cluster attempts for anions were NON-CONVERGENT (min-over-seeds selection bias). The
open question: does DYNAMICS (explicit-water droplet MD) give a solvation free energy that (a) CONVERGES
(block-averaged, small SE) and (b) reproduces a known ion ΔGhyd? If yes -> the FEP path is alive, build it.
If no -> the wall is beyond this method; flag it and move on. Charged solute handled correctly (unlike the
neutral-only batched_replica_md).

Method (cluster-continuum from MD snapshots, Bryantsev bulk-water reference):
  ΔGsolv(ion) = <E_UMA(cluster) + ΔGsolv_xtb-COSMO(cluster)>_MD  - n_w*G*_liq(H2O) - E_UMA(gas ion)
  sampled along a UMA Langevin droplet trajectory; CONVERGENCE = block averages over trajectory thirds +
  bootstrap SE. Validate on acetate- (relative to formate- to dodge the single-ion convention), whose
  experimental ΔΔGhyd(acetate-formate) ~ +32 kJ/mol is well established.
"""
import os, sys, json, subprocess, tempfile, shutil, math
sys.path.insert(0, "scripts"); sys.path.insert(0, os.path.join("..", "..", "backup", "explicit_water"))
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from thermal_solv import xtb_dgsolv

XTB = os.environ.get("XTB_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb")
SCRATCH = os.environ.get("QM_SCRATCH", "/nfs/lambda_stor_01/homes/rzhu/qm_scratch")
EV_KJ = 96.48533212; KB = 8.617333262e-5; FS = 0.09822694788; T = 298.15
RT = 8.314462618e-3 * T
NWAT = int(os.environ.get("NWAT", "30")); NREP = int(os.environ.get("NREP", "12"))
PS = float(os.environ.get("PS", "6")); DT = float(os.environ.get("DT_FS", "2.0"))
WALL_R = float(os.environ.get("WALL_R", "8.0"))
GWATER_EXP = -26.4
# (name, SMILES, charge)
IONS = [("acetate", "CC(=O)[O-]", -1), ("formate", "[O-]C=O", -1)]


def xtb_opt(smi, q):
    wd = tempfile.mkdtemp(dir=SCRATCH)
    try:
        m = Chem.AddHs(Chem.MolFromSmiles(smi)); AllChem.EmbedMolecule(m, randomSeed=1); AllChem.MMFFOptimizeMolecule(m)
        Chem.MolToXYZFile(m, f"{wd}/in.xyz")
        subprocess.run([XTB, "in.xyz", "--gfn", "2", "--chrg", str(q), "--opt", "tight"], cwd=wd,
                       env={**os.environ, "OMP_NUM_THREADS": "1"}, capture_output=True, text=True, timeout=400)
        g = open(f"{wd}/xtbopt.xyz").read()
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    sym = [l.split()[0] for l in g.splitlines()[2:] if l.strip()]
    crd = np.array([[float(x) for x in l.split()[1:4]] for l in g.splitlines()[2:] if l.strip()])
    return sym, crd


def masses(syms):
    from ase.data import atomic_masses, atomic_numbers
    mm = np.array([atomic_masses[atomic_numbers[s]] for s in syms])
    mm = np.where(mm < 2.0, 3.0, mm)                       # HMR
    return mm


def water_ref(pu):
    from ase import Atoms
    from batched_relax import batched_energies
    s, c = xtb_opt("O", 0)
    E = float(batched_energies(pu, [Atoms(symbols=s, positions=c, info={"charge": 0, "spin": 1})])[0]) * EV_KJ
    return E + GWATER_EXP + RT * math.log(55.34)


def run_ion(pu, name, smi, q):
    from ase import Atoms
    from batched_relax import batched_fire, _predict, batched_energies
    import grand_canonical_clusters as gc
    sym0, crd0 = xtb_opt(smi, q); nsol = len(sym0)
    Egas = float(batched_energies(pu, [Atoms(symbols=sym0, positions=crd0, info={"charge": q, "spin": 1})])[0]) * EV_KJ
    clus = []
    for r in range(NREP):
        cs, cc = gc.seed_waters(list(sym0), crd0, NWAT, np.random.default_rng(r + 1))
        clus.append(Atoms(symbols=cs, positions=cc, info={"charge": int(q), "spin": 1}))
    syms = clus[0].get_chemical_symbols(); nat = len(syms)
    rel, _, _ = batched_fire(pu, clus, fmax=0.15, steps=150, return_converged=True)
    P = np.stack([a.get_positions() for a in rel])
    m = masses(syms)[None, :, None]
    rng = np.random.default_rng(0); V = rng.normal(size=P.shape) * np.sqrt(KB * T / m)
    dt = DT * FS; gamma = 0.02 / FS; c1 = math.exp(-gamma * dt); c2 = math.sqrt(1 - c1 * c1)

    def forces(P):
        A = [Atoms(symbols=syms, positions=P[i], info={"charge": int(q), "spin": 1}) for i in range(NREP)]
        _, F, bi = _predict(pu, A); F = F.detach().cpu().numpy(); bi = bi.detach().cpu().numpy()
        return np.stack([F[bi == i] for i in range(NREP)])

    def energies(P):
        A = [Atoms(symbols=syms, positions=P[i], info={"charge": int(q), "spin": 1}) for i in range(NREP)]
        return batched_energies(pu, A).detach().cpu().numpy() * EV_KJ

    F = forces(P); nsteps = int(PS * 1000 / DT); Gw = run_ion._Gw
    samples = []                                            # (time, ΔGsolv estimate per replica averaged)
    for s in range(nsteps):
        V = V + 0.5 * dt * F / m; P = P + 0.5 * dt * V
        V = c1 * V + c2 * np.sqrt(KB * T / m) * rng.normal(size=V.shape); P = P + 0.5 * dt * V
        d = P - P.mean(1, keepdims=True); rr = np.linalg.norm(d, axis=2, keepdims=True)
        P = np.where(rr > WALL_R, P - 0.1 * (rr - WALL_R) * d / (rr + 1e-6), P)
        F = forces(P); V = V + 0.5 * dt * F / m
        if s % 100 == 0 and s > nsteps // 5:               # sample after 20% equilibration
            Ecl = energies(P)
            g = []
            for i in range(NREP):
                solv = xtb_dgsolv(syms, P[i], q, "cosmo")
                if solv is not None:
                    g.append(Ecl[i] + solv - NWAT * Gw - Egas)
            if g:
                samples.append((s * DT / 1000.0, float(np.mean(g))))
    ts = np.array([x[1] for x in samples])
    # convergence: block-average over thirds + bootstrap SE
    thirds = [ts[:len(ts)//3], ts[len(ts)//3:2*len(ts)//3], ts[2*len(ts)//3:]]
    blocks = [float(b.mean()) for b in thirds if len(b)]
    rngb = np.random.default_rng(0)
    se = float(np.std([ts[rngb.integers(0, len(ts), len(ts))].mean() for _ in range(300)])) if len(ts) > 3 else float("nan")
    return dict(name=name, q=q, n=len(ts), dGsolv=float(ts.mean()), se=se, blocks=blocks)


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    from batched_relax import _ensure_registered, load_uma
    _ensure_registered("uma-s-1p2p1"); pu = load_uma("uma-s-1p2p1")
    run_ion._Gw = water_ref(pu); print(f"G*_liq(H2O) = {run_ion._Gw:.1f}", flush=True)
    res = {}
    for name, smi, q in IONS:
        r = run_ion(pu, name, smi, q); res[name] = r
        print(f"{name:8s} ΔGsolv = {r['dGsolv']:8.1f} ± {r['se']:.1f} kJ  (n={r['n']}, thirds={[round(b) for b in r['blocks']]})", flush=True)
    a, f = res["acetate"], res["formate"]
    ddg = a["dGsolv"] - f["dGsolv"]
    print(f"\nΔΔGsolv(acetate - formate) = {ddg:+.1f} kJ  [exp ~ +32]  (convention-free relative)")
    conv = all(np.std(r["blocks"]) < 10 for r in res.values() if len(r["blocks"]) == 3)
    print(f"CONVERGENCE: block-drift < 10 kJ = {conv};  SE acetate {a['se']:.1f}")
    print("VERDICT: FEP path viable if CONVERGED (small block-drift/SE) AND ΔΔG ~ +32. Else wall beyond method.")
    json.dump(res, open("artifacts/solvation_fe_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
