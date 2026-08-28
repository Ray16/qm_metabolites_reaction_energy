"""Alchemical TI solvation free energy — CORRECTLY configured (not an energy average). Decouple the solute
from the explicit-water droplet along λ and integrate the mean interaction force:

    U(λ) = (1-λ) U_full + λ U_decoupled ,   U_decoupled = UMA(solute alone) + UMA(waters alone)
    ∂U/∂λ = U_decoupled - U_full = -E_interaction(solute<->waters)
    ΔG_couple = ∫₀¹ ⟨∂U/∂λ⟩_λ dλ  = -∫₀¹ ⟨E_int⟩_λ dλ

This is a proper TI (samples each λ window on the λ-mixed potential, integrates the mean force), UMA-native,
no classical FF. VALIDATION-FIRST: run on METHANOL (neutral, reliable exp ΔGhyd ≈ -21 kJ/mol from FreeSolv)
to prove the machinery reproduces a known number BEFORE trusting it on a charged phosphate. NOTE: linear
(no soft-core) coupling is fine for a neutral like methanol; a bare ion will need soft-core / more λ near
the endpoints — that is the NEXT correctness step, gated on this passing.

Each MD step does 3 batched UMA passes (full droplet / solute-only / waters-only) so ∂U/∂λ is exact.
"""
import os, sys, subprocess, tempfile, shutil, math, json
sys.path.insert(0, "scripts"); sys.path.insert(0, os.path.join("..", "..", "backup", "explicit_water"))
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

XTB = os.environ.get("XTB_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb")
SCRATCH = os.environ.get("QM_SCRATCH", "/nfs/lambda_stor_01/homes/rzhu/qm_scratch")
EV_KJ = 96.48533212; KB = 8.617333262e-5; FS = 0.09822694788; T = 298.15
NWAT = int(os.environ.get("NWAT", "20")); NREP = int(os.environ.get("NREP", "8"))
EQ_PS = float(os.environ.get("EQ_PS", "2")); PROD_PS = float(os.environ.get("PROD_PS", "4"))
DT = float(os.environ.get("DT_FS", "1.0")); WALL_R = float(os.environ.get("WALL_R", "7.0"))
LAMBDAS = [float(x) for x in os.environ.get("LAMBDAS", "0.0,0.2,0.4,0.6,0.8,1.0").split(",")]
SMI = os.environ.get("SMI", "CO"); Q = int(os.environ.get("Q", "0")); EXP = os.environ.get("EXP", "-21")


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
    return np.where(mm < 2.0, 3.0, mm)                     # HMR


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    from ase import Atoms
    from batched_relax import _ensure_registered, load_uma, _predict, batched_fire, batched_energies
    import grand_canonical_clusters as gc
    _ensure_registered("uma-s-1p2p1"); pu = load_uma("uma-s-1p2p1")

    sym0, crd0 = xtb_opt(SMI, Q); nsol = len(sym0)
    clus = []
    for r in range(NREP):
        cs, cc = gc.seed_waters(list(sym0), crd0, NWAT, np.random.default_rng(r + 1))
        clus.append(Atoms(symbols=cs, positions=cc, info={"charge": int(Q), "spin": 1}))
    syms = clus[0].get_chemical_symbols(); nat = len(syms)
    sol_idx = list(range(nsol)); wat_idx = list(range(nsol, nat))
    wsyms = [syms[i] for i in wat_idx]
    rel, _, _ = batched_fire(pu, clus, fmax=0.15, steps=150, return_converged=True)
    P0 = np.stack([a.get_positions() for a in rel])
    m = masses(syms)[None, :, None]
    print(f"solute {SMI} (q={Q}, {nsol} atoms) + {NWAT} waters, {NREP} replicas x {nat} atoms", flush=True)

    def _forces_energy(P, symbols, charge):
        A = [Atoms(symbols=symbols, positions=P[i], info={"charge": int(charge), "spin": 1}) for i in range(len(P))]
        E, F, bi = _predict(pu, A); E = E.detach().cpu().numpy() * EV_KJ
        F = F.detach().cpu().numpy(); bi = bi.detach().cpu().numpy()
        Fl = [F[bi == i] for i in range(len(P))]
        return E, Fl

    def full_and_decoupled(P):
        # full droplet
        Ef, Ff = _forces_energy(P, syms, Q)
        # solute alone (carries the charge) and waters alone (neutral)
        Es, Fs = _forces_energy(P[:, sol_idx, :], sym0, Q)
        Ew, Fw = _forces_energy(P[:, wat_idx, :], wsyms, 0)
        Ed = Es + Ew
        Fd = []
        for i in range(len(P)):
            fd = np.zeros((nat, 3)); fd[sol_idx] = Fs[i]; fd[wat_idx] = Fw[i]; Fd.append(fd)
        Ffull = np.stack(Ff)
        dUdl = Ed - Ef                                     # = -E_interaction (kJ/mol) per replica
        return Ffull, np.stack(Fd), dUdl

    dt = DT * FS; gamma = 0.02 / FS; c1 = math.exp(-gamma * dt); c2 = math.sqrt(1 - c1 * c1)
    rng = np.random.default_rng(0)
    dUdl_lambda = []
    for lam in LAMBDAS:
        P = P0.copy(); V = rng.normal(size=P.shape) * np.sqrt(KB * T / m)
        Ff, Fd, _ = full_and_decoupled(P); F = (1 - lam) * Ff + lam * Fd
        neq = int(EQ_PS * 1000 / DT); nprod = int(PROD_PS * 1000 / DT); acc = []
        for s in range(neq + nprod):
            V = V + 0.5 * dt * F / m; P = P + 0.5 * dt * V
            V = c1 * V + c2 * np.sqrt(KB * T / m) * rng.normal(size=V.shape); P = P + 0.5 * dt * V
            d = P - P.mean(1, keepdims=True); rr = np.linalg.norm(d, axis=2, keepdims=True)
            P = np.where(rr > WALL_R, P - 0.1 * (rr - WALL_R) * d / (rr + 1e-6), P)
            Ff, Fd, dUdl = full_and_decoupled(P); F = (1 - lam) * Ff + lam * Fd
            V = V + 0.5 * dt * F / m
            if s >= neq and s % 20 == 0:
                acc.append(float(np.mean(dUdl)))
        mean = float(np.mean(acc)); se = float(np.std(acc) / math.sqrt(max(1, len(acc))))
        dUdl_lambda.append((lam, mean, se))
        print(f"  λ={lam:.2f}  <∂U/∂λ> = {mean:+8.1f} ± {se:.1f} kJ/mol  (n={len(acc)})", flush=True)

    lams = np.array([x[0] for x in dUdl_lambda]); vals = np.array([x[1] for x in dUdl_lambda])
    dG_0to1 = float(np.trapz(vals, lams))                  # ∫<∂U/∂λ>dλ = G(decoupled) - G(coupled) = -ΔGsolv
    dGsolv = -dG_0to1                                       # SIGN: solvation = coupled - decoupled
    print(f"\nΔGsolv(explicit droplet, TI) = {dGsolv:+.1f} kJ/mol   [= -∫<∂U/∂λ>dλ]")
    print(f"experimental ΔGhyd ≈ {EXP} kJ/mol")
    # ENDPOINT DIAGNOSTIC: is the integrand smooth, or does λ->1 (no soft-core) blow up?
    jumps = np.abs(np.diff(vals))
    print(f"integrand |Δ between λ points| = {[round(j) for j in jumps]}  (a huge last jump = endpoint singularity -> need soft-core)")
    print("VALIDATION: PASS if within ~5-10 kJ of exp AND integrand smooth. If endpoint spikes -> add soft-core before the ion.")
    json.dump({"smi": SMI, "q": Q, "dG_couple": dG, "curve": dUdl_lambda, "exp": EXP},
              open("artifacts/solvation_ti_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
