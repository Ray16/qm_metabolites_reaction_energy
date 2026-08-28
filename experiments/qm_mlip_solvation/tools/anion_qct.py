"""Does QCT actually FIX the anion-solvation wall? The wall reactions fail because implicit xtb-COSMO
UNDER-solvates the created anion (~+15-20 kJ per anion, the logged wall error). Test directly: compute the
QCT (cluster-continuum) hydration free energy of a phosphate/carboxylate anion and compare to its COSMO
value. If QCT solvates it ~15-20 kJ MORE (more negative), that is exactly the missing correction the wall
needs -> QCT would move those reactions toward experiment.

QCT: ΔG*_hyd(A) = [E_UMA(A(H2O)n) + Gcorr(A(H2O)n) + COSMO(A(H2O)n)] - [E_UMA(A)+Gcorr(A)] - n*G*_liq(H2O)
inner shell = anion + waters H-bond-donating to its terminal O's (n from the validated coordination MD).
COSMO alone = xtb_dgsolv(A, 'cosmo'). Gap = QCT - COSMO = under-solvation COSMO misses.
"""
import os, sys, subprocess, tempfile, shutil, math, json
sys.path.insert(0, "scripts")
import numpy as np
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from thermal_solv import xtb_dgsolv, uma_gibbs_corr

EV_KJ = 96.48533212; T = 298.15; RT = 8.314462618e-3 * T
XTB = os.environ.get("XTB_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb")
SCRATCH = os.environ.get("QM_SCRATCH", "/nfs/lambda_stor_01/homes/rzhu/qm_scratch")
NSEED = int(os.environ.get("NSEED", "12"))
# (name, SMILES, charge, waters_per_terminal_O)  — coordination ~2-3/O from the validated coordination MD
ANIONS = json.loads(os.environ.get("ANIONS",
    '[["acetate","CC(=O)[O-]",-1,2],["methylphosphate2-","COP(=O)([O-])[O-]",-2,2],["HPO4-2","OP(=O)([O-])[O-]",-2,2]]'))


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


def terminal_anion_O(smi):
    """indices (in AddHs/xtb order) of terminal O's on a group bearing a formal [O-] (all resonance O's)."""
    m = Chem.AddHs(Chem.MolFromSmiles(smi)); centers = set()
    for a in m.GetAtoms():
        if a.GetSymbol() == "O" and a.GetFormalCharge() < 0:
            for nb in a.GetNeighbors():
                if nb.GetSymbol() in ("C", "S", "P"):
                    centers.add(nb.GetIdx())
    out = []
    for ci in centers:
        for nb in m.GetAtomWithIdx(ci).GetNeighbors():
            if nb.GetSymbol() == "O":
                heavy = [x for x in nb.GetNeighbors() if x.GetSymbol() != "H"]
                if len(heavy) == 1 and not any(x.GetSymbol() == "H" for x in nb.GetNeighbors()):
                    out.append(nb.GetIdx())
    return out


def build_cluster(sym0, crd0, oidx, nper, rng):
    """place nper donor waters near each terminal anion O (one O-H pointing at the anion O ~1.8 A)."""
    sym = list(sym0); crd = [c for c in crd0]
    placed = []
    for oi in oidx:
        base = crd0[oi]
        # outward direction = away from the molecular COM
        out = base - crd0.mean(0); out = out / (np.linalg.norm(out) or 1.0)
        for k in range(nper):
            d = out + 0.4 * rng.normal(size=3); d /= np.linalg.norm(d)
            # clash-avoid vs already placed water O's
            ok = False
            for _ in range(30):
                Opos = base + d * 2.7
                if not placed or min(np.linalg.norm(Opos - p) for p in placed) > 2.6:
                    ok = True; break
                d = out + 0.6 * rng.normal(size=3); d /= np.linalg.norm(d)
            placed.append(Opos)
            hb = (base - Opos); hb /= np.linalg.norm(hb)      # one H toward the anion O (donor)
            perp = np.cross(hb, rng.normal(size=3)); perp /= (np.linalg.norm(perp) or 1.0)
            sym += ["O", "H", "H"]
            crd += [Opos, Opos + 0.96 * hb, Opos + 0.96 * (-0.33 * hb + 0.94 * perp)]
    return sym, np.array(crd), len(placed)


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    from batched_relax import _ensure_registered, load_uma, batched_fire, batched_energies
    _ensure_registered("uma-s-1p2p1"); pu = load_uma("uma-s-1p2p1")

    ws, wc = xtb_opt("O", 0)
    Ew = float(batched_energies(pu, [Atoms(symbols=ws, positions=wc, info={"charge": 0, "spin": 1})])[0]) * EV_KJ
    Gwt = uma_gibbs_corr(pu, ws, wc.tolist(), 0); Sw = xtb_dgsolv(ws, wc.tolist(), 0, "cosmo")
    Gliq = Ew + Gwt + Sw + RT * math.log(55.34)
    print(f"G*_liq(H2O) = {Gliq:.1f}\n", flush=True)
    print(f"{'anion':16s} {'q':>2s} {'n_w':>3s} {'COSMO':>9s} {'QCT':>9s} {'QCT-COSMO':>10s}  interpretation")

    for name, smi, q, nper in ANIONS:
        sym0, crd0 = xtb_opt(smi, q)
        Ea = float(batched_energies(pu, [Atoms(symbols=sym0, positions=crd0, info={"charge": int(q), "spin": 1})])[0]) * EV_KJ
        Ga_thermal = uma_gibbs_corr(pu, sym0, crd0.tolist(), q)
        cosmo = xtb_dgsolv(sym0, crd0.tolist(), q, "cosmo")            # implicit-only (pipeline value)
        oidx = terminal_anion_O(smi)
        # build NSEED clusters, relax, Boltzmann over converged
        clus = []
        for s in range(NSEED):
            cs, cc, nw = build_cluster(sym0, crd0, oidx, nper, np.random.default_rng(s + 1))
            clus.append(Atoms(symbols=cs, positions=cc, info={"charge": int(q), "spin": 1}))
        n_w = nw
        rel, E, conv = batched_fire(pu, clus, fmax=0.05, steps=400, return_converged=True)
        Gs = []
        for a, e, c in zip(rel, E, conv):
            if not c: continue
            s2 = xtb_dgsolv(a.get_chemical_symbols(), a.get_positions(), q, "cosmo")
            if s2 is None: continue
            th = uma_gibbs_corr(pu, a.get_chemical_symbols(), a.get_positions().tolist(), q)
            Gs.append(float(e) * EV_KJ + th + s2)                     # G_aq(cluster)
        if not Gs:
            print(f"{name:16s} (no converged clusters)"); continue
        b = np.array(Gs); Gaq = float(b.min() - RT * math.log(np.mean(np.exp(-(b - b.min()) / RT))))  # Boltzmann
        qct = Gaq - (Ea + Ga_thermal) - n_w * Gliq                    # QCT hydration free energy
        gap = qct - cosmo
        interp = "QCT solvates MORE (wall correction!)" if gap < -8 else ("~same" if abs(gap) <= 8 else "QCT LESS (?)")
        print(f"{name:16s} {q:>2d} {n_w:>3d} {cosmo:9.1f} {qct:9.1f} {gap:+10.1f}  {interp}", flush=True)
    print("\nIf QCT-COSMO ~ -15 to -25 kJ per created anion, that is the missing solvation the wall reactions"
          " need -> QCT improves them. If ~0, COSMO was already fine and QCT won't help.")


if __name__ == "__main__":
    main()
