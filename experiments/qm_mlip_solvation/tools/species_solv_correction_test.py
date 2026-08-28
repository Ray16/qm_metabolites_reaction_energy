"""TEST (not wired): does a SPECIES-LEVEL solvation correction reproduce the empirical anchor offsets?

Physics premise (established): UMA gas-phase E ~= DLPNO-CCSD(T) (glycosyl +0.7), so the +57 phosphagen /
+20 phosphatase over-predictions cannot be an electronic-energy error -- they must live in the IMPLICIT
solvation of the created/destroyed charged species. If so, the offset is NOT a reaction fit: it is the sum
of a per-species solvation correction

    delta(species) = dGsolv_explicit_cluster  -  dGsolv_implicit_COSMO         (experiment-free)

where the explicit first-shell cluster-continuum (directed waters + continuum, exp water reference; the
method validated to cut anion pKa error 44->10) is the higher-fidelity reference. implicit COSMO is exactly
what the pipeline uses. Then the reaction's solvation correction is

    corr_rxn = sum_species  stoich * delta(species)          (added to dG_UMA)

and it should CANCEL the over-prediction:  predicted_offset = -corr_rxn  ?~=?  measured (+20 / +57).

We test two model reactions (small, charge/mass balanced) standing in for the two anchor classes:
  PHOSPHATASE monoester hydrolysis (created FREE phosphate dianion -- the logged "Pi dianion under-solv"):
      methylphosphate(2-) + H2O  ->  methanol + HPO4(2-)
  PHOSPHAGEN phosphoryl transfer (cationic guanidinium, charge-balanced -> isolates phospho-guanidinium
  solvation; note the pipeline scores phosphagen at pH-0 so a NET-anion wall is largely absent -> we expect
  this to explain only PART of +57, the rest being P-N bond-reference / Mg):
      methylphosphate(0) + methylguanidinium(+1)  ->  methanol + N-phospho-methylguanidinium(+1)

Per species: bare geom (xtb opt) -> UMA bare energy; implicit dGsolv (xtb-COSMO); explicit dGsolv via
POOL-seed first-shell clusters (batched UMA FIRE relax, Boltzmann average, exp water reference). Report
delta per species, the net reaction correction, and predicted-vs-measured offset. GPU-only, OMP=1.
"""
import os, sys, subprocess, tempfile, shutil, math, json
sys.path.insert(0, "scripts"); sys.path.insert(0, os.path.join("..", "..", "backup", "explicit_water"))
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from thermal_solv import xtb_dgsolv

XTB = os.environ.get("XTB_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb")
SCRATCH = os.environ.get("QM_SCRATCH", "/nfs/lambda_stor_01/homes/rzhu/qm_scratch")
EV_KJ = 96.48533212
T = 298.15; RT = 8.314462618e-3 * T
POOL = int(os.environ.get("POOL", "32"))
GWATER_EXP_SOLV = -26.4    # experimental hydration free energy of water (kJ); replaces COSMO's -2.8

# name: (charge, SMILES, n_first_shell_waters)   -- coordination rule: ~2 per anionic O-, 1 per cation N-H, 1 neutral polar
SPECIES = {
    "methylphosphate2-": (-2, "COP(=O)([O-])[O-]",        8),   # phosphatase reactant (ester dianion)
    "HPO4-2":            (-2, "OP(=O)([O-])[O-]",          8),   # phosphatase product (FREE phosphate dianion)
    "methanol":          ( 0, "CO",                        2),
    "water":             ( 0, "O",                         2),
    "methylphosphate0":  ( 0, "COP(=O)(O)O",               4),   # phosphagen reactant (neutral, pH-0 form)
    "methylguanidinium": ( 1, "CNC(=[NH2+])N",             4),   # phosphagen reactant (cation)
    "P-methylguanidinium":(1, "CNC(=[NH2+])NP(=O)(O)O",    6),   # phosphagen product (cationic phosphoramidate)
}

REACTIONS = {
    # class -> (measured_offset_kJ, {species: stoich})   stoich <0 reactant, >0 product
    "phosphatase_monoester": (20.3, {"methylphosphate2-": -1, "water": -1, "methanol": +1, "HPO4-2": +1}),
    "phosphagen":            (57.3, {"methylphosphate0": -1, "methylguanidinium": -1,
                                     "methanol": +1, "P-methylguanidinium": +1}),
}


def xtb_opt(smi, q):
    wd = tempfile.mkdtemp(dir=SCRATCH)
    try:
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        if AllChem.EmbedMolecule(m, randomSeed=1) != 0:
            AllChem.EmbedMolecule(m, useRandomCoords=True, randomSeed=1)
        AllChem.MMFFOptimizeMolecule(m, maxIters=300)
        Chem.MolToXYZFile(m, os.path.join(wd, "in.xyz"))
        subprocess.run([XTB, "in.xyz", "--gfn", "2", "--chrg", str(int(q)), "--opt", "tight"], cwd=wd,
                       env={**os.environ, "OMP_NUM_THREADS": "1"}, capture_output=True, text=True, timeout=600)
        g = open(os.path.join(wd, "xtbopt.xyz")).read()
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    sym = [l.split()[0] for l in g.splitlines()[2:] if l.strip()]
    crd = np.array([[float(x) for x in l.split()[1:4]] for l in g.splitlines()[2:] if l.strip()])
    return sym, crd


def boltz(Gs):
    b = np.asarray(Gs, float); m = b.min()
    return float(m - RT * math.log(np.exp(-(b - m) / RT).mean()))


def boot_se(Gs, nboot=300):
    b = np.asarray(Gs, float); n = len(b)
    if n < 3: return float("nan")
    rng = np.random.default_rng(0)
    return float(np.std([boltz(b[rng.integers(0, n, n)]) for _ in range(nboot)]))


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    from ase import Atoms
    from batched_relax import _ensure_registered, load_uma, batched_energies, batched_fire
    import grand_canonical_clusters as gc
    _ensure_registered("uma-s-1p2p1"); pu = load_uma("uma-s-1p2p1")

    ws, wc = xtb_opt("O", 0)
    Ew = float(batched_energies(pu, [Atoms(symbols=ws, positions=wc, info={"charge": 0, "spin": 1})])[0]) * EV_KJ
    Gwref = Ew + GWATER_EXP_SOLV                         # experimental-anchored liquid-water reference
    print(f"G*_liq(H2O) ref = {Gwref:.1f} kJ  (E_UMA {Ew:.1f} + exp solv {GWATER_EXP_SOLV})", flush=True)

    bare = {}; clusters = []; owner = []
    for nm, (q, smi, nw) in SPECIES.items():
        sym, crd = xtb_opt(smi, q); bare[nm] = (sym, crd, q, nw)
        if sum(1 for a in sym if a != "H") < 2:          # water: too small to cluster; handled analytically below
            continue
        for s in range(POOL):
            cs, cc = gc.seed_waters(list(sym), crd, nw, np.random.default_rng(hash(nm) % 99991 + s))
            owner.append(nm); clusters.append(Atoms(symbols=cs, positions=cc, info={"charge": int(q), "spin": 1}))
    print(f"relaxing {len(clusters)} clusters ({len(SPECIES)} species x {POOL} seeds) batched...", flush=True)
    rel, E, conv = batched_fire(pu, clusters, fmax=0.08, steps=300, return_converged=True)

    Gpool = {nm: [] for nm in SPECIES}
    for a, e, c, nm in zip(rel, E, conv, owner):
        if not c: continue
        s2 = xtb_dgsolv(a.get_chemical_symbols(), a.get_positions(), SPECIES[nm][0], "cosmo")
        if s2 is None: continue
        Gpool[nm].append(float(e) * EV_KJ + s2 - SPECIES[nm][2] * Gwref)

    print(f"\n{'species':20s} {'q':>2s} {'implicit':>9s} {'explicit':>9s} {'±SE':>5s} {'delta(e-i)':>10s} {'n':>3s}")
    rec = {}
    for nm, (q, smi, nw) in SPECIES.items():
        sym, crd, _, _ = bare[nm]
        solv_imp = xtb_dgsolv(list(sym), crd, q, "cosmo")
        if sum(1 for a in sym if a != "H") < 2:          # water: explicit == experimental reference (-26.4)
            delta = GWATER_EXP_SOLV - solv_imp
            rec[nm] = dict(q=q, implicit=solv_imp, explicit=GWATER_EXP_SOLV, se=0.0, delta=delta, n=0)
            print(f"{nm:20s} {q:+2d} {solv_imp:9.1f} {GWATER_EXP_SOLV:9.1f} {0.0:5.1f} {delta:+10.1f} {'exp':>3s}", flush=True)
            continue
        Eu = float(batched_energies(pu, [Atoms(symbols=sym, positions=crd, info={"charge": int(q), "spin": 1})])[0]) * EV_KJ
        Gs = Gpool[nm]
        if not Gs:
            print(f"{nm:20s} {q:+2d}   (no converged clusters)"); continue
        solv_exp = boltz(Gs) - Eu; se = boot_se(Gs); delta = solv_exp - solv_imp
        rec[nm] = dict(q=q, implicit=solv_imp, explicit=solv_exp, se=se, delta=delta, n=len(Gs))
        print(f"{nm:20s} {q:+2d} {solv_imp:9.1f} {solv_exp:9.1f} {se:5.1f} {delta:+10.1f} {len(Gs):3d}", flush=True)

    print("\n=== per-reaction: does species-level solvation reproduce the measured offset? ===")
    out = {}
    for cls, (measured, stoich) in REACTIONS.items():
        if not all(nm in rec for nm in stoich):
            print(f"{cls}: missing species -> skip"); continue
        corr = sum(s * rec[nm]["delta"] for nm, s in stoich.items())   # dGsolv correction added to dG_UMA
        pred_off = -corr                                                # amount it removes from the over-prediction
        contribs = {nm: round(s * rec[nm]["delta"], 1) for nm, s in stoich.items()}
        print(f"\n{cls}:  measured offset +{measured:.1f}")
        print(f"   per-species stoich*delta: {contribs}")
        print(f"   net solvation correction corr_rxn = {corr:+.1f}  ->  predicted offset -corr = {pred_off:+.1f}")
        print(f"   EXPLAINED FRACTION = {pred_off/measured*100:5.0f}%   (residual {measured-pred_off:+.1f} = non-solvation: bond-ref/Mg)")
        out[cls] = dict(measured=measured, predicted=pred_off, corr=corr, contribs=contribs)
    json.dump({"species": rec, "reactions": out}, open("artifacts/species_solv_correction_test.json", "w"), indent=1)
    print("\nwrote artifacts/species_solv_correction_test.json")


if __name__ == "__main__":
    main()
