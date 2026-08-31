"""Step-0 diagnostic for the RED "solvation/reference wall" classes: attribute each class's SYSTEMATIC
bias to a PHYSICS LAYER before designing a fix, so the fix RESOLVES a pipeline limitation instead of
fitting the class mean.

Method (per class, on the smallest NEUTRAL charge-balanced model of the class's bond change):
  electronic layer:  ΔE_UMA  vs  ΔE_DFT (PBE0/def2-TZVP)  on the SAME UMA-relaxed geometry.
      |UMA-DFT| ~ class bias  => the bias is an ELECTRONIC reference error (MLIP mis-ranks the bond).
                                 Escalate to DLPNO-CCSD(T) (ORCA) to see if DFT is the ceiling too.
      |UMA-DFT| ~ 0           => electronic is fine => the bias is SOLVATION or speciation. The neutral
                                 gas-phase ΔE is right, so the aqueous error lives in ΔGsolv (fix =
                                 explicit microsolvation of the created group) or in the microspecies.

This does NOT fit anything: it MEASURES which layer carries the bias, and the fix's validity is later
tested by whether the computed layer correction REPRODUCES the observed bias (falsifiable), not by
matching a mean. Mirrors analysis/verify_adenylylate_physics.py, generalized to a preset table.

Run (uma env, GPU):  CUDA_VISIBLE_DEVICES=0 python analysis/verify_class_physics.py --class hydratase
"""
import argparse
import os
import shutil
import subprocess
import tempfile
from collections import Counter

import numpy as np
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

EV_KJ = 96.48533212
HARTREE_KJ = 2625.499639
XTB = os.environ.get("XTB_BIN", f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb")
SCRATCH = os.environ.get("QM_SCRATCH", "/nfs/lambda_stor_01/homes/rzhu/qm_scratch")

# Each preset: a NEUTRAL, charge-balanced model that isolates ONLY the class's characteristic bond
# change, stripped of spectator anions/cofactors. `bias` = the observed class bias (kJ/mol) from the
# full-367 sweep (sigma_class_calibrated.json), the number the electronic gap must explain (or not).
PRESETS = {
    # C=C + H2O -> C-OH : fumaric acid + water -> malic acid (rxn00799, full-pipeline err +24).
    "hydratase": {"bias": 18.4, "elec_gap": 3.6,
                  "reac": [("OC(=O)/C=C/C(=O)O", 0), ("O", 0)],
                  "prod": [("OC(=O)CC(O)C(=O)O", 0)]},
    # glycosidic C-N -> C-O-P : N-glycoside + phosphoric acid -> glycosyl phosphate + amine
    # (purine-nucleoside-phosphorylase bond change; full-pipeline bias -13.8, PRT/phosphorylase members).
    "glycosyl": {"bias": -13.8, "elec_gap": -5.6,
                 "reac": [("CNC1CCCCO1", 0), ("OP(=O)(O)O", 0)],
                 "prod": [("O=P(O)(O)OC1CCCCO1", 0), ("CN", 0)]},
    # amide C-N + H2O -> carboxylic acid + amine : acetamide + water -> acetic acid + ammonia
    # (amide-hydrolysis bond change, neutral; full-pipeline bias -14.0).
    "amide": {"bias": -14.0, "elec_gap": 2.9,
              "reac": [("CC(=O)N", 0), ("O", 0)],
              "prod": [("CC(=O)O", 0), ("N", 0)]},
    # carbamoyl-P mixed anhydride transfer : carbamoyl phosphate + methylamine -> methylurea + Pi
    # (carbamoyltransfer bond change, neutral; full-pipeline bias +11.3).
    "carbamoyl": {"bias": 11.3, "elec_gap": 3.1,
                  "reac": [("NC(=O)OP(=O)(O)O", 0), ("CN", 0)],
                  "prod": [("CNC(N)=O", 0), ("OP(=O)(O)O", 0)]},
    # GROUP-4 systematic sub-types (test whether the offset is ELECTRONIC or SOLVATION):
    # carnitine acyltransferase = thioester -> ester bond swap (transacylation). model:
    # S-methyl thioacetate + methanol -> methyl acetate + methanethiol. bias -24.5 (sign-consistent).
    "thioester_ester": {"bias": -24.5, "elec_gap": 0.0,
                        "reac": [("CC(=O)SC", 0), ("CO", 0)],
                        "prod": [("CC(=O)OC", 0), ("CS", 0)]},
    # aldose->lactone oxidation, SUBSTRATE isolated with H2 as the clean 2e/2H reductant (removes NAD):
    # tetrahydropyran-2-ol -> tetrahydropyran-2-one + H2. bias -33.9. If UMA~DFT here, the -34 lives in
    # the NAD(P) redox CORE reference, not the substrate; if UMA-DFT ~ -34, the hemiacetal->lactone
    # oxidation electronic is the culprit.
    "aldose_lactone": {"bias": -33.9, "elec_gap": 0.0,
                       "reac": [("OC1CCCCO1", 0)],
                       "prod": [("O=C1CCCCO1", 0), ("[H][H]", 0)]},
    # GROUP-4 systematic sub-types added 2026-08-30 (test electronic vs solvation on the created bond):
    # Mg/NTP carboxylase (ATP+CO2, e.g. pyruvate carboxylase). Isolates the carboxy-phosphate mixed-
    # anhydride C-carboxylation (parallels `carbamoyl`): carboxyphosphate + enol(acetone) ->
    # acetoacetic acid + Pi. bias +25.6 (100% sign-consistent). If UMA~DFT the +26 is solvation.
    "mg_carboxylase": {"bias": 25.6, "elec_gap": 0.0,
                       "reac": [("OC(=O)OP(=O)(O)O", 0), ("CC(O)=C", 0)],
                       "prod": [("CC(=O)CC(=O)O", 0), ("OP(=O)(O)O", 0)]},
    # aromatic transaminase (tyrosine, rxn00493/00527). The aliphatic transaminases already land ~0, so
    # the +22 is the aromatic amino<->keto electronic reference. Isolate as reductive amination with H2 as
    # the clean 2e/2H reductant (H2 identical on UMA and DFT): 4-hydroxyphenylpyruvate + NH3 + H2 ->
    # tyrosine + H2O. If UMA~DFT the +22 is conformer/solvation, not an aromatic electronic mis-rank.
    "aromatic_transaminase": {"bias": 22.0, "elec_gap": 0.0,
                              "reac": [("O=C(O)C(=O)Cc1ccc(O)cc1", 0), ("N", 0), ("[H][H]", 0)],
                              "prod": [("O=C(O)C(N)Cc1ccc(O)cc1", 0), ("O", 0)]},
    # GAPDH acyl-phosphate (mixed anhydride creation with concurrent oxidation). Isolate the aldehyde ->
    # acyl-phosphate oxidative step with H2 as reductant: acetaldehyde + phosphoric acid -> acetyl-
    # phosphate + H2. bias -43. Tests whether the mixed-anhydride/adenylate-family reference is electronic.
    "gapdh_acylP": {"bias": -43.0, "elec_gap": 0.0,
                    "reac": [("CC=O", 0), ("OP(=O)(O)O", 0)],
                    "prod": [("CC(=O)OP(=O)(O)O", 0), ("[H][H]", 0)]},
}


def _formula(smi):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    c = Counter(a.GetSymbol() for a in m.GetAtoms())
    return c


def check_balance(reac, prod):
    net = Counter()
    for smi, _ in reac:
        net.subtract(_formula(smi))
    for smi, _ in prod:
        net.update(_formula(smi))
    bad = {k: v for k, v in net.items() if v != 0}
    return bad


def xtb_opt(smi, q):
    os.makedirs(SCRATCH, exist_ok=True)
    wd = tempfile.mkdtemp(dir=SCRATCH)
    try:
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=1)
        AllChem.MMFFOptimizeMolecule(m)
        Chem.MolToXYZFile(m, f"{wd}/in.xyz")
        subprocess.run([XTB, "in.xyz", "--gfn", "2", "--chrg", str(q), "--opt", "tight"], cwd=wd,
                       env={**os.environ, "OMP_NUM_THREADS": "4"}, capture_output=True, text=True, timeout=600)
        g = open(f"{wd}/xtbopt.xyz").read()
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    lines = [l for l in g.splitlines()[2:] if l.strip()]
    sym = [l.split()[0] for l in lines]
    crd = np.array([[float(x) for x in l.split()[1:4]] for l in lines])
    return sym, crd


def pyscf_E(sym, crd, q, xc="PBE0", basis="def2-tzvp"):
    from pyscf import gto, dft
    mol = gto.M(atom=[(s, tuple(p)) for s, p in zip(sym, crd)], basis=basis,
                charge=int(q), spin=0, verbose=0)
    mf = dft.RKS(mol); mf.xc = xc
    return float(mf.kernel()) * HARTREE_KJ


_POLAR_OH = Chem.MolFromSmarts("[OX2H]")     # hydroxyl / carboxyl O-H (donor site)
_CARBONYL_O = Chem.MolFromSmarts("[OX1]=C")  # C=O acceptor


def polar_water_count(smi):
    """Deterministic first-shell water count for a NEUTRAL polar solute: 2 per O-H donor + 1 per C=O
    acceptor. Same site on both sides of a reaction cancels; the CREATED group's waters do not."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return 0
    return 2 * len(m.GetSubstructMatches(_POLAR_OH)) + len(m.GetSubstructMatches(_CARBONYL_O))


def solv_decompose(a, P):
    """Reaction ΔGsolv three ways: implicit COSMO, implicit ALPB, explicit cluster-continuum on the
    first shell of every polar group (referenced to bulk water so the count cancels). The explicit-minus-
    implicit shift on the CREATED group is the falsifiable test: it must reproduce the non-electronic
    remainder of the class bias if the wall is solvation."""
    from metag.energetics.uma import load_uma, batched_fire, batched_energies
    from metag.energetics.thermal import xtb_dgsolv, xtb_dgsolv_relaxed
    from metag.energetics import water_clusters as gc
    pu = load_uma("uma-s-1p2p1")

    # bulk-water monomer solvation reference (ALPB/COSMO ΔGsolv of one water), for the cluster cycle
    ws, wc = xtb_opt("O", 0)
    gw_cosmo = xtb_dgsolv(ws, wc, 0, "cosmo")

    def geom(smi, q):
        s, c = xtb_opt(smi, q)
        at = Atoms(symbols=s, positions=c, info={"charge": int(q), "spin": 1})
        rel, _ = batched_fire(pu, [at], fmax=0.03, steps=400)
        return rel[0].get_chemical_symbols(), rel[0].get_positions()

    def species_solv(smi, q):
        s, c = geom(smi, q)
        gi_cosmo = xtb_dgsolv(s, c, q, "cosmo")
        gi_alpb = xtb_dgsolv(s, c, q, "gbsa")
        n = polar_water_count(smi)
        # explicit cluster-continuum: seed n waters on polar sites, relax (UMA), ΔGsolv_relaxed(cluster)
        # minus n*gw (Bryantsev monomer cycle) -> solute solvation with an explicit first shell.
        ge_expl = gi_cosmo
        if n > 0:
            rng = np.random.default_rng(1)
            clusters = []
            for _ in range(6):
                cs, cc = gc.seed_waters(s, np.asarray(c), n, rng)
                clusters.append(Atoms(symbols=cs, positions=cc, info={"charge": int(q), "spin": 1}))
            rel, E = batched_fire(pu, clusters, fmax=0.06, steps=350)      # UMA relax all seeds (batched, fast)
            best_i = int(np.argmin(E))                                     # lowest-UMA-E cluster only
            gsolv = xtb_dgsolv_relaxed(rel[best_i].get_chemical_symbols(),  # ONE xtb relaxed-solvation
                                       rel[best_i].get_positions(), q, "cosmo")
            if gsolv is not None:
                ge_expl = gsolv - n * gw_cosmo
        return gi_cosmo, gi_alpb, ge_expl, n

    def rxn_sum(fn, key):
        tot = 0.0
        for smi, q in P["prod"]:
            tot += fn[smi][key]
        for smi, q in P["reac"]:
            tot -= fn[smi][key]
        return tot

    data = {}
    print(f"{'species':30s} {'n_w':>3s} {'ΔGsolv_cosmo':>13s} {'ΔGsolv_alpb':>12s} {'ΔGsolv_expl':>12s}")
    for smi, q in P["prod"] + P["reac"]:
        if smi in data:
            continue
        gc_, ga_, ge_, n = species_solv(smi, q)
        data[smi] = {"cosmo": gc_, "alpb": ga_, "expl": ge_}
        print(f"  {smi:30s} {n:>3d} {gc_:13.1f} {ga_:12.1f} {ge_:12.1f}")
    dcosmo = rxn_sum(data, "cosmo"); dalpb = rxn_sum(data, "alpb"); dexpl = rxn_sum(data, "expl")
    remainder = P["bias"] - P["elec_gap"]
    print(f"\nreaction ΔGsolv:  COSMO {dcosmo:+.1f}   ALPB {dalpb:+.1f}   EXPLICIT {dexpl:+.1f} kJ/mol")
    print(f"explicit − implicit(COSMO) shift = {dexpl - dcosmo:+.1f} kJ/mol")
    print(f"non-electronic remainder of class bias = {remainder:+.1f} kJ/mol (bias {P['bias']:+.1f} − elec {P['elec_gap']:+.1f})")
    print("TEST: the explicit-shift should reproduce the remainder (opposite sign to the pred bias) if "
          "the wall is created-group solvation.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", required=True, choices=list(PRESETS))
    ap.add_argument("--xc", default="PBE0")
    ap.add_argument("--solv", action="store_true", help="run the solvation decomposition (implicit vs explicit)")
    a = ap.parse_args()
    P = PRESETS[a.cls]

    if a.solv:
        bad = check_balance(P["reac"], P["prod"])
        if bad:
            raise SystemExit(f"MODEL NOT BALANCED for {a.cls}: net {dict(bad)}")
        solv_decompose(a, P)
        return

    bad = check_balance(P["reac"], P["prod"])
    if bad:
        raise SystemExit(f"MODEL NOT BALANCED for {a.cls}: net {dict(bad)} -> fix the preset SMILES")
    print(f"[{a.cls}] model balanced. observed full-pipeline class bias = {P['bias']:+.1f} kJ/mol")

    from metag.energetics.uma import load_uma, batched_fire
    pu = load_uma("uma-s-1p2p1")

    def species_E(smi, q):
        s, c = xtb_opt(smi, q)
        at = Atoms(symbols=s, positions=c, info={"charge": int(q), "spin": 1})
        rel, E = batched_fire(pu, [at], fmax=0.03, steps=400)
        geo = rel[0].get_positions(); gsym = rel[0].get_chemical_symbols()
        return float(E[0]) * EV_KJ, pyscf_E(gsym, geo, q, xc=a.xc)   # UMA and DFT on the SAME geometry

    du = dd = 0.0
    print(f"{'species':30s} {'q':>2s} {'E_UMA(kJ)':>14s} {'E_DFT(kJ)':>14s}")
    for smi, q in P["prod"]:
        eu, ed = species_E(smi, q); du += eu; dd += ed
        print(f"  +{smi:28s} {q:>2d} {eu:14.1f} {ed:14.1f}")
    for smi, q in P["reac"]:
        eu, ed = species_E(smi, q); du -= eu; dd -= ed
        print(f"  -{smi:28s} {q:>2d} {eu:14.1f} {ed:14.1f}")
    gap = du - dd
    print(f"\nΔE_UMA = {du:+.1f}   ΔE_DFT[{a.xc}] = {dd:+.1f}   UMA-DFT = {gap:+.1f} kJ/mol")
    print(f"observed class bias = {P['bias']:+.1f} kJ/mol")
    frac = gap / P["bias"] if P["bias"] else float("nan")
    print(f"electronic gap explains {frac*100:.0f}% of the class bias")
    if abs(gap) > 10 and abs(gap - P["bias"]) < abs(P["bias"]) * 0.5:
        print("VERDICT: ELECTRONIC reference error dominates -> fix = higher-level (DLPNO-CCSD(T)/CBH) "
              "correction on this core; NOT explicit solvation. Escalate to CCSD(T) to check the DFT ceiling.")
    elif abs(gap) <= 10:
        print("VERDICT: electronic FINE (UMA≈DFT) -> the neutral gas-phase bond energy is right. The aqueous "
              "bias is SOLVATION (fix = explicit microsolvation of the created group) or SPECIATION.")
    else:
        print("VERDICT: partial electronic gap -> mixed layer; decompose solvation separately before fixing.")


if __name__ == "__main__":
    main()
