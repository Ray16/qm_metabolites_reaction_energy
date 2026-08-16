#!/usr/bin/env python
"""Step 7c: grand potential with the CLUSTER-CYCLE water reference (Bryantsev/Ho).

The grand-canonical pinning (step7) came from a monomer-cycle mu_water = G(H2O
monomer) + standard-state terms, which over-favors every added water because a
BOUND water's lost translational/librational entropy is not consistently referenced.

Fix (user's point): reference the added waters to THEIR OWN water cluster of the
same size -- the cluster cycle  A + (H2O)_n -> A(H2O)_n.  The same-size water
cluster G_wc(n) carries the same low-frequency-mode entropy error as the bound
waters in the solute cluster, so it CANCELS. Then:

    Ω(A) = -RT log Σ_n exp( -(G_clus(A,n) - G_wc(n) - G_clus(A,0)) / RT )

adding a water is favorable only if it binds the ion BETTER than bulk water ->
occupancy PEAKS naturally (no mu_water, no pinning). G_wc(n) cancels between
same-charge species across the charge-conserving reaction -> also validates step7b.

  MeP(-2) + MePPP(-3) -> MePPMe(-2) + PPi(-3), exp ~+1.9.
Run (uma env): CUDA_VISIBLE_DEVICES=5 python scripts/step7c_cluster_cycle.py --nmax 8
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from ase import Atoms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_THERMO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_THERMO, "backup", "explicit_water"))
import grand_canonical_clusters as gc
from batched_relax import load_uma, batched_energies, batched_fire
from step7b_charge_balanced_waters import bare_geom, xtb_corr, SPECIES, EXP

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "artifacts")
EV2KJ = 96.485
RT = gc.RT  # kJ/mol at 298.15


def relax_min_G(pu, clusters, q, label, log):
    """UMA-relax cluster seeds, return min (E_UMA + xtb_corr) in kJ, or None."""
    if not clusters:
        return None
    rel, E, conv = batched_fire(pu, clusters, fmax=0.06, steps=350, stop_frac=0.8,
                                return_converged=True, label=label)
    rel = [a for a, c in zip(rel, conv) if c]; E = E[conv] * EV2KJ
    if not len(E):
        return None
    for i in np.argsort(E):
        corr = xtb_corr(rel[i].get_chemical_symbols(), rel[i].get_positions(), q)
        if corr is not None:
            return float(E[i]) + corr
    return None


def water_cluster_ladder(pu, nmax, seeds, log):
    """G_wc(n) for n=0..nmax: pure-water cluster, min-G over seeds (kJ). G_wc(0)=0."""
    Gwc = {0: 0.0}
    for n in range(1, nmax + 1):
        rng = np.random.default_rng(1000 + n)
        R = 1.6 * (n ** (1.0 / 3.0)) + 1.0
        clusters = []
        for s in range(seeds):
            sym, coord = [], []
            for _ in range(n):
                c = rng.uniform(-R, R, 3)
                sym += ["O", "H", "H"]
                coord += [c, c + [0.96, 0, 0], c + [-0.24, 0.93, 0]]
            clusters.append(Atoms(symbols=sym, positions=np.array(coord),
                                  info={"charge": 0, "spin": 1}))
        g = relax_min_G(pu, clusters, 0, f"wc{n}", log)
        Gwc[n] = g
        log(f"    G_wc({n}) = {g:.1f} kJ" if g is not None else f"    G_wc({n}) FAILED")
    return Gwc


def species_omega(pu, name, q, smi, nmax, seeds, Gwc, log):
    """Cluster-cycle grand potential Ω(A) referenced to same-size water clusters."""
    bsym, bcoord = bare_geom(pu, q, smi)
    n_anion = sum(1 for s in bsym if s == "O")  # generous seed sites; seed_waters picks anionic
    G = {}
    for n in range(0, nmax + 1):
        if n == 0:
            g = relax_min_G(pu, [Atoms(symbols=bsym, positions=bcoord, info={"charge": int(q), "spin": 1})],
                            q, f"{name}n0", log)
        else:
            rng = np.random.default_rng(abs(hash((name, n))) % (2**32))
            clusters = []
            for s in range(seeds):
                csym, ccoord = gc.seed_waters(bsym, bcoord, n, rng)
                clusters.append(Atoms(symbols=csym, positions=ccoord, info={"charge": int(q), "spin": 1}))
            g = relax_min_G(pu, clusters, q, f"{name}n{n}", log)
        G[n] = g
    g0 = G[0]
    # referenced free energy per occupancy: ΔG_solv-ish = G_clus(n) - G_wc(n) - G_clus(0)
    ref = {n: (G[n] - Gwc[n] - g0) for n in G if G[n] is not None and Gwc.get(n) is not None}
    lo = min(ref.values())
    Z = sum(math.exp(-(v - lo) / RT) for v in ref.values())
    omega = g0 + lo - RT * math.log(Z)  # includes bare G(0) so Ω is an absolute-ish species free energy
    occ = {n: math.exp(-(ref[n] - lo) / RT) / Z for n in ref}
    peak = max(occ, key=occ.get)
    top = sorted(occ.items(), key=lambda kv: -kv[1])[:4]
    log(f"    {name:7s} q{q:+d}: Ω {omega:.1f}  peak n={peak}  occ {[(k, round(v,2)) for k,v in top]}")
    return omega, peak, occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=6)
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)
    log(f"loading UMA... cluster-cycle grand potential, nmax={a.nmax} seeds={a.seeds}")
    pu = load_uma()
    log("  water-cluster reference ladder:")
    Gwc = water_cluster_ladder(pu, a.nmax, a.seeds, log)
    if any(Gwc[n] is None for n in Gwc):
        log("  FATAL: water ladder incomplete"); return
    Omega, peaks = {}, {}
    for name, (q, smi) in SPECIES.items():
        om, pk, _ = species_omega(pu, name, q, smi, a.nmax, a.seeds, Gwc, log)
        Omega[name], peaks[name] = om, pk
    dG = (Omega["MePPMe"] + Omega["PPi"]) - (Omega["MeP"] + Omega["MePPP"])
    log(f"\n==== nucleotidyl ΔG, cluster-cycle grand potential ====")
    log(f"  water occupancy peaks: {peaks}")
    log(f"  ΔG = {dG:+.1f} kJ/mol   vs exp {EXP:+.1f}   err {dG-EXP:+.1f}")
    log(f"  (monomer-cycle (superseded, removed) PINNED -> +33; charge-balanced step7b for comparison)")
    json.dump(dict(Omega=Omega, peaks=peaks, dG=dG, exp=EXP, nmax=a.nmax, seeds=a.seeds,
                   Gwc={str(k): v for k, v in Gwc.items()}),
              open(os.path.join(OUT, "step7c_cluster_cycle.json"), "w"), indent=2)
    log("wrote artifacts/step7c_cluster_cycle.json")


if __name__ == "__main__":
    main()
