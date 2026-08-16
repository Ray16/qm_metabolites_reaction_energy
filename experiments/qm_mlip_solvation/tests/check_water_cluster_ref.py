"""Does the same-size water-CLUSTER reference (Bryantsev cluster cycle) fix the explicit
created/destroyed-anion bookkeeping? Compute G_wc(2) = G of a relaxed (H2O)2 cluster the
SAME way as the solute cluster, then test whether G_clus(AcO.2W) - G_wc(2) reproduces the
implicit G_aq(AcO-). If it lands ~100 kJ off, the reference is NOT the (full) fix: the
anion-water binding has no cancellation partner (the wall)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "backup", "explicit_water"))
from batched_relax import load_uma, batched_energies, batched_fire
from step7b_charge_balanced_waters import bare_geom
from thermal_solv import uma_gibbs_corr, xtb_dgsolv
import grand_canonical_clusters as gc
from ase import Atoms
EV2KJ = 96.485

def G_of(pu, sym, pos, q):
    a = Atoms(symbols=list(sym), positions=pos, info={"charge": int(q), "spin": 1})
    E = float(batched_energies(pu, [a])[0]) * EV2KJ
    solv = xtb_dgsolv(list(sym), pos, q, "cosmo")
    thermal = uma_gibbs_corr(pu, list(sym), pos, q)
    return E + solv + thermal, E, solv, thermal

def main():
    pu = load_uma()
    # single liquid-water monomer reference (E+solv+thermal, NO conc term for this test)
    wsym, wpos = bare_geom(pu, 0, "O")
    Gw1, Ew, sw, tw = G_of(pu, wsym, wpos, 0)
    print(f"G(1 water, E+solv+thermal) = {Gw1:.1f}")

    # relaxed (H2O)2 cluster (manual H-bonded dimer geometry, UMA relaxes it)
    c2s = ["O", "H", "H", "O", "H", "H"]
    c2p = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0],
                    [2.80, 0.0, 0.0], [3.35, -0.45, 0.60], [3.35, -0.45, -0.60]])
    cl = Atoms(symbols=c2s, positions=c2p, info={"charge": 0, "spin": 1})
    rel, E, conv = batched_fire(pu, [cl], fmax=0.05, steps=400, stop_frac=1.0,
                                return_converged=True, label="wc2")
    rs = rel[0].get_chemical_symbols(); rp = rel[0].get_positions()
    Gwc2, Ewc, swc, twc = G_of(pu, rs, rp, 0)
    print(f"G_wc(2) [(H2O)2 cluster, E+solv+thermal] = {Gwc2:.1f}")
    print(f"  dimer binding vs 2 monomers = {Gwc2 - 2*Gw1:+.1f} kJ (expect ~ -20)")

    # the AcO- 2-water cluster G (E+solv+thermal, bare-solute thermal) from the pipeline run:
    #   Gens(E+solv) -1001782.4 + thermal(bare) 54.8 = -1001727.6
    G_clus_AcO = -1001727.6
    G_implicit_AcO = -600277.5     # implicit AcO- from t_01713_i.log
    print(f"\nTEST: G_clus(AcO.2W) - G_wc(2) = {G_clus_AcO - Gwc2:.1f}")
    print(f"      implicit G_aq(AcO-)      = {G_implicit_AcO:.1f}")
    print(f"      residual (should be ~0 if cluster ref fixes it) = "
          f"{(G_clus_AcO - Gwc2) - G_implicit_AcO:+.1f} kJ/mol")
    print(f"\nFor comparison, my MONOMER ref gave residual "
          f"{(G_clus_AcO - 2*Gw1) - G_implicit_AcO:+.1f} kJ/mol")

if __name__ == "__main__":
    main()
