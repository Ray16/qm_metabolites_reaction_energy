"""Diagnose the rxn01713 carboxylate explicit-water cluster: is the placement good, and
where does the +129 kJ over-stabilization come from? Build AcO- + 2 waters exactly as the
pipeline does, relax with UMA, report H-bond geometry + gas-phase binding energy."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "backup", "explicit_water"))
from batched_relax import load_uma, batched_energies, batched_fire
from step7b_charge_balanced_waters import bare_geom
from water_count import water_count
import grand_canonical_clusters as gc
from ase import Atoms
EV2KJ = 96.485

def E(pu, sym, pos, q):
    a = Atoms(symbols=list(sym), positions=pos, info={"charge": int(q), "spin": 1})
    return float(batched_energies(pu, [a])[0]) * EV2KJ

def main():
    pu = load_uma()
    smi = "CC(=O)[O-]"; q = -1
    n_water, sites = water_count(smi)
    print(f"AcO-  water_count = {n_water}  sites={sites}")
    bsym, bpos = bare_geom(pu, q, smi)
    E_bare = E(pu, bsym, bpos, q)

    # seed + relax cluster (same path as explicit_G)
    rng = np.random.default_rng(0)
    csym, cpos = gc.seed_waters(bsym, bpos, n_water, rng)
    cl = Atoms(symbols=csym, positions=cpos, info={"charge": int(q), "spin": 1})
    rel, Ecl, conv = batched_fire(pu, [cl], fmax=0.05, steps=400, stop_frac=1.0,
                                  return_converged=True, label="chk")
    rsym = rel[0].get_chemical_symbols(); rpos = rel[0].get_positions()
    E_cluster = float(Ecl[0]) * EV2KJ

    # single bare water
    wsym, wpos = bare_geom(pu, 0, "O")
    E_water = E(pu, wsym, wpos, 0)

    # carboxylate O indices (the two O of CC(=O)[O-]): find O atoms in bare solute
    o_idx = [i for i, s in enumerate(bsym) if s == "O"]
    n_solute = len(bsym)
    # water O indices in relaxed cluster
    wO = [n_solute + 3 * i for i in range(n_water)]
    print(f"\ncarboxylate O atoms: {o_idx}   water O atoms: {wO}")
    print("H-bond geometry (relaxed cluster):")
    for wo in wO:
        # nearest carboxylate O to this water O
        dO = {oi: np.linalg.norm(rpos[wo] - rpos[oi]) for oi in o_idx}
        near = min(dO, key=dO.get)
        # water H's are wo+1, wo+2
        for h in (wo + 1, wo + 2):
            d = np.linalg.norm(rpos[h] - rpos[near])
            tag = "  <-- H-bond" if d < 2.2 else ""
            print(f"  waterO{wo}->carboxO{near}: O-O {dO[near]:.2f} A ; H{h}...O {d:.2f} A{tag}")

    E_bind_gas = E_cluster - E_bare - n_water * E_water
    print(f"\nGAS-PHASE binding of {n_water} waters to AcO-:")
    print(f"  E(cluster) {E_cluster:.1f} - E(bare) {E_bare:.1f} - {n_water}*E(water) {n_water*E_water:.1f}")
    print(f"  = {E_bind_gas:+.1f} kJ/mol   ({E_bind_gas/n_water:+.1f} per water)")
    print("\nINTERPRETATION: if ~ -120, the +129 'over-stabilization' is the real gas-phase")
    print("anion-water H-bond energy that a BARE-water reference fails to cancel (needs the")
    print("same-size WATER-CLUSTER reference, Bryantsev cluster cycle, not monomer).")
    # save xyz
    out = os.path.join(os.path.dirname(__file__), "..", "artifacts", "gly_floor", "AcO_2w_relaxed.xyz")
    with open(out, "w") as f:
        f.write(f"{len(rsym)}\nAcO- + 2 waters relaxed, charge {q}\n")
        for s, (x, y, z) in zip(rsym, rpos):
            f.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")
    print(f"\nsaved cluster: {out}")

if __name__ == "__main__":
    main()
