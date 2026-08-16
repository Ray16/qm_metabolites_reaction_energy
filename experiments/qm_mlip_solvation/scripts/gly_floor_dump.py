"""Glycosyl-floor localization, step 1 (uma env): relax the truncated rxn00605 species
with UMA, dump gas-phase geometry + charge + UMA electronic energy. A sibling script
re-scores the SAME geometries with AIMNet2 (independent DFT-trained potential). If the
two methods give the same reaction ΔE_elec, the +11 kJ floor is NOT electronic
(-> solvation/speciation); if they diverge ~11, it IS the MLIP electronic limit.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "backup", "explicit_water"))
from batched_relax import load_uma, batched_energies, batched_fire
from step4e_targeted import pool_confs, boltz
from ase import Atoms
from rdkit import Chem
EV2KJ = 96.485

# truncated rxn00605 (radius 2), coeff for ΔE_elec
SPECIES = {
    "donorCap":  (-1, 0,  "CC(O)O"),
    "G6Pt":      (-1, -1, "O=P([O-])(O)O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O"),
    "glycoside": (+1, 0,  "C[C@H](O)O[C@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O"),
    "Pi":        (+1, -1, "O=P([O-])(O)O"),
}
OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts", "gly_floor")
os.makedirs(OUT, exist_ok=True)

def relax_best(pu, smi, q, seeds=(1, 2, 3), keep=12, pool=160):
    best = None
    for seed in seeds:
        cands = pool_confs(smi, q, n=pool, seed=seed)
        order = np.argsort(batched_energies(pu, cands))[:keep]
        sel = [cands[i] for i in order]
        rel, E, conv = batched_fire(pu, sel, fmax=0.05, steps=300, stop_frac=0.9,
                                    return_converged=True, label="gf")
        for a, e, c in zip(rel, E, conv):
            if c and (best is None or e < best[0]):
                best = (float(e), a.get_chemical_symbols(), a.get_positions().copy())
    return best

def main():
    pu = load_uma()
    rows = {}
    for name, (coeff, q, smi) in SPECIES.items():
        b = relax_best(pu, smi, q)
        E_kj = b[0] * EV2KJ
        sym, pos = b[1], b[2]
        # write xyz with charge in comment
        p = os.path.join(OUT, f"{name}.xyz")
        with open(p, "w") as f:
            f.write(f"{len(sym)}\ncharge={q} uma_E_kj={E_kj:.3f} coeff={coeff}\n")
            for s, (x, y, z) in zip(sym, pos):
                f.write(f"{s} {x:.6f} {y:.6f} {z:.6f}\n")
        rows[name] = dict(coeff=coeff, charge=q, uma_E_kj=round(E_kj, 3), xyz=p, n=len(sym))
        print(f"{name:10s} q{q:+d} UMA_E {E_kj:.1f} kJ  -> {p}", flush=True)
    dE = sum(r["coeff"] * r["uma_E_kj"] for r in rows.values())
    print(f"\nUMA gas ΔE_elec(reaction) = {dE:+.1f} kJ/mol", flush=True)
    json.dump(rows, open(os.path.join(OUT, "uma.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
