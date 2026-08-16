#!/usr/bin/env python
"""Diagnose the glycosyl +16 kJ error: is free fructose mis-defined / over-stabilized?

Solving the reaction: ΔG = -1803975.9 - G(fructose)  [other 3 species fixed from the
unified run]. Current furanose G=-1803987.7 -> ΔG +11.8. To hit exp -4.2 free fructose
must be ~16 kJ LESS stable (G ~ -1803971.7). So free fructose is 16 kJ TOO STABLE.

Two hypotheses, tested here:
  (A) TAUTOMER: free fructose is β-pyranose (dominant in water), not furanose. But
      pyranose is MORE stable -> predicted to move ΔG the WRONG way (worse). Confirm.
  (B) INTRAMOLECULAR H-BOND over-stabilization in implicit solvent: the min-E free
      fructose forms an OH···OH network real water would break. Add explicit waters to
      the polyol OHs -> should DESTABILIZE the free sugar (raise G) -> ΔG toward exp.

Run (uma env): CUDA_VISIBLE_DEVICES=3 python scripts/diag_fructose.py
"""
import os
import sys
import numpy as np
from ase import Atoms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_THERMO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_THERMO, "backup", "explicit_water"))
import grand_canonical_clusters as gc
from batched_relax import load_uma, batched_energies, batched_fire
from step7b_charge_balanced_waters import bare_geom
from unified_pipeline import implicit_G
from thermal_solv import uma_gibbs_corr, xtb_dgsolv, EV2KJ

# other 3 species free energies from the unified glycosyl run (fixed)
OTHERS = -1803975.9   # = (G[MeUDP] + G[Suc]) - G[MeUDPGlc]
EXP = -4.2

FRU_FORMS = {
    "furanose(current)": (0, "OC[C@H]1OC(O)(CO)[C@@H](O)[C@@H]1O"),
    "beta-pyranose":     (0, "OC[C@@]1(O)OC[C@@H](O)[C@H](O)[C@H]1O"),
    "open-chain-keto":   (0, "OCC(=O)[C@@H](O)[C@H](O)[C@H](O)CO"),
}


def fructose_with_waters(pu, smi, n_water, log):
    """G_aq of free fructose with n explicit waters seeded on its OH oxygens (break
    intramolecular H-bonds). Bare-solute thermal; relaxed-in-solvent not needed (neutral)."""
    bsym, bcoord = bare_geom(pu, 0, smi)
    # seed waters near ALL oxygens (polyol H-bond sites), min-G over a few seeds
    best = None
    for s in range(4):
        rng = np.random.default_rng(700 + s)
        cs, cc = gc.seed_waters(bsym, bcoord, n_water, rng)  # seeds near O sites
        best = (cs, cc) if best is None else best
        clusters = [Atoms(symbols=cs, positions=cc, info={"charge": 0, "spin": 1})]
        rel, E, conv = batched_fire(pu, clusters, fmax=0.06, steps=350,
                                    return_converged=True, label=f"fruW{n_water}s{s}")
        if conv[0]:
            e = float(E[0]) * EV2KJ
            solv = xtb_dgsolv(rel[0].get_chemical_symbols(), rel[0].get_positions(), 0, "cosmo")
            if solv is not None:
                th = uma_gibbs_corr(pu, bsym, bcoord, 0)
                g = e + solv + th
                log(f"    fructose + {n_water}w (seed {s}): E {e:.1f} + solv {solv:.1f} + th {th:.1f} = {g:.1f}")
                return g
    return None


def main():
    log = lambda s: print(s, flush=True)
    log("loading UMA... glycosyl free-fructose diagnostic")
    pu = load_uma()

    log("\n(A) TAUTOMER forms — G_aq and resulting glycosyl ΔG:")
    log(f"  {'form':20s} {'G_fru':>14s} {'ΔG_rxn':>8s} {'err':>7s}")
    for name, (q, smi) in FRU_FORMS.items():
        g = implicit_G(pu, q, smi, None, None, None, log, name)
        if g is None:
            log(f"  {name}: FAILED"); continue
        dG = OTHERS - g
        log(f"  {name:20s} {g:14.1f} {dG:8.1f} {dG-EXP:+7.1f}")

    log("\n(B) INTRAMOLECULAR H-BOND test — explicit waters on free furanose fructose:")
    log(f"  {'n_water':>7s} {'G_fru':>14s} {'ΔG_rxn':>8s} {'err':>7s}")
    fur = FRU_FORMS["furanose(current)"][1]
    for nw in (3, 6):
        g = fructose_with_waters(pu, fur, nw, log)
        if g is None:
            log(f"  {nw}w: FAILED"); continue
        dG = OTHERS - g
        log(f"  {nw:7d} {g:14.1f} {dG:8.1f} {dG-EXP:+7.1f}")
    log("\n  Interpret: (A) pyranose more stable => ΔG MORE positive (tautomer not the fix).")
    log("  (B) if explicit waters RAISE G_fru (less stable) => ΔG toward exp => intra-Hbond confirmed.")


if __name__ == "__main__":
    main()
