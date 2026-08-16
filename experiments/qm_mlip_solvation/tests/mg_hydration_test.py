"""Regime-1 (Mg2+) GATING test: (a) does UMA even run on Mg2+ and [Mg(H2O)6]2+?
(b) does our cluster-continuum reproduce the Mg2+ absolute hydration free energy
(experiment ~ -1830 kJ/mol)? If this is in the right ballpark, first-principles explicit-Mg
is viable; if UMA gives garbage on the +2 ion, the whole regime needs a different engine.
Validation only -- nothing fitted.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "backup", "explicit_water"))
from batched_relax import load_uma, batched_energies, batched_fire
from step7b_charge_balanced_waters import bare_geom
from thermal_solv import uma_gibbs_corr, xtb_dgsolv
from ase import Atoms
EV2KJ = 96.485
RT_LN_C = 8.314e-3 * 298.15 * np.log(55.34)   # gas->liquid std state, +9.96 kJ/mol

def G_liq_water(pu):
    s, c = bare_geom(pu, 0, "O")
    E = float(batched_energies(pu, [Atoms(symbols=list(s), positions=c, info={"charge":0,"spin":1})])[0])*EV2KJ
    return E + uma_gibbs_corr(pu, list(s), c, 0) + xtb_dgsolv(list(s), c, 0, "cosmo") + RT_LN_C

def hexaaqua_geom(d=2.07):
    """Octahedral [Mg(H2O)6]2+ starting geometry (UMA relaxes it)."""
    sym = ["Mg"]; pos = [[0,0,0]]
    axes = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    for u in axes:
        u = np.array(u, float)
        O = d*u
        # a perpendicular direction
        perp = np.cross(u, [0.3,0.5,0.8]); perp/=np.linalg.norm(perp)
        h1 = O + 0.96*(0.6*u + 0.8*perp)      # H's point outward (away from Mg)
        h2 = O + 0.96*(0.6*u - 0.8*perp)
        sym += ["O","H","H"]; pos += [O.tolist(), h1.tolist(), h2.tolist()]
    return sym, np.array(pos)

def main():
    pu = load_uma()
    # (a) bare Mg2+ ion
    mg = Atoms(symbols=["Mg"], positions=[[0,0,0]], info={"charge":2,"spin":1})
    E_mg = float(batched_energies(pu, [mg])[0]) * EV2KJ
    print(f"UMA runs on Mg2+ (bare ion): E = {E_mg:.1f} kJ  ({'OK' if np.isfinite(E_mg) else 'NAN/FAIL'})", flush=True)

    # (b) [Mg(H2O)6]2+ cluster
    s, c = hexaaqua_geom()
    cl = Atoms(symbols=s, positions=c, info={"charge":2,"spin":1})
    rel, E, conv = batched_fire(pu, [cl], fmax=0.05, steps=500, stop_frac=1.0,
                                return_converged=True, label="MgW6")
    rs, rp = rel[0].get_chemical_symbols(), rel[0].get_positions()
    E_clus = float(E[0]) * EV2KJ
    # Mg-O distances after relax (coordination sanity)
    mgpos = rp[0]; od = sorted(np.linalg.norm(rp[i]-mgpos) for i,a in enumerate(rs) if a=="O")
    print(f"[Mg(H2O)6]2+ relaxed: E_UMA {E_clus:.1f}; Mg-O dists {[round(x,2) for x in od]}", flush=True)
    solv = xtb_dgsolv(rs, rp, 2, "cosmo")
    thermal = uma_gibbs_corr(pu, rs, rp, 2)
    Gw = G_liq_water(pu)
    print(f"  dGsolv(cluster) {solv:.1f}, thermal {thermal:.1f}, G_liq(H2O) {Gw:.1f}", flush=True)

    # Bryantsev cluster cycle: dGhyd(Mg2+) = G_aq(cluster) - E_gas(Mg2+) - 6*G_liq(H2O)
    dGhyd = (E_clus + thermal + solv) - E_mg - 6*Gw
    print(f"\n=== Mg2+ hydration FE (n=6 cluster) = {dGhyd:+.0f} kJ/mol   vs experiment ~ -1830 ===", flush=True)
    print(f"    error vs exp: {dGhyd-(-1830):+.0f} kJ/mol", flush=True)
    print("    (n=6 misses outer shells; sign+magnitude in range => Mg machinery VIABLE)", flush=True)

if __name__ == "__main__":
    main()
