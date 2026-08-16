#!/usr/bin/env python
"""Microspecies (protonation-state) enumeration + transformed free energy at a pH.

Standard workflow (user): don't fix ONE protonation state — enumerate the microspecies
populated at the target pH, compute each (conformer ensemble → relax → Boltzmann), and
combine them. We use the TRANSFORMED free energy (Alberty/eQuilibrator): write the
reaction with H+ IMPLICIT and give each species a pH-dependent free energy

    G'(species) = -RT ln Σ_i exp( -[ G_i - N_H(i)·g_proton(pH) ] / RT )

where the sum runs over microspecies i, G_i is the QM free energy (elec+solv+thermal),
N_H(i) is the microspecies' total H-atom count, and g_proton(pH) is the aqueous proton
free energy. This handles BOTH microspecies averaging AND proton release/uptake in one
formula — no manual n_H+ term (a deprotonation appears as a lower-N_H microspecies).

ENUMERATION here uses Dimorphite-DL (swap for cxcalc/ML-pKa if it misses states). The
QM computes the RELATIVE energies, so we rely on the tool for coverage, not pKa accuracy.
"""
import os

from rdkit import Chem

T = 298.15
RT = 8.314e-3 * T                                  # kJ/mol
# aqueous proton free energy at pH (step3b/step6 convention): G_H_gas + ΔGsolv_H - RT ln10·pH
G_H_GAS, DGSOLV_H = -26.3, -1104.5


def g_proton(pH):
    return G_H_GAS + DGSOLV_H - 2.303 * RT * pH    # ~ -1170.7 kJ/mol at pH 7


def _n_hydrogens(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return sum(a.GetTotalNumHs() + (1 if a.GetSymbol() == "H" else 0) for a in m.GetAtoms())


def enumerate_microspecies(smi, ph=7.0, window=1.0, max_states=4):
    """Candidate microspecies near `ph` as [(smiles, formal_charge, n_H)], deduped and
    capped to the `max_states` closest to neutrality of the enumerator's pH window.
    Falls back to the input species if Dimorphite is unavailable or returns nothing."""
    try:
        from dimorphite_dl import protonate_smiles
        outs = protonate_smiles(smi, ph_min=ph - window, ph_max=ph + window,
                                max_variants=max_states * 2)
    except Exception:
        outs = []
    states = []
    seen = set()
    for s in (outs or []):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        can = Chem.MolToSmiles(m)
        if can in seen:
            continue
        seen.add(can)
        q = Chem.GetFormalCharge(m)
        nh = _n_hydrogens(can)
        states.append((can, q, nh))
    if not states:                                  # fallback: keep the given species
        m = Chem.MolFromSmiles(smi)
        q = Chem.GetFormalCharge(m) if m is not None else 0
        states = [(smi, q, _n_hydrogens(smi) or 0)]
    return states[:max_states]


def transformed_G(micro_G, ph=7.0):
    """Combine per-microspecies (n_H, G_i) into the species' transformed free energy
    G'(pH) = -RT ln Σ exp(-[G_i - n_H·g_proton]/RT). micro_G = list of (n_H, G_i_kJ)."""
    import math
    gp = g_proton(ph)
    prime = [G - nH * gp for nH, G in micro_G if G is not None]
    if not prime:
        return None
    lo = min(prime)
    Z = sum(math.exp(-(p - lo) / RT) for p in prime)
    return lo - RT * math.log(Z)


if __name__ == "__main__":
    for name, smi in [("methylphosphate", "COP(=O)([O-])[O-]"),
                      ("triphosphate-Me", "COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])O"),
                      ("acetate", "CC(=O)[O-]"),
                      ("MNA+", "C[n+]1cccc(C(N)=O)c1")]:
        print(f"  {name:16s} -> " +
              " | ".join(f"{s} q{q:+d} nH{nh}" for s, q, nh in enumerate_microspecies(smi)))
