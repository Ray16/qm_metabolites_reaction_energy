"""General reactive-core localization (the GENERAL heuristic behind cofactor_ring / thiol-cap).

Principle: localize the calculation to the transformation and its immediate chemical environment,
then replace the CONSERVED distal scaffold with a uniform methyl cap. Because the scaffold is cut
identically on the matched reactant/product, it cancels in ΔG -- an isodesmic substitution, no
per-cofactor code. Subsumes: spectator truncation, cofactor_ring (NAD tail -> ring), thiol-cap
(GSH peptide -> cysteine).

Why the existing truncate.py fails here: its `reaction_center` correctly excludes the changed
region (bond-order MCS), but the MIRROR step (`keep_r |= images of keep_p`) transitively re-grows
to cover the whole shared scaffold, so a redox cofactor is kept whole. Here we grow each species
INDEPENDENTLY from its own changed atoms (no mirror), then VERIFY atom+charge balance -- the
symmetric cut of a conserved scaffold balances by construction; if it doesn't, we refuse.

Handles multi-coefficient couples (e.g. 2 GSH -> GSSG) by expanding to unit species and pairing.
Returns (new_species, n_Hplus) or None (caller falls back). NOT fitted to any data.
"""
from collections import Counter
from rdkit import Chem

from truncate import (mcs_atom_map, reaction_center, grow, truncate_species,
                      formula_charge)


def _units(species):
    """Expand {name:(coeff,q,smi)} into per-molecule units [(name,side,q,smi)], side=+1 prod/-1 react."""
    out = []
    for name, (c, q, s) in species.items():
        side = 1 if c > 0 else -1
        for k in range(abs(int(c))):
            out.append((f"{name}#{k}" if abs(c) > 1 else name, side, q, s))
    return out


def _core(smi_from, smi_to, radius):
    """Localized capped SMILES of `smi_from` = its changed atoms (vs its partner `smi_to`) grown by
    `radius`, with the conserved remainder methyl-capped. Independent grow (no mirror over-growth)."""
    a = Chem.MolFromSmiles(smi_from); b = Chem.MolFromSmiles(smi_to)
    if a is None or b is None:
        return None
    amap, n = mcs_atom_map(a, b)
    center = reaction_center(a, amap, b)
    if not center:                                    # nothing changed on this side (pure spectator)
        return smi_from                               # keep as-is; balance handled globally
    keep = grow(a, center, radius)
    capped, _ = truncate_species(smi_from, keep)
    return capped


def localize_reaction(species, radius=2):
    """General reactive-core localization. Returns (new_species_dict, n_Hplus) or None."""
    units = _units(species)
    react = [u for u in units if u[1] < 0]
    prod = [u for u in units if u[1] > 0]
    if not react or not prod:
        return None
    # greedy max-MCS pairing between reactant and product units
    R = [Chem.MolFromSmiles(u[3]) for u in react]
    P = [Chem.MolFromSmiles(u[3]) for u in prod]
    if any(m is None for m in R + P):
        return None
    scores = []
    for i, r in enumerate(R):
        for j, p in enumerate(P):
            _, n = mcs_atom_map(r, p)
            scores.append((n, i, j))
    scores.sort(reverse=True)
    partner = {}                                      # unit-index (react) -> unit-index (prod) and vice versa
    ur, up = set(), set()
    for n, i, j in scores:
        if i in ur or j in up:
            continue
        ur.add(i); up.add(j); partner[("r", i)] = j; partner[("p", j)] = i
    if len(ur) != len(react) or len(up) != len(prod):
        return None                                   # unequal sides / unpaired -> refuse
    # localize every unit against its partner
    new_units = []
    for i, (name, side, q, smi) in enumerate(react):
        core = _core(smi, prod[partner[("r", i)]][3], radius)
        if core is None:
            return None
        new_units.append((name, -1, core))
    for j, (name, side, q, smi) in enumerate(prod):
        core = _core(smi, react[partner[("p", j)]][3], radius)
        if core is None:
            return None
        new_units.append((name, 1, core))
    # collapse identical (side, SMILES) units back into coefficients
    coeff = Counter()
    smiles = {}
    for name, side, smi in new_units:
        coeff[(side, smi)] += side
        smiles[(side, smi)] = smi
    # balance: sum coeff*H + n_H+ == 0 and sum coeff*q + n_H+ == 0 must be consistent
    Hbal = 0; Qbal = 0
    new_species = {}
    idx = 0
    for (side, smi), c in coeff.items():
        f, q = formula_charge(smi)
        Hbal += c * f.get("H", 0)
        Qbal += c * q
        idx += 1
        new_species[f"core{idx}_t"] = [c, q, smi]
    nH = -Hbal
    if Qbal + nH != 0:                                # charge & H can't both balance -> refuse
        return None
    return new_species, int(nH)


if __name__ == "__main__":
    import json, sys
    d = json.load(open("scripts/reactions_tecrdb_all.json")) if len(sys.argv) < 2 else json.load(open(sys.argv[1]))
    for rid in ["rxn00810", "rxn01011", "rxn00086"]:
        if rid not in d:
            continue
        sp = {n: tuple(v) for n, v in d[rid]["species"].items()}
        out = localize_reaction(sp, radius=2)
        print(f"\n=== {rid}: {d[rid]['note'][8:45]} ===")
        if out is None:
            print("  REFUSED (falls back)")
            continue
        ns, nH = out
        for n, (c, q, s) in ns.items():
            print(f"   c={c:+d} q={q:+d}  {s}")
        print("   n_Hplus =", nH)
