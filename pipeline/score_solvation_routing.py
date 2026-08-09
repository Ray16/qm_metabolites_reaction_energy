#!/usr/bin/env python
"""Evaluate ALPB->CPCM-X solvation-routing policies at the reaction level.

Rebuilds each species' Boltzmann-averaged G_aq with dGsolv taken from either the
ALPB baseline or the CPCM-X recompute according to a policy, then scores all
TECRDB-full reactions and reports MAE vs experiment. Whole-molecule solvation is
one method per molecule, so policies select which molecules use CPCM-X.
"""
from __future__ import annotations
import json, os, sys, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
sys.path.insert(0, THERMO)
from qm_thermo import config                                          # noqa: E402
from qm_thermo.reactions import Reaction, SpeciesInfo, reaction_dG    # noqa: E402

RT = 8.314462618e-3 * config.DEFAULT_CONDITIONS.temperature_K


def ensemble_G(conf_terms):
    """Boltzmann-average aqueous G, matching qm_thermo.thermo exactly:
    G = g_min - RT ln( sum_i exp(-(g_i - g_min)/RT) )."""
    vals = [e + s + t for e, s, t in conf_terms]
    vmin = min(vals)
    z = sum(math.exp(-(v - vmin) / RT) for v in vals)
    return vmin - RT * math.log(z)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaq", default=os.path.join(THERMO, "mlip", "G_aq_tecrdb_full.json"))
    ap.add_argument("--cpcmx", default=os.path.join(HERE, "cpcmx_dgsolv_tecrdb_full.json"))
    ap.add_argument("--reactions", default=os.path.join(HERE, "tecrdb_full_reactions.json"))
    ap.add_argument("--experiment", default=os.path.join(HERE, "tecrdb_full_experiment.json"))
    ap.add_argument("--species", default=os.path.join(HERE, "tecrdb_full_species.json"))
    ap.add_argument("--metabolites", default=os.path.join(HERE, "tecrdb_full_metabolites.json"))
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    gaq = json.load(open(args.gaq))
    cpcmx = json.load(open(args.cpcmx)) if os.path.isfile(args.cpcmx) else {}
    reactions = json.load(open(args.reactions))
    experiment = json.load(open(args.experiment))
    spec = json.load(open(args.species))
    mets = {m["id"]: m for m in json.load(open(args.metabolites))}
    S = {c: SpeciesInfo(c, n_hydrogens=int(v["n_hydrogens"]), charge=int(v["charge"]))
         for c, v in spec.items()}
    C = config.DEFAULT_CONDITIONS

    # per-species conformer terms with ALPB solvation
    def conf_terms(c, solv_override=None):
        out = []
        for i, cf in enumerate(gaq[c]["conformers"]):
            e = cf.get("E_elec_kJ", cf.get("E_UMA_kJ"))
            s = cf["dGsolv_kJ"]
            if solv_override is not None and i < len(solv_override) and solv_override[i] is not None:
                s = solv_override[i]
            out.append((e, s, cf["G_RRHO_kJ"]))
        return out

    # group membership for policies
    def groups(c):
        m = Chem.MolFromSmiles(mets[c]["smiles"])
        if m is None:
            return set()
        g = set()
        if m.HasSubstructMatch(Chem.MolFromSmarts("[P]")): g.add("P")
        if m.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX1H0-,OX2H1]")): g.add("carboxylate")
        if m.HasSubstructMatch(Chem.MolFromSmarts("[SX2H,SX1-]")): g.add("thiol")
        if m.HasSubstructMatch(Chem.MolFromSmarts("c[OX2H,OX1-]")): g.add("phenol")
        if m.HasSubstructMatch(Chem.MolFromSmarts("[NX4+,NX3;H2,H3]")): g.add("amine")
        return g

    grp = {c: groups(c) for c in gaq}
    have_cpcmx = {c for c in cpcmx if any(x is not None for x in cpcmx[c])}

    policies = {
        "ALPB baseline (all)": lambda c: False,
        "CPCM-X: all non-P w/ data": lambda c: c in have_cpcmx and "P" not in grp[c],
        "CPCM-X: carboxylate/phenol/thiol, no amine, no P":
            lambda c: c in have_cpcmx and (grp[c] & {"carboxylate", "phenol", "thiol"})
                      and "amine" not in grp[c] and "P" not in grp[c],
        "CPCM-X: phenol/thiol only (best pKa groups)":
            lambda c: c in have_cpcmx and (grp[c] & {"phenol", "thiol"}) and "P" not in grp[c],
    }

    exp = {r: experiment[r]["dG_kJ"] for r in reactions}
    results = {}
    for name, use in policies.items():
        G = {}
        for c in gaq:
            override = cpcmx.get(c) if use(c) else None
            G[c] = ensemble_G(conf_terms(c, override))
        scored, err = {}, {}
        for rid, st in reactions.items():
            if any(c not in G for c in st):
                continue
            dg = reaction_dG(Reaction(rid, st), G, S, conditions=C).dG_transformed_kJ
            scored[rid] = dg
            err[rid] = dg - exp[rid]
        a = np.array(list(err.values()))
        n_routed = sum(1 for c in gaq if use(c))
        results[name] = (float(np.abs(a).mean()), int((np.abs(a) > 50).sum()), n_routed, len(a))
        print(f"{name:52s} MAE {results[name][0]:5.1f}  |err|>50:{results[name][1]:3d}  "
              f"routed_species={n_routed:3d}  n_rxn={results[name][3]}")

    # per-reaction improvement of the best non-baseline policy vs baseline
    print("\n(baseline is the published ALPB composite; lower MAE = routing helps)")


if __name__ == "__main__":
    main()
