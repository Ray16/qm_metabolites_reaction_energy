#!/usr/bin/env python
"""Route 2: can QC pick the right microspecies from relative energies?

Scores curated tautomer / hydration / anomer equilibria. Each pair differs by a
proton shift, a water addition, or a ring configuration, so the (multiply-)
charged-anion solvation error that sinks absolute dG should largely CANCEL --
this is the regime where QC should be usable and where group-contribution
methods are blind.

dG(pair) = G_aq(product) - G_aq(reactant) [ - G_aq(water) if the product added
one]. Compared against curated experimental dG. Two things matter: the SIGN
(did QC pick the right dominant species) and the magnitude error.
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); THERMO = os.path.dirname(HERE)
sys.path.insert(0, THERMO)
from qm_thermo import config
from qm_thermo.composite import extract_ensemble_energy
from qm_thermo.reactions import Reaction, SpeciesInfo, reaction_dG

C = config.DEFAULT_CONDITIONS
RT = C.R_kJ * C.temperature_K
# Water is the solvent: its activity is unity (~55.5 M), not the 1 M solute
# standard state the composite is built on. dfG'(H2O) enters at that activity,
# so a solute-referenced G(H2O) must be shifted up by RT ln(55.5).
WATER_ACTIVITY_KJ = RT * math.log(55.5)


def g(bd, sid):
    rec = bd[sid]
    return (extract_ensemble_energy(rec, temperature_K=C.temperature_K).gibbs_kJ
            if "conformers" in rec else float(rec["G_aq_kJ"]))


def main():
    bd = json.load(open(os.path.join(THERMO, "mlip", "G_aq_speciation_val.json")))
    pairs = json.load(open(os.path.join(HERE, "speciation_val_pairs.json")))
    mets = {m["id"]: m for m in json.load(open(os.path.join(HERE, "speciation_val_metabolites.json")))}
    micro = json.load(open(os.path.join(THERMO, "mlip", "G_aq_microspecies.json")))

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    def nH(smiles):
        return sum(1 for a in Chem.AddHs(Chem.MolFromSmiles(smiles)).GetAtoms()
                   if a.GetSymbol() == "H")

    WATER = "__h2o__"
    G = {WATER: float(micro["h2o"]["G_aq_kJ"]) + WATER_ACTIVITY_KJ}
    S = {WATER: SpeciesInfo(WATER, n_hydrogens=2, charge=0)}
    for sid, m in mets.items():
        G[sid] = g(bd, sid)
        S[sid] = SpeciesInfo(sid, n_hydrogens=nH(m["smiles"]), charge=int(m["charge"]))

    rows = []
    for p in pairs:
        stoich = {p["reactant"]: -1.0, p["product"]: 1.0}
        if p["add_water"]:
            stoich[WATER] = stoich.get(WATER, 0.0) - 1.0     # A + H2O -> A(OH)2
        dG = reaction_dG(Reaction(p["id"], stoich), G, S, conditions=C).dG_transformed_kJ
        rows.append({**p, "dG_qc": dG, "err": dG - p["dG_ref_kJ"],
                     "sign_ok": (dG > 0) == (p["dG_ref_kJ"] > 0) or abs(p["dG_ref_kJ"]) < 1.0})

    print(f"{'pair':16s} {'class':10s} {'QC dG':>8s} {'exp dG':>8s} {'err':>7s} {'frac_maj_QC':>11s} {'frac_maj_exp':>12s}")
    for r in sorted(rows, key=lambda r: r["class"]):
        fq = 1 / (1 + np.exp(-abs(r["dG_qc"]) / RT))     # fraction in the majority species
        fe = 1 / (1 + np.exp(-abs(r["dG_ref_kJ"]) / RT))
        mark = "" if r["sign_ok"] else "  <-- WRONG species"
        print(f"{r['id']:16s} {r['class']:10s} {r['dG_qc']:8.1f} {r['dG_ref_kJ']:8.1f} "
              f"{r['err']:7.1f} {fq:11.2f} {fe:12.2f}{mark}")

    err = [r["err"] for r in rows]
    print(f"\nMAE {np.mean(np.abs(err)):.1f}  RMSE {np.sqrt(np.mean(np.square(err))):.1f}  "
          f"signs correct {sum(r['sign_ok'] for r in rows)}/{len(rows)}")
    for cls in ("hydration", "tautomer", "anomer"):
        sub = [r for r in rows if r["class"] == cls]
        if sub:
            print(f"  {cls:10s} n={len(sub)}  MAE {np.mean([abs(r['err']) for r in sub]):5.1f}  "
                  f"signs {sum(r['sign_ok'] for r in sub)}/{len(sub)}")
    out = os.path.join(THERMO, "results", "benchmark", "speciation_scored.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
