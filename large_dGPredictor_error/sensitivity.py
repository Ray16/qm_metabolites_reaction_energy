#!/usr/bin/env python
"""Where does the remaining 34 kJ/mol live? Sensitivity of dG'^o to the transform
parameters we do not compute from first principles.

Two knobs enter the transform as fitted/assumed constants rather than QM results:

  proton_reference_kJ  -- the aqueous-proton free energy on the QM total-energy
                          scale. Enters a reaction as -dN_H * mu_H, so it biases
                          ONLY reactions with a net proton change.
  ionic_strength_M     -- enters through Debye-Huckel via d(z^2 - N_H), so it
                          biases reactions with a net charge/H change.

If the redox failures were driven by either, a modest change would fix them
without touching the QM. This reports the derivative and the best-fit value, so
we can tell "systematic we can remove" from "real QM error".

Run:  /homes/rzhu/miniforge3/envs/palm/bin/python sensitivity.py
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
sys.path.insert(0, THERMO)
from qm_thermo import config                                          # noqa: E402
from qm_thermo.reactions import Reaction, SpeciesInfo, reaction_dG    # noqa: E402

BREAKDOWN = os.path.join(THERMO, "uma_workflow", "G_aq_ensemble_fast.json")
RXN_CSV = os.path.join(HERE, "top10_reactions_stereo_significant.csv")


def load():
    bd = json.load(open(BREAKDOWN))
    spec = json.load(open(os.path.join(HERE, "species.json")))
    reactions = json.load(open(os.path.join(HERE, "reactions.json")))
    G_aq = {c: r["G_aq_kJ"] for c, r in bd.items()}
    species = {c: SpeciesInfo(c, n_hydrogens=int(v["n_hydrogens"]),
                              charge=int(v["charge"])) for c, v in spec.items()}
    meta = {r["modelseed_rxn"]: r for r in csv.DictReader(open(RXN_CSV))}
    return G_aq, species, reactions, meta


def evaluate(G_aq, species, reactions, meta, *, mu_H=None, I=None):
    """dG'^o per reaction at each reaction's measured pH, with optional overrides."""
    base = config.DEFAULT_CONDITIONS
    out = {}
    for rid, stoich in reactions.items():
        m = meta[rid]
        pH = (float(m["pH_min"]) + float(m["pH_max"])) / 2.0
        cond = config.Conditions(
            pH=pH,
            ionic_strength_M=base.ionic_strength_M if I is None else I,
            proton_reference_kJ=(base.proton_reference_kJ if mu_H is None else mu_H),
        )
        out[rid] = reaction_dG(Reaction(rid, stoich), G_aq, species,
                               conditions=cond).dG_transformed_kJ
    return out


def main():
    G_aq, species, reactions, meta = load()
    exp = {r: float(m["tecrdb_dG_kJ"]) for r, m in meta.items() if r in reactions}
    base_mu = config.DEFAULT_CONDITIONS.proton_reference_kJ
    base_I = config.DEFAULT_CONDITIONS.ionic_strength_M

    def mae(pred):
        return np.mean([abs(pred[r] - exp[r]) for r in reactions])

    ref = evaluate(G_aq, species, reactions, meta)
    print(f"baseline (pH-matched)              MAE = {mae(ref):.1f} kJ/mol\n")

    # ---- proton reference ----
    print("proton reference mu_H (kJ/mol):")
    best = (None, 1e9)
    for d in (-20, -10, -5, 0, 5, 10, 20):
        m = mae(evaluate(G_aq, species, reactions, meta, mu_H=base_mu + d))
        print(f"   {base_mu + d:9.1f} ({d:+3d})   MAE = {m:6.1f}")
        if m < best[1]:
            best = (base_mu + d, m)
    # fine scan for the optimum
    grid = np.arange(base_mu - 30, base_mu + 30, 0.5)
    maes = [mae(evaluate(G_aq, species, reactions, meta, mu_H=v)) for v in grid]
    k = int(np.argmin(maes))
    print(f"   best-fit mu_H = {grid[k]:.1f} ({grid[k] - base_mu:+.1f})  "
          f"MAE = {maes[k]:.1f}")

    # ---- ionic strength ----
    print("\nionic strength I (M):")
    for I in (0.0, 0.1, 0.25, 0.5, 1.0):
        print(f"   {I:5.2f}   MAE = {mae(evaluate(G_aq, species, reactions, meta, I=I)):6.1f}")

    # ---- which reactions are even sensitive? ----
    print("\nper-reaction sensitivity (kJ/mol of dG per unit change):")
    up_mu = evaluate(G_aq, species, reactions, meta, mu_H=base_mu + 10)
    up_I = evaluate(G_aq, species, reactions, meta, I=base_I + 0.25)
    print(f"{'rxn':10}{'err':>8}{'d/dmuH(+10)':>13}{'d/dI(+0.25)':>13}")
    for rid in sorted(reactions, key=lambda r: -abs(ref[r] - exp[r])):
        print(f"{rid:10}{ref[rid] - exp[rid]:8.1f}{up_mu[rid] - ref[rid]:13.1f}"
              f"{up_I[rid] - ref[rid]:13.1f}")


if __name__ == "__main__":
    main()
