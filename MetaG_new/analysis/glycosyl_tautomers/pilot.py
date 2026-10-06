"""PILOT (non-invasive): does a tautomer ENSEMBLE lower the aqueous free energy of the freed
nucleobase enough to explain the glycosyl/PRT/nucleoside +~9 kJ bias?

The frozen pipeline scores each species from ONE input SMILES (pool_confs embeds a single topology;
no tautomer/anomer enumeration). If the freed base is scored as a single tautomer rather than the
solution mixture, its G is too HIGH by the configurational free energy -RT ln Z_taut, so a reaction
that RELEASES the base reads too endergonic (+bias). This measures that lowering per base, with NO
change to the pipeline and NO anchor.

G_taut uses the production implicit_G estimator: unique conformer minima, per-conformer solvation,
and thermal corrections. Ensemble: G_ens = -RT ln sum_i exp(-G_i/RT).
Lowering = G(input tautomer) - G_ens  (>=0; how much the single-SMILES scoring overestimates G).

Run: gpu_reserve run <idx> -- env PYTHONPATH=src <uma-python> analysis/glycosyl_tautomers/pilot.py
"""
import json
import os

import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from metag.energetics.uma import load_uma
from metag.energetics.microstates import combine
from metag.pipeline import implicit_G, effective_config, SpeciesRearranged

# the actual freed-base SMILES the pipeline uses (from the bundled benchmark), q=0 neutral bases
BASES = {
    "adenine":   "Nc1ncnc2[nH]cnc12",
    "guanine":   "Nc1nc2[nH]cnc2c(=O)[nH]1",
    "uracil":    "O=c1cc[nH]c(=O)[nH]1",
    "hypoxanthine": "O=c1[nH]cnc2[nH]cnc12",
    "xanthine":  "O=c1[nH]c(=O)c2[nH]cnc2[nH]1",
}


def enumerate_tautomers(smi, cap=6):
    """Canonical tautomer set (RDKit). Returns input-canonical first, then alternatives by score."""
    if cap is not None and cap < 1:
        raise ValueError("tautomer cap must leave room for the input state")
    te = rdMolStandardize.TautomerEnumerator()
    m = Chem.MolFromSmiles(smi)
    if m is None or Chem.GetFormalCharge(m) != 0:
        raise ValueError("pilot requires a valid neutral input SMILES")
    inp = Chem.MolToSmiles(m)
    taut_mols = te.Enumerate(m)
    ranked = sorted(taut_mols, key=lambda t: (-te.ScoreTautomer(t), Chem.MolToSmiles(t)))
    outs = [Chem.MolToSmiles(t) for t in ranked]
    # dedup, keep neutral only, cap
    uniq = []
    for s in [inp] + outs:
        mm = Chem.MolFromSmiles(s)
        if mm is None:
            continue
        if Chem.GetFormalCharge(mm) != 0:
            continue
        if s not in uniq:
            uniq.append(s)
    return uniq[:cap], inp


def score_species(pu, smi):
    """Use the same deduplicated, thermally resolved estimator as production."""
    warnings = []
    try:
        g, _ = implicit_G(pu, 0, smi, (1, 2), 10, 48, lambda *_: None, smi, warnings)
    except SpeciesRearranged:
        return None
    # A degraded/partial conformer ensemble cannot justify a tautomer ranking.
    return float(g) if g is not None and np.isfinite(g) and not warnings else None


def summarize(Gs, inp_canon):
    """Never replace an uncomputed input state with the lowest computed alternative."""
    if Gs.get(inp_canon) is None:
        return {"status": "input_state_failed", "lowering_kJ": None}
    good = {t: g for t, g in Gs.items() if g is not None}
    ensemble = combine({"name": t, "G": g} for t, g in good.items())
    return {"status": "complete_selected_set" if len(good) == len(Gs) else "partial_selected_set",
            "G_per_tautomer": good, "G_ensemble": ensemble["G"],
            "G_input_tautomer": good[inp_canon], "lowering_kJ": good[inp_canon] - ensemble["G"],
            "populations": ensemble["populations"], "failed_states": [t for t, g in Gs.items() if g is None]}


def main():
    pu = load_uma()
    out = {}
    for base, smi in BASES.items():
        tauts, inp_canon = enumerate_tautomers(smi)
        print(f"\n=== {base}  input={smi}  ({len(tauts)} tautomers) ===", flush=True)
        Gs = {}
        for t in tauts:
            g = score_species(pu, t)
            Gs[t] = g
            tag = "  <- INPUT" if t == inp_canon else ""
            print(f"   {t:40s} G={('%.1f'%g) if g is not None else 'FAIL':>8s} kJ{tag}", flush=True)
        summary = summarize(Gs, inp_canon)
        print(f"   --> {summary['status']}: lowering={summary['lowering_kJ']}", flush=True)
        out[base] = {"input_smiles": smi, "input_canonical": inp_canon,
                     "config": effective_config(), "tautomer_cap": 6,
                     "scope": "selected neutral tautomers only; no anomers or sugar ring forms",
                     **summary}
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilot_results.json"), "w"), indent=2)
    print("\n=== SUMMARY: tautomer-ensemble lowering per base (kJ; >0 means single-SMILES overestimates G) ===")
    for base, d in out.items():
        print(f"   {base:14s} {d['lowering_kJ']} ({d['status']})")
    print("\nLowerings describe only the selected, successfully scored neutral tautomers."
          "\nA null result does not exclude omitted tautomers or establish the cause of a reaction error.")


if __name__ == "__main__":
    main()
