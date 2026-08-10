#!/usr/bin/env python
"""eQuilibrator ΔrG'° for the full TECRDB-full reaction set, at the same
conditions as the QC composite (pH 7, I=0.25 M, 298.15 K).

Records, per reaction: eQuilibrator dG'° and uncertainty, or the failure reason.
Reactions eQuilibrator CANNOT score (unresolved compound / no group decomposition)
are the regime where the QC method is uniquely needed, so they are reported too.
"""
from __future__ import annotations
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results", "eq", "equilibrator_full.json")


def main():
    from equilibrator_api import ComponentContribution, Q_
    cc = ComponentContribution()
    cc.p_h = Q_(7.0)
    cc.ionic_strength = Q_("0.25M")
    cc.temperature = Q_("298.15K")

    reactions = json.load(open(os.path.join(HERE, "tecrdb_full_reactions.json")))
    m2k = json.load(open(os.path.join(HERE, "modelseed_to_kegg.json")))
    out = json.load(open(OUT)) if os.path.isfile(OUT) else {}

    n_ok = n_fail = 0
    for i, (rid, st) in enumerate(reactions.items()):
        if rid in out:
            continue
        if not all(c in m2k for c in st):
            out[rid] = {"dG_kJ": None, "error": "no KEGG mapping for a participant"}
            n_fail += 1
            continue
        # build "coeff kegg:C##### = ..." ; proton already excluded upstream
        left = " + ".join(f"{-v:g} kegg:{m2k[c]}" for c, v in st.items() if v < 0)
        right = " + ".join(f"{v:g} kegg:{m2k[c]}" for c, v in st.items() if v > 0)
        formula = f"{left} = {right}"
        try:
            rxn = cc.parse_reaction_formula(formula)
            if not rxn.is_balanced():
                # eQuilibrator can still transform; note it but proceed
                pass
            dg = cc.standard_dg_prime(rxn)
            out[rid] = {"dG_kJ": float(dg.value.m_as("kJ/mol")),
                        "sd_kJ": float(dg.error.m_as("kJ/mol")),
                        "balanced": bool(rxn.is_balanced())}
            n_ok += 1
        except Exception as e:
            out[rid] = {"dG_kJ": None, "error": f"{type(e).__name__}: {str(e)[:120]}"}
            n_fail += 1
        if (i + 1) % 25 == 0:
            json.dump(out, open(OUT, "w"), indent=1)
            print(f"  {i+1}/{len(reactions)}  ok={n_ok} fail={n_fail}", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    covered = sum(1 for v in out.values() if v.get("dG_kJ") is not None)
    print(f"DONE: {len(out)} reactions, eQuilibrator covers {covered} "
          f"({covered/len(out):.0%}), cannot score {len(out)-covered}", flush=True)


if __name__ == "__main__":
    main()
