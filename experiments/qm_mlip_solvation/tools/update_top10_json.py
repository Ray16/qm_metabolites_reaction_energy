"""Rebuild pipeline/current_pipeline_top10.json from the CURRENT pipeline results
(pH-0 + ring-cofactor + truncation). Extracts the predicted ΔG for each of the 8 unique
reactions: the 5 re-run in logs/top10_current/, the 3 already-current in logs/ph0_sweep/
(rxn00579/rxn01675/rxn01005 are non-redox -> pH-0 result == COFACTOR_RING result).
"""
import os, re, json

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..")                                   # qm_mlip_solvation
PIPE = os.path.join(EXP, "..", "..", "pipeline")                 # thermodynamic_calc/pipeline

SRC = {  # reaction -> which log dir holds its current-pipeline result
    # glutathione reductase: ring+thiol (both couples via canonical cores) -- the current mechanism
    "rxn00086": "ringthiol", "rxn00070": "ringthiol",
    "rxn00605": "top10_current", "rxn01713": "top10_current", "rxn01834": "top10_current",
    "rxn00579": "ph0_sweep", "rxn01675": "ph0_sweep", "rxn01005": "ph0_sweep",
}


def dG(rid, sub):
    p = os.path.join(EXP, "logs", sub, f"{rid}.log")
    if not os.path.exists(p):
        return None
    m = re.search(r"ΔG = ([+-]?\d+\.\d+)", open(p, errors="ignore").read())
    return round(float(m.group(1)), 1) if m else None


def main():
    out = {"_provenance": "CURRENT pipeline (pH-0 anion routing + nicotinamide ring-cofactor for "
           "redox + spectator truncation + convergent sampling + implicit solvation), 2026-08-16. "
           "redox rxn00086/00070 use COFACTOR_RING; nucleotidyl/glycosyl use PH0_AUTO. Reverses = -forward."}
    missing = []
    for rid, sub in SRC.items():
        v = dG(rid, sub)
        if v is None:
            missing.append(rid)
        else:
            out[rid] = v
    if missing:
        print(f"MISSING (not yet done): {missing} -- not writing")
        return
    json.dump(out, open(os.path.join(PIPE, "current_pipeline_top10.json"), "w"), indent=2)
    print("wrote current_pipeline_top10.json:")
    for k, v in out.items():
        if not k.startswith("_"):
            print(f"  {k}: {v:+.1f}")


if __name__ == "__main__":
    main()
