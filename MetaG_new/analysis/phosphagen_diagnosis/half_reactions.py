"""Localize the phosphagen-kinase error with independent half-reactions (no new QM).

Creatine kinase = ATP hydrolysis - phosphocreatine hydrolysis. Both halves are rebuilt from the
frozen 2026-10-01c species cache (ALPB) on the same truncated/pH-0 cores the pipeline scores, with the
pipeline's Alberty transform, and compared with textbook pH-7 values. Also bounds how much the
unsourced first P-N pKa (P_N_LADDER[0] = 2.70) could contribute.

    python half_reactions.py [--cache DIR] [--water JSON] [--out JSON]
"""
import argparse
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "..", "..", "..", "MetaG", "analysis", "sweep_20261001")
RT_LN10 = 2.303 * 8.314e-3 * 298.15
PH = 7.0

CORES = {
    "MeP3": "COP(=O)(O)OP(=O)(O)OP(=O)(O)O", "MeP2": "COP(=O)(O)OP(=O)(O)O", "Pi": "O=P(O)(O)O",
    "guan": "CNC(N)=[NH2+]", "Pguan": "CNC(N)=[NH+]P(=O)(O)O",
    "guan_creatine": "CN(C)C(N)=[NH2+]", "Pguan_creatine": "CN(C)C(N)=[NH+]P(=O)(O)O",
}
LADDER = {"MeP3": [1.5, 1.5, 1.5, 6.5], "MeP2": [1.5, 1.5, 6.5], "Pi": [2.15, 7.20, 12.35],
          "P_N": [2.70, 4.58]}
REFERENCE = {"ATP hydrolysis": -36.0, "PCr hydrolysis": -43.0}   # textbook pH-7 values, approximate


def transform(react, prod):
    term = lambda pks: sum(math.log10(1 + 10 ** (PH - p)) for p in pks)
    return RT_LN10 * (sum(term(l) for l in react) - sum(term(l) for l in prod))


def load(cache, solv="alpb", physics="2026-10-01c"):
    want = {v: k for k, v in CORES.items()}
    G = {}
    for path in glob.glob(os.path.join(cache, "*.json")):
        d = json.load(open(path))
        st = d.get("settings", {})
        if d.get("smi") in want and d.get("method") == "implicit" and st.get("solv") == solv \
                and st.get("physics") == physics:
            G[want[d["smi"]]] = d["G"]
    missing = set(CORES) - set(G)
    if missing:
        raise SystemExit(f"cache lacks {sorted(missing)}")
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(SWEEP, "cache"))
    ap.add_argument("--water", default=os.path.join(SWEEP, "water_ref_G_expt.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "half_reactions.json"))
    a = ap.parse_args()
    G = load(a.cache)
    W = json.load(open(a.water))["G"]
    rows = {}
    ss = G["MeP2"] + G["Pi"] - G["MeP3"] - W
    pk = transform([LADDER["MeP3"]], [LADDER["MeP2"], LADDER["Pi"]])
    rows["ATP hydrolysis"] = {"species_sum": ss, "pka_transform": pk, "dG": ss + pk}
    for tag, g, pg in (("arginine-type", "guan", "Pguan"), ("creatine", "guan_creatine", "Pguan_creatine")):
        ss = G[g] + G["Pi"] - G[pg] - W
        pk = transform([LADDER["P_N"]], [LADDER["Pi"]])
        rows[f"PCr hydrolysis ({tag} core)"] = {"species_sum": ss, "pka_transform": pk, "dG": ss + pk}
    for k, r in rows.items():
        ref = REFERENCE["ATP hydrolysis" if k.startswith("ATP") else "PCr hydrolysis"]
        r.update(reference=ref, error=r["dG"] - ref)
    shift = {str(p): RT_LN10 * (math.log10(1 + 10 ** (PH - p)) - math.log10(1 + 10 ** (PH - 2.70)))
             for p in (2.70, 1.5, 1.0, 0.5)}
    out = {"physics": "2026-10-01c", "solv": "alpb", "half_reactions": rows,
           "P_N_pKa1_sensitivity_kJ": shift,
           "note": "PCr-hydrolysis error localizes the phosphagen bias to the phosphoguanidinium species; "
                   "the P-N pKa1 shift is bounded (<=~13 kJ for pKa1>=0.5) and its source is unconfirmed."}
    json.dump(out, open(a.out, "w"), indent=2)
    for k, r in rows.items():
        print(f"{k:34s} species_sum {r['species_sum']:+7.1f} pKa {r['pka_transform']:+5.1f} "
              f"dG {r['dG']:+7.1f} ref {r['reference']:+6.1f} err {r['error']:+6.1f}")
    print("P-N pKa1 shift:", {k: round(v, 1) for k, v in shift.items()})


if __name__ == "__main__":
    main()
