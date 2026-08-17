"""Error histogram over ALL TECRDB reactions: x = signed error (dG_pred - dG_exp), y = # reactions.
Overlaid UMA full pipeline (pH-0 + COFACTOR_RING + truncation) vs retrained-dGPredictor.
One consistent experimental reference for both. dpi 300, no baked-in captions.
"""
import os, re, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..")                                  # qm_mlip_solvation
THERMO = os.path.join(EXP, "..", "..")
FIG = os.path.join(EXP, "figures", "error_histogram_uma_vs_dgp.png")

import importlib.util
spec = importlib.util.spec_from_file_location("pfa", os.path.join(HERE, "ph0_final_analysis.py"))
pfa = importlib.util.module_from_spec(spec); spec.loader.exec_module(pfa)
d = pfa.d                                                       # reactions_tecrdb_all.json


def _dG(sub, rid):
    p = os.path.join(EXP, "logs", sub, f"{rid}.log")
    if not os.path.exists(p):
        return None
    m = re.search(r"ΔG = ([+-]?\d+\.\d+)", open(p, errors="ignore").read())
    return float(m.group(1)) if m else None


def is_redox(rid):
    smis = " ".join(s[2] for s in d[rid]["species"].values()); note = d[rid]["note"].lower()
    return ("c1cc[n+]" in smis.lower() and "C(N)=O" in smis) or any(
        k in note for k in ["dehydrogenase", "reductase", "oxidase", "oxidoreductase"])


def uma_dG(rid):
    """Full pipeline: ringcofactor for redox, else gated pH-0 (baseline for isomerase / missing)."""
    if is_redox(rid):
        v = _dG("ringcofactor", rid)
        if v is not None and abs(v) < 1e5:
            return v
    if pfa.is_isomerization(d[rid]["species"]):
        return _dG("full367", rid)
    v = _dG("ph0_sweep", rid)
    return v if v is not None else _dG("full367", rid)


def main():
    exp = {r: v["exp"][0] for r, v in d.items()}               # TECRDB experiment (reactions_tecrdb_all)
    dgp = json.load(open(os.path.join(THERMO, "results", "eq", "dgpredictor_retrained_full.json")))
    uma_err, dgp_err = [], []
    for rid in d:
        u = uma_dG(rid)
        if u is None:
            continue
        ue = u - exp[rid]
        if abs(ue) > 200:                                      # drop QM garbage / loader failures
            continue
        if rid not in dgp or dgp[rid].get("dG_kJ") is None:
            continue
        uma_err.append(ue)
        dgp_err.append(dgp[rid]["dG_kJ"] - exp[rid])
    uma_err = np.array(uma_err); dgp_err = np.array(dgp_err)
    n = len(uma_err)
    mae = lambda x: np.mean(np.abs(x)); med = lambda x: np.median(np.abs(x))

    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    bins = np.arange(-60, 61, 4)
    ax.hist(np.clip(dgp_err, bins[0], bins[-1]), bins=bins, color="#D1495B", alpha=0.85,
            edgecolor="white", linewidth=0.4,
            label=f"dGPredictor (retrained)\nMAE {mae(dgp_err):.0f} · median {med(dgp_err):.0f} · bias {np.mean(dgp_err):+.0f}")
    ax.hist(np.clip(uma_err, bins[0], bins[-1]), bins=bins, histtype="step", lw=2.6, color="#1a7a6c",
            label=f"UMA pipeline (pH-0 + cofactor cores)\nMAE {mae(uma_err):.0f} · median {med(uma_err):.0f} · bias {np.mean(uma_err):+.0f}")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(r"error:  $\Delta_r G'^{\circ}_{\rm pred} - \Delta_r G'^{\circ}_{\rm exp}$  (kJ/mol)", fontsize=12)
    ax.set_ylabel("number of reactions", fontsize=12)
    ax.set_title(f"Prediction error over {n} TECRDB reactions", fontsize=13)
    ax.legend(frameon=False, fontsize=10.5, loc="upper right", bbox_to_anchor=(1.0, 1.0),
              handlelength=1.4, labelspacing=1.0)
    ax.set_xlim(-62, 62)
    ax.margins(y=0.02)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG, dpi=300, bbox_inches="tight")
    print(f"wrote {FIG}")
    print(f"n={n}  UMA MAE {mae(uma_err):.1f} med {med(uma_err):.1f}  |  dGP MAE {mae(dgp_err):.1f} med {med(dgp_err):.1f}")


if __name__ == "__main__":
    main()
