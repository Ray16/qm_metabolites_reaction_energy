"""UQ calibration figures from artifacts/sigma_class_calibrated.json (written by
tools/calibrate_uncertainty.py). Two SINGLE-PANEL PNGs, no titles, font >= 18, dpi 300:

  figures/uq_error_vs_sigma.png  — |predicted-experiment| vs the reaction's calibrated sigma_class,
                                    per reaction, with the 1-sigma and 2-sigma guide lines. Shows the
                                    calibrated bar CONTAINS the error (points fall below 2*sigma).
  figures/uq_sigma_by_class.png  — calibrated sigma_class per mechanism class (sorted), n annotated.
                                    Shows which classes the model honestly flags as uncertain
                                    (phosphagen / CoA / glycosyl wide; isomerase / clean tight).

ZERO GPU. Run after tools/calibrate_uncertainty.py.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")

# repo figure style: base font 18, dpi 300 (ModelSEED_FAISS/CLAUDE.md)
_STYLE = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
                      "rxnfp_clustering", "scripts")
try:
    sys.path.insert(0, _STYLE)
    import _style  # noqa: F401  (sets rcParams)
except Exception:
    matplotlib.rcParams.update({"font.size": 18, "savefig.dpi": 300, "figure.dpi": 300})
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
CALIB = os.path.join(EXP, "artifacts", "sigma_class_calibrated.json")
FIGDIR = os.path.join(EXP, "figures")


def main():
    c = json.load(open(CALIB))
    os.makedirs(FIGDIR, exist_ok=True)
    per = c["per_reaction"]
    sig = np.array([r["sigma"] for r in per])
    aerr = np.array([abs(r["err"]) for r in per])

    # --- Figure A: |err| vs sigma_class, with 1x/2x guide lines ---
    fig, ax = plt.subplots(figsize=(8.5, 7))
    jit = sig + np.linspace(-0.4, 0.4, len(sig))            # tiny spread so class columns are visible
    ax.scatter(jit, aerr, s=34, alpha=0.55, color="#3b6fb0", edgecolor="none")
    hi = float(max(sig.max(), aerr.max())) * 1.05
    xs = np.array([0, hi])
    ax.plot(xs, xs, "-", color="#444444", lw=2, label="|err| = σ")
    ax.plot(xs, 2 * xs, "--", color="#999999", lw=2, label="|err| = 2σ")
    within2 = float((aerr <= 2 * sig).mean()) * 100
    ax.set_xlabel("calibrated σ$_{class}$  (kJ/mol)")
    ax.set_ylabel("|predicted − experiment|  (kJ/mol)")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, aerr.max() * 1.05)
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.97, 0.03, f"{within2:.0f}% within 2σ", transform=ax.transAxes,
            ha="right", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "uq_error_vs_sigma.png"), bbox_inches="tight")
    plt.close(fig)

    # --- Figure B: sigma_class per class, sorted, n annotated ---
    classes = c["classes"]
    items = sorted(classes.items(), key=lambda kv: kv[1]["sigma"])
    labels = [k for k, _ in items]
    sigmas = [v["sigma"] for _, v in items]
    ns = [v["n"] for _, v in items]
    fig, ax = plt.subplots(figsize=(10, 8))
    ypos = np.arange(len(labels))
    ax.barh(ypos, sigmas, color="#3b6fb0", alpha=0.85)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("calibrated σ$_{class}$  (kJ/mol)")
    for y, s, n in zip(ypos, sigmas, ns):
        ax.text(s + 0.4, y, f"n={n}", va="center", ha="left")
    ax.set_xlim(0, max(sigmas) * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "uq_sigma_by_class.png"), bbox_inches="tight")
    plt.close(fig)

    print(f"wrote figures/uq_error_vs_sigma.png ({within2:.0f}% within 2σ) and figures/uq_sigma_by_class.png")
    print(f"CV coverage (from artifact): 1σ={c.get('cv_coverage_1sigma')}  2σ={c.get('cv_coverage_2sigma')}")


if __name__ == "__main__":
    main()
