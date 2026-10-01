"""Three-way MetaG vs group contribution vs eQuilibrator on TECRDB: predicted vs experiment, per method, MAE.

Reference: the STANDARDIZED TECRDB reference (each measurement Legendre-transformed to pH 7, I = 0, no Mg2+;
experiments/qm_mlip_solvation/scripts/reactions_tecrdb_std.json). Inputs: MetaG ΔG from the production results
directory (METAG_RESULTS, default analysis/sweep_20261001/final); eQuilibrator and group contribution evaluated
at the same conditions (analysis/sweep_20261001/{eq,gc}_real_tecrdb_std.json). Common reaction set only.

Regimes (state in the caption, not on the figure): GC and eQuilibrator are IN-SAMPLE (fitted to TECRDB).
MetaG has no TECRDB-fitted parameters, but (i) its routing policy and rule choices were selected while
looking at TECRDB -> a DEVELOPMENT result, not held-out, and (ii) its hydration term uses a calibration fitted to
external hydration constants (HYDRATION_CAL). The standardized reference was built with eQuilibrator's species
data, so eQuilibrator on it is partly circular. The printed MAEs are the only quantitative statement.
"""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
                     "xtick.labelsize": 18, "ytick.labelsize": 18, "figure.dpi": 300, "savefig.dpi": 300})
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                    # thermodynamic_calc

# ONE reference for every panel: the STANDARDIZED TECRDB reference (pH 7, I = 0, no Mg2+), with MetaG's ΔG read
# from the production results and eQuilibrator / group contribution evaluated at the same conditions.
import glob, statistics
RES = os.environ.get("METAG_RESULTS", os.path.join(HERE, "sweep_20261001", "final"))
metag = {json.load(open(f))["reaction"]: json.load(open(f))["dG"] for f in glob.glob(os.path.join(RES, "*.json"))}
std = json.load(open(os.path.join(ROOT, "experiments", "qm_mlip_solvation", "scripts", "reactions_tecrdb_std.json")))
exp = {k: statistics.median(v["exp"]) for k, v in std.items()}
eq = {r: v["dG_kJ"] for r, v in json.load(open(os.path.join(HERE, "sweep_20261001", "eq_real_tecrdb_std.json")))["predictions"].items()}
gc = {r: v["dG_kJ"] for r, v in json.load(open(os.path.join(HERE, "sweep_20261001", "gc_real_tecrdb_std.json")))["predictions"].items()}
K = [k for k in metag if metag[k] is not None and k in exp and k in gc and k in eq and abs(gc[k]) < 1e6 and abs(eq[k]) < 1e6]
M_LAB = "MetaG\n(no TECRDB-fitted parameters; development)"
preds = {M_LAB: np.array([metag[k] for k in K]),
         "Group Contribution\n(in-sample)": np.array([gc[k] for k in K]),
         "eQuilibrator\n(in-sample)": np.array([eq[k] for k in K])}
E = np.array([exp[k] for k in K])
COL = {M_LAB: "#009E73",
       "Group Contribution\n(in-sample)": "#0072B2", "eQuilibrator\n(in-sample)": "#D55E00"}

lim = [np.percentile(np.concatenate([E]+list(preds.values())), 1) - 20,
       np.percentile(np.concatenate([E]+list(preds.values())), 99) + 20]
bins = np.linspace(lim[0], lim[1], 34)
fig = plt.figure(figsize=(20, 7.6))
outer = fig.add_gridspec(1, 3, wspace=0.28)
for i, (name, P) in enumerate(preds.items()):
    inner = outer[i].subgridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(inner[1, 0])
    axtop = fig.add_subplot(inner[0, 0], sharex=ax)
    axright = fig.add_subplot(inner[1, 1], sharey=ax)
    err = np.abs(P - E)
    # scatter + parity + ±20 band
    ax.fill_between(lim, [lim[0]-20, lim[1]-20], [lim[0]+20, lim[1]+20], color="0.88", zorder=0)
    ax.plot(lim, lim, "--", color="black", lw=1.2, zorder=1)
    ax.scatter(E, P, s=38, color=COL[name], edgecolor="black", linewidth=0.3, alpha=0.75, zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("experiment ΔrG′°  (kJ/mol)")
    if i == 0:
        ax.set_ylabel("predicted ΔrG′°  (kJ/mol)")
    ax.text(0.04, 0.96, f"MAE {err.mean():.1f}\nmedian {np.median(err):.1f}\nn={len(K)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=18,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    # both marginals in the method colour: top = experiment (x), right = predicted (y)
    axtop.hist(E, bins=bins, color=COL[name], edgecolor="white", linewidth=0.3)
    axright.hist(P, bins=bins, orientation="horizontal", color=COL[name], edgecolor="white", linewidth=0.3)
    axtop.set_title(name, fontsize=18)
    for a in (axtop, axright):                          # marginals show SHAPE only; drop ticks (keeps 18pt rule clean)
        a.axis("off")
    # dashed median guides so the reader sees the experiment-vs-predicted spread mismatch
    axtop.axvline(np.median(E), color="0.25", ls=":", lw=1)
    axright.axhline(np.median(P), color="0.25", ls=":", lw=1.2)
out = os.path.join(HERE, "..", "figures", "tecrdb_threeway.png")
fig.savefig(out, bbox_inches="tight")
print(f"n={len(K)}  MAE: MetaG {np.abs(preds[list(preds)[0]]-E).mean():.1f}  "
      f"GC {np.abs(preds[list(preds)[1]]-E).mean():.1f}  eQ {np.abs(preds[list(preds)[2]]-E).mean():.1f}")
print(f"wrote {os.path.abspath(out)}")
