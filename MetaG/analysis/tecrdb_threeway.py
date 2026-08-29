"""Three-way UMA vs GC vs eQ on TECRDB -- the one dataset with REAL experimental ground truth (measured
Keq -> ΔrG'°). Predicted vs experiment, per method, with MAE.

HONEST FRAMING (put in the caption/slide, not baked on the figure): TECRDB is the incumbents' TRAINING
set -- GC and eQ errors here are IN-SAMPLE, so their low MAE is expected. UMA is first-principles, NOT fit
to TECRDB. The point is not "UMA wins" (it doesn't, on home turf) but that a from-scratch physics method
lands within ~3 kJ of methods fitted to this exact data -- and (separately, see reference_scoreboard) it
is the one that HOLDS on the ModelSEED frontier where the incumbents extrapolate and fail.

UMA (current pipeline): error from metag/data/sigma_class_calibrated.json per_reaction (logs/production).
GC/eQ: ModelSEED thermodynamics. experiment: results/benchmark/tecrdb_full_scored.json.
"""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "figure.dpi": 300, "savefig.dpi": 300})
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                    # thermodynamic_calc
DB = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/ModelSEEDDatabase/Biochemistry"

cal = json.load(open(os.path.join(HERE, "..", "metag", "data", "sigma_class_calibrated.json")))["per_reaction"]
uma_err = {r["rid"]: r["err"] for r in cal}                      # UMA - experiment (current pipeline)
exp = json.load(open(os.path.join(ROOT, "results", "benchmark", "tecrdb_full_scored.json")))["experiment_kJ"]
gc, eq = {}, {}
for f in glob.glob(f"{DB}/reaction_*.json"):
    for r in json.load(open(f)):
        if r["id"] in uma_err:
            t = r.get("thermodynamics") or {}
            if t.get("Group contribution"): gc[r["id"]] = t["Group contribution"][0]
            if t.get("eQuilibrator"): eq[r["id"]] = t["eQuilibrator"][0]
K = [k for k in uma_err if k in exp and k in gc and k in eq and abs(gc[k]) < 1e6 and abs(eq[k]) < 1e6]
E = np.array([exp[k] for k in K])
preds = {"UMA (MetaG)\nfirst-principles, NOT fit to TECRDB": np.array([uma_err[k] + exp[k] for k in K]),
         "Group Contribution\n(in-sample)": np.array([gc[k] for k in K]),
         "eQuilibrator\n(in-sample)": np.array([eq[k] for k in K])}
COL = {"UMA (MetaG)\nfirst-principles, NOT fit to TECRDB": "#009E73",
       "Group Contribution\n(in-sample)": "#0072B2", "eQuilibrator\n(in-sample)": "#D55E00"}

lim = [np.percentile(np.concatenate([E]+list(preds.values())), 1) - 20,
       np.percentile(np.concatenate([E]+list(preds.values())), 99) + 20]
fig, axes = plt.subplots(1, 3, figsize=(19, 6.6), sharex=True, sharey=True)
for ax, (name, P) in zip(axes, preds.items()):
    err = np.abs(P - E)
    ax.fill_between(lim, [lim[0]-20, lim[1]-20], [lim[0]+20, lim[1]+20], color="0.88", zorder=0)
    ax.plot(lim, lim, "--", color="black", lw=1.2, zorder=1)
    ax.scatter(E, P, s=40, color=COL[name], edgecolor="black", linewidth=0.3, alpha=0.75, zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_title(name, fontsize=17)
    ax.set_xlabel("experiment ΔrG′°  (kJ/mol)")
    ax.text(0.04, 0.96, f"MAE {err.mean():.1f}\nmedian {np.median(err):.1f}\nn={len(K)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=16,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
axes[0].set_ylabel("predicted ΔrG′°  (kJ/mol)")
fig.tight_layout()
out = os.path.join(HERE, "..", "figures", "tecrdb_threeway.png")
fig.savefig(out, bbox_inches="tight")
print(f"n={len(K)}  MAE: UMA {np.abs(preds[list(preds)[0]]-E).mean():.1f}  "
      f"GC {np.abs(preds[list(preds)[1]]-E).mean():.1f}  eQ {np.abs(preds[list(preds)[2]]-E).mean():.1f}")
print(f"wrote {os.path.abspath(out)}")
