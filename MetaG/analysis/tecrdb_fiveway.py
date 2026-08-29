"""Five-way UMA vs dGPredictor(original) vs dGPredictor(retrained,held-out) vs GC vs eQ on TECRDB
(real experimental ground truth). Predicted vs experiment, per method, with MAE + in-sample/held-out tag.

Each panel is labelled with its evaluation MODE, because that is the whole point:
  dGP original    IN-SAMPLE (fixed published weights, trained on TECRDB) -> MAE 3.0 (the famous number)
  dGP retrained   HELD-OUT  (our CV)                                     -> MAE 6.5
  eQuilibrator    IN-SAMPLE (fit to TECRDB)                              -> MAE 8.0
  Group Contrib.  IN-SAMPLE (fit to TECRDB)                              -> MAE 9.3
  UMA (MetaG)     HELD-OUT  (first-principles, never fit to TECRDB)      -> MAE 11.6
HONEST: the additivity family owns TECRDB (its training domain) even held-out; UMA's edge is on the
ModelSEED frontier (reference_scoreboard), not here.
"""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "figure.dpi": 300, "savefig.dpi": 300})
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
DB = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/ModelSEEDDatabase/Biochemistry"

exp = json.load(open(os.path.join(ROOT, "results", "benchmark", "tecrdb_full_scored.json")))["experiment_kJ"]
uma_err = {r["rid"]: r["err"] for r in json.load(open(os.path.join(HERE, "..", "metag", "data", "sigma_class_calibrated.json")))["per_reaction"]}
dgp_orig = {r: v["dG_kJ"] for r, v in json.load(open(os.path.join(ROOT, "results", "eq", "dgpredictor_full.json"))).items() if v.get("dG_kJ") is not None}
dgp_ho_err = {r: v["dgp_heldout"] for r, v in json.load(open(os.path.join(ROOT, "experiments", "qm_mlip_solvation", "artifacts", "heldout_dgp_vs_uma.json"))).items()}
gc, eq = {}, {}
for f in glob.glob(f"{DB}/reaction_*.json"):
    for r in json.load(open(f)):
        t = r.get("thermodynamics") or {}
        if t.get("Group contribution"): gc[r["id"]] = t["Group contribution"][0]
        if t.get("eQuilibrator"): eq[r["id"]] = t["eQuilibrator"][0]

K = [k for k in uma_err if k in exp and k in dgp_orig and k in dgp_ho_err and k in gc and k in eq
     and abs(gc[k]) < 1e6 and abs(eq[k]) < 1e6]
E = np.array([exp[k] for k in K])
# (title, mode, predictions, colour) ordered best->worst MAE
PANELS = [
    ("dGPredictor (original)", "IN-SAMPLE", np.array([dgp_orig[k] for k in K]), "#CC79A7"),
    ("dGPredictor (retrained)", "HELD-OUT", np.array([dgp_ho_err[k] + exp[k] for k in K]), "#56B4E9"),
    ("eQuilibrator", "IN-SAMPLE", np.array([eq[k] for k in K]), "#D55E00"),
    ("Group Contribution", "IN-SAMPLE", np.array([gc[k] for k in K]), "#0072B2"),
    ("UMA (MetaG)", "HELD-OUT", np.array([uma_err[k] + exp[k] for k in K]), "#009E73"),
]
allv = np.concatenate([E] + [P for _, _, P, _ in PANELS])
lim = [np.percentile(allv, 1) - 20, np.percentile(allv, 99) + 20]
bins = np.linspace(lim[0], lim[1], 30)

fig = plt.figure(figsize=(30, 6.6))
outer = fig.add_gridspec(1, 5, wspace=0.30)
for i, (name, mode, P, col) in enumerate(PANELS):
    inner = outer[i].subgridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(inner[1, 0]); axtop = fig.add_subplot(inner[0, 0], sharex=ax); axr = fig.add_subplot(inner[1, 1], sharey=ax)
    err = np.abs(P - E)
    ax.fill_between(lim, [lim[0]-20, lim[1]-20], [lim[0]+20, lim[1]+20], color="0.88", zorder=0)
    ax.plot(lim, lim, "--", color="black", lw=1.1, zorder=1)
    ax.scatter(E, P, s=26, color=col, edgecolor="black", linewidth=0.25, alpha=0.75, zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("experiment ΔrG′°  (kJ/mol)")
    if i == 0: ax.set_ylabel("predicted ΔrG′°  (kJ/mol)")
    ax.text(0.05, 0.95, f"MAE {err.mean():.1f}\nmed {np.median(err):.1f}", transform=ax.transAxes,
            va="top", ha="left", fontsize=15, bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    axtop.hist(E, bins=bins, color=col, edgecolor="white", linewidth=0.3)
    axr.hist(P, bins=bins, orientation="horizontal", color=col, edgecolor="white", linewidth=0.3)
    tag = "  [in-sample]" if mode == "IN-SAMPLE" else "  [held-out]"
    axtop.set_title(name + "\n" + mode + (" · fit to TECRDB" if mode == "IN-SAMPLE" else " · fair"), fontsize=15)
    for a in (axtop, axr): a.axis("off")
fig.suptitle(f"TECRDB — five methods vs experiment (n={len(K)})", fontsize=18, y=1.02)
out = os.path.join(HERE, "..", "figures", "tecrdb_fiveway.png")
fig.savefig(out, bbox_inches="tight")
print("n=%d  MAE: " % len(K) + "  ".join(f"{n.split(' (')[0]} {np.abs(P-E).mean():.1f}" for n, _, P, _ in PANELS))
print(f"wrote {os.path.abspath(out)}")
