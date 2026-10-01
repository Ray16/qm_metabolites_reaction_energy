"""Five-way UMA vs dGPredictor(original) vs dGPredictor(retrained,held-out) vs GC vs eQ on TECRDB
(real experimental ground truth). Predicted vs experiment, per method, with MAE + in-sample/held-out tag.

Each panel is labelled with its evaluation MODE, because that is the whole point:
  dGP original    IN-SAMPLE (fixed published weights, trained on TECRDB)        -> MAE 3.0 (the famous number)
  dGP retrained   HELD-OUT  (5x5 family-grouped CV, groups=mechanism class)     -> MAE 8.3
  eQuilibrator    IN-SAMPLE (fit to TECRDB, GENUINE component-contribution run)  -> MAE 3.6
  Group Contrib.  IN-SAMPLE (fit to TECRDB, GENUINE pure group-contribution run) -> MAE 6.6
  MetaG           DEVELOPMENT (no TECRDB-fitted parameters, but routing policy chosen on TECRDB)
dGP HELD-OUT uses GroupKFold on the reaction's mechanism class (w.classify in heldout_dgp_vs_uma.py),
not plain random KFold: TECRDB has near-duplicate reactions within a class (same enzyme family,
near-identical substrates), so random folds leak near-twins across train/test and understate the honest
error (random-fold CV gave 6.5; family-grouped gives the stricter, honest 8.3).
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
# prefer the CURRENT merged UMA errors (stale-for-untouched + fresh-re-score for the 29 affected);
# fall back to the raw (stale) calibration if the merge hasn't been produced yet.
_cur = os.path.join(HERE, "uma_tecrdb_current.json")
if os.path.exists(_cur):
    uma_err = json.load(open(_cur))
    print("[using CURRENT merged UMA errors]")
else:
    uma_err = {r["rid"]: r["err"] for r in json.load(open(os.path.join(HERE, "..", "metag", "data", "sigma_class_calibrated.json")))["per_reaction"]}
    print("[WARNING: using STALE calibration UMA errors -- run merge_current_uma.py]")
dgp_orig = {r: v["dG_kJ"] for r, v in json.load(open(os.path.join(ROOT, "results", "eq", "dgpredictor_full.json"))).items() if v.get("dG_kJ") is not None}
# dGP shown two ways for EACH of two variants: in-sample (leaked) and family-grouped-CV held-out (fair).
#   retrained (Freiburger, ModelSEED vocab): in-sample dgpredictor_retrained_full; fair dgp_retrained_heldout.
#   original (published KEGG vocab): in-sample dgpredictor_full (published weights); fair = original vocab
#   features refit per fold (heldout_dgp_vs_uma). Both variants land at ~8.3 kJ once fairly held out.
dgp_ins = {r: v["dG_kJ"] for r, v in json.load(open(os.path.join(ROOT, "results", "eq", "dgpredictor_retrained_full.json"))).items() if v.get("dG_kJ") is not None}
dgp_ret_ho = {r: v["dgp_retrained_heldout"] for r, v in json.load(open(os.path.join(ROOT, "results", "eq", "dgp_retrained_heldout.json"))).items()}
dgp_ho_err = {r: v["dgp_heldout"] for r, v in json.load(open(os.path.join(ROOT, "experiments", "qm_mlip_solvation", "artifacts", "heldout_dgp_vs_uma.json"))).items()}
# eQuilibrator + Group Contribution: GENUINE runs (run_equilibrator.py / run_group_contribution.py,
# eqapi env), NOT the cached `thermodynamics.*` fields in the ModelSEED JSON. Both cached fields were
# stale/broken for this comparison (slope 0.20, corr 0.3-0.4 vs experiment; hydrolases at +97.9). The
# real component-contribution (eQ, MAE 3.6 corr 0.90) and pure group-contribution (GC, MAE 6.6 corr 0.75)
# sit on the diagonal as they should -- eQuilibrator/GCM are the family dGPredictor descends from. GC
# genuinely cannot score reactions whose compounds fall outside eQuilibrator's training set (no group
# decomposition) -> those are NaN and masked in the GC panel only (its box shows its own smaller n).
eq = {r: v["dG_kJ"] for r, v in json.load(open(os.path.join(HERE, "eq_real_tecrdb.json")))["predictions"].items()}
gc = {r: v["dG_kJ"] for r, v in json.load(open(os.path.join(HERE, "gc_real_tecrdb.json")))["predictions"].items()}

K = [k for k in uma_err if k in exp and k in dgp_orig and k in dgp_ho_err and k in eq and abs(eq[k]) < 1e6]
# retrained in-sample is missing on a few undecomposable reactions -> NaN there; masked per-panel so
# the other five panels keep the full n and their established MAEs (11.4 etc.) unchanged.
E = np.array([exp[k] for k in K])
# ROW 1 = IN-SAMPLE (fit to TECRDB); ROW 2 = FAIR held-out CV (incumbents refit per fold);
# ROW 3 = MetaG: no fitted parameters, but its routing policy was selected on TECRDB -> DEVELOPMENT, not held-out.
ROWS = [
    ("IN-SAMPLE  (fit to TECRDB)", [
        ("dGPredictor (retrained)", np.array([dgp_ins.get(k, np.nan) for k in K]), "#56B4E9"),
        ("dGPredictor (original)", np.array([dgp_orig[k] for k in K]), "#CC79A7"),
        ("eQuilibrator", np.array([eq[k] for k in K]), "#D55E00"),
        ("Group Contribution", np.array([gc.get(k, np.nan) for k in K]), "#0072B2")]),
    ("FAIR  (held-out CV)", [
        ("dGPredictor (retrained)", np.array([dgp_ret_ho.get(k, np.nan) for k in K]), "#56B4E9"),
        ("dGPredictor (original)", np.array([dgp_ho_err[k] + exp[k] for k in K]), "#CC79A7")]),
    ("DEVELOPMENT  (no fitted params)", [
        ("MetaG", np.array([uma_err[k] + exp[k] for k in K]), "#009E73")]),
]
ncol = max(len(r[1]) for r in ROWS)
allv = np.concatenate([E] + [P for _, ps in ROWS for _, P, _ in ps])
lim = [np.nanpercentile(allv, 1) - 20, np.nanpercentile(allv, 99) + 20]
bins = np.linspace(lim[0], lim[1], 30)

nrow = len(ROWS)
fig = plt.figure(figsize=(6.4 * ncol, 6.4 * nrow))
outer = fig.add_gridspec(nrow, ncol, wspace=0.32, hspace=0.32)
for ri, (row_label, panels) in enumerate(ROWS):
    for ci, (name, P, col) in enumerate(panels):
        inner = outer[ri, ci].subgridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.04, hspace=0.04)
        ax = fig.add_subplot(inner[1, 0]); axtop = fig.add_subplot(inner[0, 0], sharex=ax); axr = fig.add_subplot(inner[1, 1], sharey=ax)
        fin = np.isfinite(P)                          # a panel may miss a few reactions (NaN) -> mask them
        Ef, Pf = E[fin], P[fin]
        err = np.abs(Pf - Ef)
        ax.plot(lim, lim, "--", color="black", lw=1.1, zorder=1)
        ax.scatter(Ef, Pf, s=26, color=col, edgecolor="black", linewidth=0.25, alpha=0.75, zorder=3)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("experiment ΔrG′°  (kJ/mol)")
        if ci == 0: ax.set_ylabel("predicted ΔrG′°  (kJ/mol)")
        tag = f"MAE {err.mean():.1f}\nmed {np.median(err):.1f}" + (f"\nn {fin.sum()}" if fin.sum() < len(P) else "")
        ax.text(0.05, 0.95, tag, transform=ax.transAxes,
                va="top", ha="left", fontsize=15, bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        axtop.hist(Ef, bins=bins, color=col, edgecolor="white", linewidth=0.3)
        axr.hist(Pf, bins=bins, orientation="horizontal", color=col, edgecolor="white", linewidth=0.3)
        axtop.set_title(name, fontsize=15)
        for a in (axtop, axr): a.axis("off")
    # row label on the far left, centered on this row's gridspec cell (generalizes to any nrow)
    cell = outer[ri, 0].get_position(fig); yc = 0.5 * (cell.y0 + cell.y1)
    fig.text(0.015, yc, row_label, rotation=90, va="center", ha="center", fontsize=17, fontweight="bold")
fig.suptitle(f"TECRDB (native-condition reference) — predicted vs experiment (n={len(K)})", fontsize=19, y=0.98)
out = os.path.join(HERE, "..", "figures", "tecrdb_fiveway.png")
fig.savefig(out, bbox_inches="tight")
print("n=%d  MAE: " % len(K) + "  ".join(f"{n.split(chr(10))[0].split(' (')[0].split(' —')[0]} {np.nanmean(np.abs(P-E)):.1f}" for _, ps in ROWS for n, P, _ in ps))
print(f"wrote {os.path.abspath(out)}")
