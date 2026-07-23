#!/usr/bin/env python
"""Regenerate the benchmark analysis plots with CORRECT units (kJ/mol) and the
REAL experimental reference (TECRDB), after the kcal->kJ fix.

Unit handling (raw ModelSEED files untouched):
  - QM (qm_dG_transformed_kJ) is already kJ/mol.
  - eQuilibrator/ModelSEED column in reaction_benchmark.csv is kcal/mol -> *4.184.
  - TECRDB experimental dG is kJ/mol (-R*T*ln K', R=8.31e-3 kJ/(K*mol)).

Figure A  qm_vs_experiment.png : QM & eQuilibrator vs TECRDB experiment, per
          reaction (only reactions with real measurements). The honest accuracy
          check. eQ is in-sample (trained on TECRDB); QM is out-of-sample.
Figure B  qm_error_by_charge_kJ.png : QM deviation from the eQuilibrator
          PREDICTION, stratified by max |charge|. Uses the prediction (not
          experiment) because only it covers all charge levels; labeled as such.
"""
from __future__ import annotations

import csv, glob, json, math, os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

KCAL_TO_KJ = 4.184
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THERMO = os.path.join(ROOT, "thermodynamic_calc")
MSEED = os.path.join(ROOT, "ModelSEEDDatabase", "Biochemistry")
PROTON = "cpd00067"


def fnum(x):
    try:
        v = float(x); return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def load_benchmark():
    """rxn_id -> (qm_kJ, eq_kJ)  [eq converted kcal->kJ]."""
    out = {}
    for r in csv.DictReader(open(os.path.join(THERMO, "results/benchmark/reaction_benchmark.csv"))):
        qm = fnum(r["qm_dG_transformed_kJ"])
        eq = fnum(r["eq_dG_kJ"])
        out[r["rxn_id"]] = (qm, None if eq is None else eq * KCAL_TO_KJ)
    return out


def load_dgpredictor():
    """rxn_id -> dGPredictor dG_mean. Verified kJ/mol and same (ModelSEED)
    direction as eQ: median dGP/eQ_kJ ratio = 1.00 over 87 shared reactions.
    JSON shape: {rxn_id: {KEGG_R: {dG_mean, dG_uncer}}} or a string ('no KEGG ID
    found'). Multi-mapped rxns are dropped only when the values disagree."""
    td = os.path.join(MSEED, "Thermodynamics", "dGPredictor", "json_files")
    out = {}
    for f in glob.glob(os.path.join(td, "reaction_*_dG.json")):
        for rid, inner in json.load(open(f)).items():
            if not isinstance(inner, dict):
                continue
            vals = [v["dG_mean"] for v in inner.values()
                    if isinstance(v, dict) and v.get("dG_mean") is not None]
            if not vals:
                continue
            if len(vals) > 1 and max(vals) - min(vals) > 0.5:
                continue
            out[rid] = vals[0]
    return out


def load_gcm():
    """rxn_id -> GCM (Group Contribution, Jankowski 2008) reaction dG in kJ/mol.

    Compound dGf is col7 of ModelSEED/KEGG_{Charged,Original}_MolAnalysis.tbl,
    in KCAL/MOL -- verified against known values (H2O col7 -56.7 == -56.7 kcal/mol
    ΔGf; CO2 -92.3; Pi -262.0). Reaction dG = sum(coeff * dGf) over non-proton
    species (Charged form preferred), then *4.184 -> kJ/mol. Sentinel 1e7 = 'no
    group energy' -> reaction unscored if any species lacks one. NOTE: GCM is
    purely additive, so it returns ~0 for group-conserving reactions
    (transaminations, isomerizations)."""
    SENT = 1e7
    cpd = {}
    for proc in ("Charged", "Original"):  # Charged preferred
        fn = os.path.join(MSEED, "Thermodynamics", "ModelSEED",
                          f"KEGG_{proc}_MolAnalysis.tbl")
        for line in open(fn):
            a = line.rstrip("\n").split("\t")
            if len(a) < 9:
                continue
            try:
                dg = float(a[7])
            except ValueError:
                continue
            if abs(dg) >= SENT:
                continue
            cpd.setdefault(a[0], dg)
    c2k = {}
    for line in open(os.path.join(MSEED, "Aliases",
                                  "Unique_ModelSEED_Compound_Aliases.txt")):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[2] == "KEGG" and p[1].startswith("C") and p[1][1:].isdigit():
            c2k.setdefault(p[0], set()).add(p[1])
    c2k = {k: next(iter(v)) for k, v in c2k.items() if len(v) == 1}

    bench = set(load_benchmark())
    out = {}
    for path in glob.glob(os.path.join(MSEED, "reaction_*.json")):
        for rec in json.load(open(path)):
            if rec["id"] not in bench:
                continue
            s, ok = 0.0, True
            for r in rec["stoichiometry"]:
                if r["compound"] == "cpd00067":
                    continue
                k = c2k.get(r["compound"])
                if k is None or k not in cpd:
                    ok = False
                    break
                s += cpd[k] * r["coefficient"]
            if ok:
                out[rec["id"]] = s * KCAL_TO_KJ
    return out


def load_xtb():
    """rxn_id -> GFN2-xTB/ALPB(water) transformed Delta_rG'^o (kJ/mol).

    Semi-empirical baseline: GFN2-xTB single-shot Gibbs (--ohess) in ALPB water on
    the DFT-optimised lowest conformer, run through the same Alberty/Debye-Huckel
    transform as the QM bar. Shows what dropping DFT for a pure semi-empirical
    method costs. Missing file -> empty (bar simply omitted)."""
    p = os.path.join(THERMO, "results/benchmark/xtb_reaction_dG.json")
    if not os.path.exists(p):
        return {}
    return {k: float(v) for k, v in json.load(open(p)).items()}


def load_aimnet2():
    """rxn_id -> AIMNet2-composite transformed Delta_rG'^o (kJ/mol).

    Composite prototype (aimnet2_workflow/run_composite.py): AIMNet2 gas-phase
    electronic energy (wB97M quality) + xtb-ALPB solvation + xtb thermal G_RRHO,
    on the xtb-optimised geometry. Missing file -> empty."""
    p = os.path.join(THERMO, "results/benchmark/aimnet2_reaction_dG.json")
    if not os.path.exists(p):
        return {}
    return {k: float(v) for k, v in json.load(open(p)).items()}


def load_uma():
    """rxn_id -> UMA-composite transformed Delta_rG'^o (kJ/mol).

    Composite prototype (uma_workflow/run_uma_composite.py): Meta UMA / OMol25
    gas-phase electronic energy (uma-s-1p2) + xtb-ALPB solvation + xtb thermal
    G_RRHO, on the xtb-optimised geometry -- identical solvation/thermal to the
    AIMNet2-composite bar, so the difference isolates the electronic energy.
    Missing file -> empty."""
    p = os.path.join(THERMO, "results/benchmark/uma_reaction_dG.json")
    if not os.path.exists(p):
        return {}
    return {k: float(v) for k, v in json.load(open(p)).items()}


def load_experiment():
    """rxn_id -> (exp_kJ, sd, name, is_nearstd)."""
    out = {}
    p = os.path.join(THERMO, "results/benchmark/experimental_dG_TECRDB.csv")
    for r in csv.DictReader(open(p)):
        near = fnum(r["dG_exp_nearstd_kJ"])
        if near is not None:
            exp, sd, isn = near, (fnum(r["dG_exp_nearstd_sd_kJ"]) or 0.0), True
        else:
            lo, hi = fnum(r["dG_all_min_kJ"]), fnum(r["dG_all_max_kJ"])
            exp, sd, isn = (lo + hi) / 2, abs(hi - lo) / 2, False
        out[r["rxn_id"]] = (exp, sd, r["reaction"], isn)
    return out


def load_charges():
    """rxn_id -> max |charge| of any species (via ModelSEED stoich + metabolite charges)."""
    charge = {m["id"]: m["charge"]
              for m in json.load(open(os.path.join(THERMO, "central_metabolites_in_opentecr.json")))}
    bench = set(load_benchmark())
    out = {}
    for path in glob.glob(os.path.join(MSEED, "reaction_*.json")):
        for rec in json.load(open(path)):
            if rec["id"] not in bench:
                continue
            cs = {s["compound"] for s in rec["stoichiometry"] if s["compound"] != PROTON}
            chs = [charge.get(c) for c in cs]
            if chs and all(c is not None for c in chs):
                out[rec["id"]] = max(abs(c) for c in chs)
    return out


def figure_A(bench, exp, dgp, gcm, xtb, aim, uma):
    # distinct reactions (dedup by reaction name), with experimental data
    seen, rows = set(), []
    for rid, (e, sd, name, isn) in exp.items():
        if name in seen:
            continue
        qm, eq = bench.get(rid, (None, None))
        if qm is None or eq is None:
            continue
        seen.add(name)
        rows.append((name, e, sd, qm, eq, dgp.get(rid), gcm.get(rid), isn,
                     xtb.get(rid), aim.get(rid), uma.get(rid)))
    rows.sort(key=lambda r: r[1])

    def _fmt(n):  # drop state symbols, wrap at the equilibrium arrow for full names
        n = n.replace("(aq)", "").replace("(l)", "").replace("  ", " ").strip()
        return n.replace(" = ", "\n⇌ ")  # U+21CC equilibrium arrow
    labels = [_fmt(r[0]) for r in rows]
    e = np.array([r[1] for r in rows]); sd = np.array([r[2] for r in rows])
    qm = np.array([r[3] for r in rows]); eq = np.array([r[4] for r in rows])
    dg = np.array([np.nan if r[5] is None else r[5] for r in rows])
    gc = np.array([np.nan if r[6] is None else r[6] for r in rows])
    xt = np.array([np.nan if r[8] is None else r[8] for r in rows])
    am = np.array([np.nan if r[9] is None else r[9] for r in rows])
    um = np.array([np.nan if r[10] is None else r[10] for r in rows])
    x = np.arange(len(rows)); w = 0.10

    fig, ax = plt.subplots(figsize=(max(10, 2.8 * len(rows)), 5.8))
    ax.bar(x - 3.5 * w, e, w, yerr=sd, capsize=3, color="0.25", label="TECRDB experiment")
    ax.bar(x - 2.5 * w, eq, w, color="#4C9F70", label="eQuilibrator")
    ax.bar(x - 1.5 * w, dg, w, color="#3B7DB3", label="dGPredictor")
    ax.bar(x - 0.5 * w, gc, w, color="#E8A33D", label="GCM (Jankowski)")
    ax.bar(x + 0.5 * w, qm, w, color="#C1666B", label="QM (DFT)")
    ax.bar(x + 1.5 * w, xt, w, color="#8E6FB3", label="GFN2-xTB")
    ax.bar(x + 2.5 * w, am, w, color="#5AA9A3", label="AIMNet2-composite")
    ax.bar(x + 3.5 * w, um, w, color="#D08A3E", label="UMA-composite")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=7.5)
    ax.set_ylabel(r"$\Delta_r G'^{\circ}$ (kJ/mol)")
    ax.set_title(f"QM vs three empirical methods vs real experiment (TECRDB)  "
                 f"(n={len(rows)} reactions)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "qm_vs_experiment.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    mae = lambda p: np.nanmean(np.abs(p - e))
    print(f"wrote {out}  (n={len(rows)})  MAE: eQ {mae(eq):.1f}  dGP {mae(dg):.1f}  "
          f"GCM {mae(gc):.1f}  QM {mae(qm):.1f}  xTB {mae(xt):.1f}  "
          f"AIMNet2 {mae(am):.1f}  UMA {mae(um):.1f} kJ/mol")
    for r in rows:
        name, ee, s, q, eqv, dgv, gcv, isn, xtv, amv, umv = r
        tag = "" if isn else "  [*off-std]"
        dgs = " n/a" if dgv is None else f"{dgv:6.1f}"
        gcs = " n/a" if gcv is None else f"{gcv:6.1f}"
        xts = " n/a" if xtv is None else f"{xtv:7.1f}"
        ams = " n/a" if amv is None else f"{amv:7.1f}"
        ums = " n/a" if umv is None else f"{umv:7.1f}"
        print(f"   {name[:30]:30s} exp={ee:6.1f} eQ={eqv:6.1f} dGP={dgs} GCM={gcs} "
              f"QM={q:7.1f} xTB={xts} AIMNet2={ams} UMA={ums}{tag}")


def figure_B(bench, charges):
    # QM minus eQuilibrator PREDICTION, by max|charge|, deduped on (qm,eq) pair
    seen, pts = set(), []
    for rid, (qm, eq) in bench.items():
        q = charges.get(rid)
        if qm is None or eq is None or q is None:
            continue
        key = (round(qm, 2), round(eq, 2), q)
        if key in seen:
            continue
        seen.add(key)
        pts.append((q, abs(qm - eq)))
    levels = sorted({p[0] for p in pts})
    by = {q: [d for qq, d in pts if qq == q] for q in levels}

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    cmap = plt.cm.viridis
    colors = {q: cmap(i / max(1, len(levels) - 1)) for i, q in enumerate(levels)}
    rng = np.random.default_rng(0)
    box = ax.boxplot([by[q] for q in levels], positions=range(len(levels)),
                     widths=0.55, showfliers=False, patch_artist=True,
                     medianprops=dict(color="black", lw=1.5))
    for patch, q in zip(box["boxes"], levels):
        patch.set_facecolor(colors[q]); patch.set_alpha(0.35)
    ymax = max(max(v) for v in by.values())
    for i, q in enumerate(levels):
        ys = np.array(by[q]); xs = i + rng.uniform(-0.16, 0.16, len(ys))
        ax.scatter(xs, ys, s=42, color=colors[q], edgecolor="white", lw=0.6, zorder=3)
        ax.text(i, ymax * 1.04, f"MAD {ys.mean():.1f}\nn={len(ys)}", ha="center", fontsize=9)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([("neutral\n(0)" if q == 0 else f"±{q}") for q in levels])
    ax.set_xlabel("Max |charge| of any species in the reaction")
    ax.set_ylabel(r"$|\Delta_r G_{\mathrm{QM}} - \Delta_r G_{\mathrm{eQuilibrator}}|$  (kJ/mol)")
    ax.set_title("QM vs eQuilibrator PREDICTION grows with ionic character (kJ/mol)\n"
                 "reference is the empirical prediction, not experiment (broad coverage)",
                 fontsize=10.5)
    ax.set_ylim(-2, ymax * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(ROOT, "figures", "qm_error_by_charge_kJ.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"\nwrote {out}  ({len(pts)} distinct reactions)")
    for q in levels:
        ys = np.array(by[q])
        print(f"   maxQ={q}: n={len(ys):2d}  MAD(QM-eQ)={ys.mean():5.1f} kJ/mol")


def main():
    # Only the experiment-referenced figure is a valid benchmark; figure_B
    # (vs the eQuilibrator prediction) is retired and intentionally not drawn.
    bench = load_benchmark()
    exp = load_experiment()
    dgp = load_dgpredictor()
    gcm = load_gcm()
    xtb = load_xtb()
    aim = load_aimnet2()
    uma = load_uma()
    print(f"benchmark reactions: {len(bench)}; experiment: {len(exp)}; "
          f"dGPredictor: {len(dgp)}; GCM-scorable: {len(gcm)}; xTB: {len(xtb)}; "
          f"AIMNet2: {len(aim)}; UMA: {len(uma)}\n")
    figure_A(bench, exp, dgp, gcm, xtb, aim, uma)


if __name__ == "__main__":
    main()
