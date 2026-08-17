"""Generate honest figures for the unified-pipeline slide deck from the sweep logs.
Reuses ph0_final_analysis parsing. dpi 300, constrained_layout, no text overlap.
Outputs -> experiments/qm_mlip_solvation/figures/deck_*.png
"""
import os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib.util
spec = importlib.util.spec_from_file_location("pfa", os.path.join(HERE, "ph0_final_analysis.py"))
pfa = importlib.util.module_from_spec(spec); spec.loader.exec_module(pfa)

d = pfa.d
FIGDIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ---- collect per-reaction baseline + gated-coherent errors ---------------------------------
base, ph0 = {}, {}
for rid in d:
    be, _ = pfa._err(os.path.join(HERE, "..", "logs", "full367", f"{rid}.log"))
    pe, _ = pfa._err(os.path.join(HERE, "..", "logs", "ph0_sweep", f"{rid}.log"))
    if be is not None and abs(be) <= 200:
        base[rid] = be
    if pe is not None and abs(pe) <= 200:
        ph0[rid] = pe

coh = {}
for rid in base:
    iso = pfa.is_isomerization(d[rid]["species"])
    coh[rid] = base[rid] if (iso or rid not in ph0) else ph0[rid]

# HONEST comparison set: reactions where pH-0 has actually been computed (baseline AND pH-0
# both present). Including not-yet-processed reactions at their baseline value would dilute the
# improvement and misrepresent the pipeline. Matches tools/ph0_final_analysis.py exactly.
common = [r for r in base if r in ph0]
cls = {r: pfa.rxn_class(r) for r in common}

CLASSES = ["clean", "huge/floppy", "isomerase", "thioester", "anion"]
# brand-neutral, colorblind-safe pair
C_BASE, C_PIPE = "#9aa0a6", "#1a73e8"

def mae(xs): return float(np.mean([abs(x) for x in xs])) if xs else float("nan")

# ---- Figure 1: per-class MAE, baseline vs unified pipeline ---------------------------------
b_by = {c: mae([base[r] for r in common if cls[r] == c]) for c in CLASSES}
g_by = {c: mae([coh[r] for r in common if cls[r] == c]) for c in CLASSES}
n_by = {c: sum(1 for r in common if cls[r] == c) for c in CLASSES}
allb, allg = mae([base[r] for r in common]), mae([coh[r] for r in common])

fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
x = np.arange(len(CLASSES) + 1)
labels = [f"{c}\n(n={n_by[c]})" for c in CLASSES] + [f"ALL\n(n={len(common)})"]
bvals = [b_by[c] for c in CLASSES] + [allb]
gvals = [g_by[c] for c in CLASSES] + [allg]
w = 0.38
ax.bar(x - w/2, bvals, w, label="Baseline (implicit-anion QM)", color=C_BASE)
ax.bar(x + w/2, gvals, w, label="Unified pipeline (auto-routed)", color=C_PIPE)
for xi, (bv, gv) in enumerate(zip(bvals, gvals)):
    ax.text(xi - w/2, bv + 0.6, f"{bv:.0f}", ha="center", va="bottom", fontsize=9, color="#444")
    ax.text(xi + w/2, gv + 0.6, f"{gv:.0f}", ha="center", va="bottom", fontsize=9, color=C_PIPE, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("MAE vs TECRDB  (kJ/mol)", fontsize=12)
ax.set_title(f"Per-class accuracy: automatic routing halves the error  "
             f"({allb:.0f} → {allg:.0f} kJ/mol)", fontsize=13)
ax.legend(fontsize=10, framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, max(bvals) * 1.15)
fig.savefig(os.path.join(FIGDIR, "deck_per_class_mae.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ---- Figure 2: before/after |error| scatter (points below diagonal = improved) -------------
pts = [(abs(base[r]), abs(coh[r]), cls[r]) for r in common]
fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
cmap = {"clean": "#1a73e8", "huge/floppy": "#d93025", "isomerase": "#188038",
        "thioester": "#f9ab00", "anion": "#9334e6"}
lim = 120
ax.plot([0, lim], [0, lim], "--", color="#888", lw=1, zorder=1)
ax.fill_between([0, lim], [0, lim], lim, color="#1a73e8", alpha=0.05, zorder=0)
for c in CLASSES:
    xs = [b for b, g, cc in pts if cc == c]
    ys = [g for b, g, cc in pts if cc == c]
    ax.scatter(xs, ys, s=34, color=cmap[c], label=c, alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Baseline |error|  (kJ/mol)", fontsize=12)
ax.set_ylabel("Unified pipeline |error|  (kJ/mol)", fontsize=12)
ax.text(0.97*lim, 0.5, "below line =\npipeline improves", ha="right", va="bottom",
        fontsize=10, color="#1a73e8", style="italic")
ax.set_title("Hard reactions collapse toward the floor; isomerases unchanged (gated)", fontsize=11)
ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
ax.set_aspect("equal")
fig.savefig(os.path.join(FIGDIR, "deck_before_after_scatter.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"wrote 2 figures to {os.path.abspath(FIGDIR)}")
print(f"  n_common={len(common)}  baseline MAE={allb:.1f}  gated MAE={allg:.1f}")
for c in CLASSES:
    print(f"  {c:12s} n={n_by[c]:3d}  base={b_by[c]:5.1f}  pipe={g_by[c]:5.1f}")
