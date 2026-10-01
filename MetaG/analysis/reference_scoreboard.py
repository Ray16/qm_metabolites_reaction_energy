"""The ONLY honest accuracy plot: error of each method against a REAL reference (experiment or DFT),
on the reactions where such a reference exists. No incumbent-as-truth. References are scarce -- that is
itself the finding (ModelSEED has no experimental ΔG) -- so this set is small and will grow as DFT lands.

Reference types:
  exp   = small-molecule experimental ΔG of the reaction TYPE (gas-phase); UMA-only (GC/eQ have no value
          for these non-metabolic anchors) -> shows UMA's ABSOLUTE accuracy on the chemistry.
  cycle = aqueous ΔrG'° from a measured parent-reaction thermodynamic cycle (6 ligases); full 3-way.
  DFT   = PBE0/def2-TZVP gas-phase electronic (neutral reaction -> solvation/thermal small & cancelling,
          so a fair proxy); full 3-way. (Caveat: PBE0 not exact for extended conjugation.)
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
                     "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18,
                     "figure.dpi": 300, "savefig.dpi": 300})
HERE = os.path.dirname(os.path.abspath(__file__))

# (label, ref_type, reference, UMA, GC, eQ)  -- GC/eQ = None where not applicable (exp anchors)
DATA = [
    ("2H2 + O2 -> 2H2O",              "exp",   -457.2, -450.2, None, None),
    ("benzene + O2 -> phenol",         "exp",   -162.6, -159.0, None, None),
    ("hydroquinone + O2 -> quinone",   "exp",   -102.0, -104.5, None, None),
    ("epoxide + H2O -> diol",          "exp",    -81.5,  -55.8, None, None),
    ("rxn00226 acyl-adenylate",        "cycle",  +25.0,  +19.9,  -9.6, +281.5),
    ("rxn00418 aminoacyl-adenylate",   "cycle",  +25.0,  +28.2, -11.1, +278.5),
    ("rxn00054 phenoxazinone (dimer.)","DFT",   -892.8, -981.3,-286.3,-236.8),
]

rows = [(lbl, rt, abs(u-r), (abs(gc-r) if gc is not None else None), (abs(eq-r) if eq is not None else None))
        for lbl, rt, r, u, gc, eq in DATA]

# ---- console summary ----
print(f"{'reaction':32s} {'ref':6s} {'|UMA-ref|':>9s} {'|GC-ref|':>9s} {'|eQ-ref|':>9s}")
for (lbl, rt, r, u, gc, eq) in DATA:
    g = f"{abs(gc-r):9.1f}" if gc is not None else f"{'--':>9s}"
    e = f"{abs(eq-r):9.1f}" if eq is not None else f"{'--':>9s}"
    print(f"{lbl:32s} {rt:6s} {abs(u-r):9.1f} {g} {e}")
three = [x for x in rows if x[3] is not None]
print(f"\n3-way subset (n={len(three)}): median |err|  UMA={np.median([x[2] for x in three]):.0f}  "
      f"GC={np.median([x[3] for x in three]):.0f}  eQ={np.median([x[4] for x in three]):.0f} kJ/mol")

# ---- figure: horizontal dot plot, log-x (errors span 2..660 kJ) ----
fig, ax = plt.subplots(figsize=(11, 0.62 * len(rows) + 2))
y = np.arange(len(rows))[::-1]
for yi, (lbl, rt, eu, eg, ee) in zip(y, rows):
    if eg is not None:
        ax.plot([eu, max(eg, ee)], [yi, yi], "-", color="0.85", lw=2, zorder=1)
ax.scatter([x[2] for x in rows], y, s=170, marker="D", color="#009E73", edgecolor="black",
           linewidth=0.7, label="MetaG", zorder=4)
ax.scatter([x[3] for x in rows if x[3] is not None], [yi for yi, x in zip(y, rows) if x[3] is not None],
           s=130, color="#0072B2", edgecolor="black", linewidth=0.4, label="Group Contribution", zorder=3)
ax.scatter([x[4] for x in rows if x[4] is not None], [yi for yi, x in zip(y, rows) if x[4] is not None],
           s=130, color="#D55E00", edgecolor="black", linewidth=0.4, label="eQuilibrator", zorder=3)
ax.set_yticks(y); ax.set_yticklabels([f"{lbl}  [{rt}]" for lbl, rt, *_ in rows])
ax.set_xscale("log")
ax.set_xlabel("|error vs real reference|  (kJ/mol)   — lower is better")
ax.axvline(10, color="0.6", ls=":", lw=1.2); ax.text(10, len(rows)-0.4, " 10 kJ", color="0.25", fontsize=18)
ax.set_xlim(1, 1200)
ax.legend(loc="center right", framealpha=0.96)     # upper/center-right is empty (top rows are UMA-only, left side)
ax.margins(y=0.06)
fig.tight_layout()
out = os.path.join(HERE, "..", "figures", "reference_scoreboard.png")
fig.savefig(out, bbox_inches="tight")
print(f"\nwrote {os.path.abspath(out)}")
