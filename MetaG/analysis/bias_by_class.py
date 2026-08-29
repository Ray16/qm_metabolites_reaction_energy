"""Per-mechanism-class UMA diagnostic on TECRDB (current pipeline): signed bias vs MAE.

x = signed bias (mean err) -- a class ON the |bias|=MAE diagonal is SYSTEMATICALLY off (a solvation /
reference wall, anchor-fixable); a class near x=0 with MAE>0 is SCATTER (sampling/conformer, not a bias).
The dashed wedge is the identity MAE >= |bias| (always true). Bubble size ~ n reactions.
Uses merged current UMA errors (uma_tecrdb_current.json) + the per-reaction class labels.
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "figure.dpi": 300, "savefig.dpi": 300})
HERE = os.path.dirname(os.path.abspath(__file__))

err = json.load(open(os.path.join(HERE, "uma_tecrdb_current.json")))
cls = {r["rid"]: r["class"] for r in json.load(open(os.path.join(HERE, "..", "metag", "data", "sigma_class_calibrated.json")))["per_reaction"]}
by = {}
for rid, e in err.items():
    c = cls.get(rid)
    if c:
        by.setdefault(c, []).append(e)
rows = []
for c, es in by.items():
    es = np.array(es)
    rows.append((c, es.mean(), np.abs(es).mean(), len(es)))   # class, bias, MAE, n

def short(c):
    return (c.replace("(P-N/Mg)", "").replace("(other)", "").replace("/PRT/nucleoside", "")
            .replace("amide/amidine-hydrolysis", "amide-hydrolysis").strip())

fig, ax = plt.subplots(figsize=(14, 9))
lim = max(max(abs(b) for _, b, _, _ in rows), max(m for _, _, m, _ in rows)) + 6
ax.fill_between([-lim, 0, lim], [lim, 0, lim], [lim, lim, lim], color="#fdecec", zorder=0)   # MAE>=|bias| wedge
ax.plot([-lim, 0, lim], [lim, 0, lim], "--", color="#e07a7a", lw=1.4, zorder=1)
ax.axvline(0, ls="--", color="#7ab77a", lw=1.4, zorder=1)
for c, b, m, n in rows:
    ratio = abs(b) / m if m > 0 else 0
    col = "#c0392b" if ratio > 0.6 else ("#7f8c8d" if ratio > 0.35 else "#27ae60")
    ax.scatter(b, m, s=60 + 34 * n, color=col, edgecolor="black", linewidth=0.5, alpha=0.8, zorder=3)
# labels: pixel offsets + leader lines; direction by x-sign, vertical stagger to de-collide the centre
seen_y = []
for c, b, m, n in sorted(rows, key=lambda r: (r[0])):
    ha = "right" if b <= 0 else "left"
    ox = -14 if b <= 0 else 14
    oy = 12
    for sy in seen_y:                                    # nudge up if too close to an existing label
        if abs((m + oy / 3.0) - sy) < 1.3:
            oy += 16
    seen_y.append(m + oy / 3.0)
    ax.annotate(short(c), (b, m), xytext=(ox, oy), textcoords="offset points", fontsize=12,
                ha=ha, va="bottom", arrowprops=dict(arrowstyle="-", color="0.55", lw=0.6))
ax.scatter([], [], s=120, color="#27ae60", label="scatter (near x=0) → sampling-limited")
ax.scatter([], [], s=120, color="#c0392b", label="on diagonal (|bias|≈MAE) → solvation/reference wall")
ax.set_xlim(-lim, lim); ax.set_ylim(0, lim + 4)
ax.set_xlabel("UMA signed bias per class  (kJ/mol)")
ax.set_ylabel("UMA MAE per class  (kJ/mol)")
ax.legend(loc="upper center", framealpha=0.95, fontsize=13)
fig.tight_layout()
out = os.path.join(HERE, "..", "figures", "diag_signed_bias_by_class.png")
fig.savefig(out, bbox_inches="tight")
print(f"classes: {len(rows)}")
for c, b, m, n in sorted(rows, key=lambda r: -r[2]):
    print(f"  {c:26s} bias {b:+6.1f}  MAE {m:5.1f}  n={n}")
print(f"wrote {os.path.abspath(out)}")
