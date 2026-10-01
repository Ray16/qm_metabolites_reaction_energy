"""Per-mechanism-class UMA diagnostic on TECRDB (current pipeline): signed bias vs MAE.

x = signed bias (mean err) -- a class ON the |bias|=MAE diagonal is SYSTEMATICALLY off (a solvation /
reference wall, anchor-fixable); a class near x=0 with MAE>0 is SCATTER (sampling/conformer, not a bias).
The dashed wedge is the identity MAE >= |bias| (always true). Bubble size ~ n reactions.
Uses merged current UMA errors (uma_tecrdb_current.json) + the per-reaction class labels.

Anchor status is checked PER REACTION via metag/routing/anchor.py's actual subclass()+ANCHORS
membership (not a coarse keyword match on the class name) -- a coarse class can mix anchored and
un-anchored reactions (e.g. "phosphatase" holds both the corrected phosphatase_monoester and the
un-corrected phosphatase_monoester_cationic; "CoA-thioester" holds both the corrected thioester_ppi/
thioester_pi anchors and many plain acyl-CoA reactions that never match either). Any coarse class
with a mixed anchor status is SPLIT into an "(anchored)"/"(not anchored)" pair of bubbles so the gold
ring only ever covers reactions that actually received the empirical offset.
"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "figure.dpi": 300, "savefig.dpi": 300})
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from metag.routing import anchor as ra
from metag import pipeline as _pl
ANCHORS_ON = _pl._flag("ANCHOR_CORRECT")      # the deployed pipeline applies no anchors since 2026-10-01

err = json.load(open(os.path.join(HERE, "uma_tecrdb_current.json")))
tec = json.load(open(os.path.join(HERE, "..", "..", "experiments", "qm_mlip_solvation", "scripts", "reactions_tecrdb_all.json")))
cls = {r["rid"]: r["class"] for r in json.load(open(os.path.join(HERE, "..", "metag", "data", "sigma_class_calibrated.json")))["per_reaction"]}

def reaction_subclass(rid):
    sp = tec.get(rid, {}).get("species")
    return ra.subclass(sp) if sp is not None else None

# for the two coarse classes that mix anchored + un-anchored reactions, split by the ACTUAL
# anchor.py subclass -- CoA-thioester further splits into its two distinct anchored mechanisms
# (thioester_ppi = AMP-forming, thioester_pi = ADP-forming) rather than merging them into one
# "(anchored)" bubble, so every gold circle corresponds to exactly one anchor.py subclass.
SUBCLASS_LABEL = {"thioester_ppi": "thioester (AMP-forming)", "thioester_pi": "thioester (ADP-forming)",
                  "phosphatase_monoester": "phosphatase (monoester)"}

by = {}
for rid, e in err.items():
    c = cls.get(rid)
    if not c:
        continue
    if ANCHORS_ON and c in ("phosphatase", "CoA-thioester"):
        sub = reaction_subclass(rid)
        key = SUBCLASS_LABEL.get(sub, c)   # un-anchored members keep the plain coarse-class name --
    else:                                  # the ring colour (gold vs. black) already says anchored or not
        key = c
    by.setdefault(key, []).append(e)

def display_name(key):
    return key

rows = []
for key, es in by.items():
    es = np.array(es)
    rows.append((display_name(key), es.mean(), np.abs(es).mean(), len(es)))   # class, bias, MAE, n

def short(c):
    return (c.replace("(P-N/Mg)", "").replace("(other)", "").replace("/PRT/nucleoside", "")
            .replace("amide/amidine-hydrolysis", "amide-hydrolysis").strip())

def is_anchored(c):
    """True iff this bubble's reactions actually received an anchor.py empirical offset (checked
    per-reaction above, not by keyword-matching the class name -- see module docstring)."""
    return ANCHORS_ON and (c in SUBCLASS_LABEL.values() or c == "phosphagen(P-N/Mg)")

GOLD = "#E69F00"                                  # anchor-corrected ring colour
YMAX = 25.0                                       # fixed y-max (user request)
LIM = YMAX                                        # symmetric wedge / x-range so the diagonal hits the corners

fig, ax = plt.subplots(figsize=(15.5, 10.5))
ax.fill_between([-LIM, 0, LIM], [LIM, 0, LIM], [LIM, LIM, LIM], color="#fdecec", zorder=0)   # MAE>=|bias| wedge
ax.plot([-LIM, 0, LIM], [LIM, 0, LIM], "--", color="#e07a7a", lw=1.4, zorder=1)
ax.axvline(0, ls="--", color="#7ab77a", lw=1.4, zorder=1)
for c, b, m, n in rows:
    ratio = abs(b) / m if m > 0 else 0
    col = "#c0392b" if ratio > 0.6 else ("#7f8c8d" if ratio > 0.35 else "#27ae60")
    anc = is_anchored(c)
    ax.scatter(b, m, s=60 + 34 * n, color=col,
               edgecolor=(GOLD if anc else "black"), linewidth=(3.4 if anc else 0.5),
               alpha=0.85, zorder=4 if anc else 3)

# Compact labels: outer points get a short local offset (tiny leader); only the dense central
# cluster (|bias|<3, MAE>9) is pulled to a short stacked column just left of centre.
def col_of(c):
    return GOLD if is_anchored(c) else "#1a1a1a"

def in_cluster(b, m):
    return abs(b) < 3 and m > 9

def in_low_cluster(b, m):
    return abs(b) < 3 and m < 4   # the tight anchored trio near the origin (MAE < 4)

BELOW = {"flavin/FAD-redox"}    # send below to clear the carbamoyltransfer label to its right
for c, b, m, n in rows:
    if in_cluster(b, m) or in_low_cluster(b, m):
        continue
    ha, dx = ("left", 11) if b >= 0 else ("right", -11)
    low = m < 9 or short(c) in BELOW
    dy, va = (-16, "top") if low else (13, "bottom")         # low points -> label into empty bottom
    ax.annotate(short(c), xy=(b, m), xytext=(dx, dy), textcoords="offset points", fontsize=18,
                ha=ha, va=va, color=col_of(c),
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.8, shrinkA=1, shrinkB=5))

# tight low-MAE anchored trio -> short stacked column in the empty lower-right area
low_cluster = sorted([r for r in rows if in_low_cluster(r[1], r[2])], key=lambda r: -r[2])
ys_low = np.linspace(4.6, 0.6, len(low_cluster))
for (c, b, m, n), yl in zip(low_cluster, ys_low):
    ax.annotate(short(c), xy=(b, m), xytext=(9.5, yl), textcoords="data", fontsize=18,
                ha="left", va="center", color=col_of(c),
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.8, shrinkA=1, shrinkB=5))

# dense central cluster -> short stacked column in the empty upper-right area
cluster = sorted([r for r in rows if in_cluster(r[1], r[2])], key=lambda r: -r[2])
ys = np.linspace(19.2, 12.8, len(cluster))
for (c, b, m, n), yl in zip(cluster, ys):
    ax.annotate(short(c), xy=(b, m), xytext=(4.3, yl), textcoords="data", fontsize=18,
                ha="left", va="center", color=col_of(c),
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.8, shrinkA=1, shrinkB=5))

# legend proxies (drawn off-canvas), moved to a horizontal band ABOVE the axes
ax.scatter([], [], s=170, color="#27ae60", edgecolor="black", lw=0.5, label="scatter → sampling-limited")
ax.scatter([], [], s=170, color="#c0392b", edgecolor="black", lw=0.5, label="|bias|≈MAE → solvation/reference wall")
if ANCHORS_ON:
    ax.scatter([], [], s=170, facecolor="#dddddd", edgecolor=GOLD, lw=3.4, label="anchor-corrected")
for c, b, m, n in rows:                            # classes beyond the fixed axes: pin at the edge, label value
    if m > YMAX or abs(b) > LIM:
        xb = max(-LIM, min(LIM, b)) * 0.96
        ax.scatter([xb], [YMAX * 0.96], marker="^", s=260, color="#c0392b", edgecolor="black", lw=0.6, zorder=4)
        ax.annotate(f"{short(c)} (bias {b:+.0f}, MAE {m:.0f}, n={n})", (xb, YMAX * 0.96), xytext=(-12, -30),
                    textcoords="offset points", ha="right", fontsize=18)
ax.set_xlim(-LIM, LIM); ax.set_ylim(0, YMAX)
ax.set_xlabel("signed bias per class  (kJ/mol)")
ax.set_ylabel("MAE per class  (kJ/mol)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
          framealpha=0.95, fontsize=18, handletextpad=0.4, columnspacing=1.1)
fig.tight_layout()
out = os.path.join(HERE, "..", "figures", "diag_signed_bias_by_class.png")
fig.savefig(out, bbox_inches="tight")
print(f"classes: {len(rows)}")
for c, b, m, n in sorted(rows, key=lambda r: -r[2]):
    print(f"  {c:26s} bias {b:+6.1f}  MAE {m:5.1f}  n={n}")
print(f"wrote {os.path.abspath(out)}")
