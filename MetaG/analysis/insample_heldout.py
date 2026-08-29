"""In-sample vs held-out MAE on TECRDB, four methods. The point: does the incumbents' accuracy survive
a fair (held-out) evaluation, and how does first-principles UMA compare?

Numbers (kJ/mol, MAE vs experiment, this repo):
  dGPredictor  in-sample 5.7 (median 2.0 = the "famous ~3")  ->  held-out CV 6.5   (well-regularized;
               leakage shows mostly in the median 2.0->4.5, not the MAE)
  eQuilibrator in-sample 8.0                                  ->  held-out: needs CC CV (not in repo)
  Group Contr. in-sample 9.3                                  ->  held-out: needs GC CV (not in repo)
  UMA (MetaG)  n/a (never fit to TECRDB)                      ->  held-out 11.6 (inherently out-of-sample)

HONEST READ: on TECRDB (the additivity family's TRAINING domain) the incumbents win even HELD-OUT
(dGP 6.5 < UMA 11.6). UMA's advantage is NOT here -- it is on the ModelSEED frontier (reference_scoreboard),
where the incumbents extrapolate and fail. dGP held-out is a fair proxy for the GC/eQ family held-out.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
                     "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 15,
                     "figure.dpi": 300, "savefig.dpi": 300})
HERE = os.path.dirname(os.path.abspath(__file__))

# (method, in-sample MAE or None, held-out MAE or None, reason-if-held-out-missing)
M = [("dGPredictor\n(original)", 3.0, None, "fixed\nweights"),     # famous ~3; published, can't hold out
     ("dGPredictor\n(retrained)", 5.7, 6.5, None),                 # our CV: the honest leakage-corrected pair
     ("eQuilibrator", 8.0, None, "CV\nnot run"),
     ("Group\nContribution", 9.3, None, "CV\nnot run"),
     ("UMA (MetaG)", None, 11.6, None)]

x = np.arange(len(M)); w = 0.38
fig, ax = plt.subplots(figsize=(13, 6.6))
for i, (name, ins, ho, miss) in enumerate(M):
    if ins is not None:
        ax.bar(i - w/2, ins, w, color="#9ecae1", edgecolor="black", linewidth=0.6,
               label="in-sample (fit to TECRDB)" if i == 0 else None)
        ax.text(i - w/2, ins + 0.3, f"{ins:.1f}", ha="center", va="bottom", fontsize=15)
    else:
        ax.text(i - w/2, 0.5, "never\nfit", ha="center", va="bottom", fontsize=13, color="0.5", style="italic")
    if ho is not None:
        c = "#009E73" if name.startswith("UMA") else "#08519c"
        ax.bar(i + w/2, ho, w, color=c, edgecolor="black", linewidth=0.6,
               label="held-out (fair)" if i == 1 else None)
        ax.text(i + w/2, ho + 0.3, f"{ho:.1f}", ha="center", va="bottom", fontsize=15)
    elif miss:
        ax.text(i + w/2, 0.5, miss, ha="center", va="bottom", fontsize=13, color="0.5", style="italic")
ax.set_xticks(x); ax.set_xticklabels([m[0] for m in M])
ax.set_ylabel("MAE vs experiment  (kJ/mol)")
ax.set_ylim(0, 13.5)
ax.set_title("TECRDB accuracy: in-sample vs held-out  (lower is better)", fontsize=17)
ax.legend(loc="upper left", framealpha=0.96)
# UMA colour patch note in legend
from matplotlib.patches import Patch
h, l = ax.get_legend_handles_labels()
h.append(Patch(facecolor="#009E73", edgecolor="black")); l.append("UMA held-out (first-principles)")
ax.legend(h, l, loc="upper left", framealpha=0.96)
fig.tight_layout()
out = os.path.join(HERE, "..", "figures", "insample_vs_heldout.png")
fig.savefig(out, bbox_inches="tight")
print(f"wrote {os.path.abspath(out)}")
