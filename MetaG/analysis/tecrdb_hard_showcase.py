"""The hardest TECRDB reactions (redox / glycosyl / glyoxalase / nucleotidyl) where group-contribution
fundamentally fails: UMA vs dGPredictor(retrained) vs EXPERIMENT. Ground-truthed (TECRDB Keq), current
pipeline. This is the flip side of the aggregate TECRDB comparison (where additivity wins on easy
reactions): on the reactions additivity CAN'T represent, UMA is far more accurate.
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
                     "xtick.labelsize": 13, "ytick.labelsize": 15, "legend.fontsize": 15,
                     "figure.dpi": 300, "savefig.dpi": 300})
HERE = os.path.dirname(os.path.abspath(__file__))

SUB = [("rxn00086", "redox"), ("rxn00070", "redox"), ("rxn00605", "glycosyl"),
       ("rxn01713", "glycosyl"), ("rxn01834", "glyoxalase"), ("rxn00579", "glycosyl"),
       ("rxn01675", "nucleotidyl"), ("rxn01005", "nucleotidyl")]
cur = json.load(open(os.path.join(HERE, "uma_tecrdb_current.json")))
tec = json.load(open(os.path.join(HERE, "..", "..", "experiments", "qm_mlip_solvation", "scripts", "reactions_tecrdb_all.json")))
dgp = json.load(open(os.path.join(HERE, "..", "..", "results", "eq", "dgpredictor_retrained_full.json")))

rids = [r for r, _ in SUB]
exp = np.array([tec[r]["exp"][0] for r in rids])
uma = np.array([exp[i] + cur[r] for i, r in enumerate(rids)])       # UMA dG = exp + err
dg = np.array([dgp[r]["dG_kJ"] for r in rids])
uma_mae = np.abs(uma - exp).mean(); dg_mae = np.abs(dg - exp).mean()

x = np.arange(len(rids)); w = 0.27
fig, ax = plt.subplots(figsize=(14, 7))
ax.bar(x - w, exp, w, color="#3b3b3b", label="TECRDB (experiment)")
ax.bar(x,     dg,  w, color="#c0392b", label=f"dGPredictor (retrained)   subset MAE {dg_mae:.0f}")
ax.bar(x + w, uma, w, color="#1a9e8f", label=f"UMA (MetaG)   subset MAE {uma_mae:.0f}")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"{r}\n{m}" for r, m in SUB])
ax.set_ylabel("ΔrG′°  (kJ/mol)")
ax.legend(loc="upper right", framealpha=0.96)
fig.tight_layout()
out = os.path.join(HERE, "..", "figures", "deck_top10_comparison.png")
fig.savefig(out, bbox_inches="tight")
print(f"n={len(rids)}  UMA MAE {uma_mae:.1f}  dGP MAE {dg_mae:.1f}")
for i, r in enumerate(rids):
    print(f"  {r} {SUB[i][1]:12s} exp {exp[i]:+6.1f}  UMA {uma[i]:+6.1f}  dGP {dg[i]:+7.1f}")
print(f"wrote {os.path.abspath(out)}")
