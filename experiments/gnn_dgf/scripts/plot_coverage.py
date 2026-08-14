#!/usr/bin/env python3
"""Two figures:
  (1) coverage_modelseed.png  — stacked horizontal bar, ModelSEED compound coverage
      per method on the common 45,662-active-compound universe.
  (2) gnn_vs_dgpretrained_hard10.png — GNN vs retrained-dGP on the hard reactions
      where retrained-dGP struggles (the ten from qm_vs_dgpredictor_top10).
"""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

# ---------- palette (colorblind-safe) ----------
GREEN="#2a9d5c"; ORANGE="#e08a1e"; GREY="#b8b8b8"; INK="#222222"
plt.rcParams.update({"font.size":11,"axes.edgecolor":"#888","svg.fonttype":"none"})

# ============ FIGURE 1: coverage ============
ACTIVE=45662; FLOOR=8861+6454   # no-structure + R-group (shared, ill-defined dGf)
COMPLETE=30335
# covered counts (of complete structures)
methods=[
 ("GNN (learned graph)",            30335),
 ("dGPredictor — retrained",         26461),
 ("eQuilibrator (structure-based)", 13284),
 ("dGPredictor — original (KEGG)",   5340),
]
# reference: ModelSEED's own legacy GCM (different gap semantics — shown faint)
labels=[m[0] for m in methods]; cov=np.array([m[1] for m in methods])
methodfail=COMPLETE-cov            # compounds with a valid complete structure the method still can't score
floor=np.full(len(methods),FLOOR)

fig,ax=plt.subplots(figsize=(11.5,5.8))
fig.subplots_adjust(left=0.25,right=0.99,top=0.83,bottom=0.28)
y=np.arange(len(methods))[::-1]
ax.barh(y,cov,color=GREEN,label="covered (finite ΔGf)")
ax.barh(y,methodfail,left=cov,color=ORANGE,label="valid structure, method can't score")
ax.barh(y,floor,left=cov+methodfail,color=GREY,label="no structure / R-group (floor, all methods)")
for yi,c in zip(y,cov):
    ax.text(c-350,yi,f"{c:,}",va="center",ha="right",color="white",fontweight="bold",fontsize=10)
    ax.text(c+300,yi, f"{100*c/COMPLETE:.0f}% of complete structures",va="center",ha="left",color=INK,fontsize=9.5)
ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=11)
ax.set_xlim(0,ACTIVE); ax.set_ylim(-0.6,4.05)
ax.set_xlabel("ModelSEED compounds  (of 45,662 active)")
ax.axvline(COMPLETE,color="#555",ls=":",lw=1)
ax.text(COMPLETE-350,3.75,"complete structures (30,335)",color="#555",fontsize=8.5,va="center",ha="right")
# title + plain-language subtitle stacked in the top margin (no overlap with bars)
fig.text(0.5,0.955,"ModelSEED formation-energy coverage by method",
         ha="center",fontsize=13,fontweight="bold")
fig.text(0.5,0.905,"How many ModelSEED compounds each method can assign a formation energy to "
         "(green) — GNN alone reaches every complete structure.",
         ha="center",fontsize=9.5,color="#555")
for s in ("top","right"): ax.spines[s].set_visible(False)
# legend BELOW the axes (bars fill the panel — no room inside)
h,l=ax.get_legend_handles_labels()
fig.legend(h,l,loc="lower center",bbox_to_anchor=(0.5,0.115),ncol=3,frameon=False,fontsize=9)
fig.text(0.012,0.055,"Fair denominator = 30,335 complete structures (parseable SMILES, no R-group).",
    fontsize=8,color="#666")
fig.text(0.012,0.018,"eQ 'can't score' = cache-miss 5,719 + infinite-uncertainty 11,261;  "
    "retrained-dGP gap = 3,874 post-snapshot compounds its frozen vocab can't decompose.",
    fontsize=8,color="#666")
fig.savefig(f"{FIG}/coverage_modelseed.png",dpi=300,bbox_inches="tight"); plt.close(fig)
print("wrote", f"{FIG}/coverage_modelseed.png")

# ============ FIGURE 2: all four methods vs experiment on the hard reactions ====
# The subset is defined by where the *naively ModelSEED-retrained* dGP overfits.
# eQ & original-dGP have these TECRDB reactions IN their training set (in-sample),
# so they nail them; the GNN number is HELD-OUT. Honest framing: the GNN is the
# robust/regularized version of "retrain dGP on ModelSEED" — it matches the
# curated incumbents held-out where the naive retrain blows up.
pred=json.load(open(os.path.join(HERE,"artifacts","predictions.json")))
RES="/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/results/eq"
eqf =json.load(open(f"{RES}/equilibrator_full.json"))
dgo =json.load(open(f"{RES}/dgpredictor_full.json"))
dgpr=json.load(open(f"{RES}/dgpredictor_retrained_full.json"))
order=[("rxn00086","redox"),("rxn00070","redox"),("rxn00605","glycosyl"),
       ("rxn01713","glycosyl"),("rxn01834","glyoxalase"),("rxn00579","glycosyl"),
       ("rxn01675","nucleotidyl"),("rxn01005","nucleotidyl")]
labs=[]; e=[]; eq=[]; do=[]; dr=[]; g=[]
for rid,cls in order:
    labs.append(f"{rid}\n{cls}")
    e.append(pred[rid]["exp"]); g.append(pred[rid]["gnn"])
    eq.append(eqf[rid]["dG_kJ"]); do.append(dgo[rid]["dG_kJ"]); dr.append(dgpr[rid]["dG_kJ"])
e=np.array(e); eq=np.array(eq); do=np.array(do); dr=np.array(dr); g=np.array(g)
mae=lambda p:np.mean(np.abs(p-e))
mae_eq,mae_do,mae_dr,mae_g=mae(eq),mae(do),mae(dr),mae(g)

# 5 bars per reaction: experiment, eQ(in-sample), dGP-orig(in-sample),
# dGP-retrained(extrapolates), GNN(held-out)
series=[("TECRDB (experiment)",           e,  "#444444", None),
        ("eQuilibrator · in-sample",      eq, "#7b3294", mae_eq),
        ("dGPredictor orig · in-sample",  do, "#2ca25f", mae_do),
        ("dGPredictor retrained · extrap.",dr,"#d1495b", mae_dr),
        ("GNN · held-out",                g,  "#2a7fb8", mae_g)]
x=np.arange(len(labs)); w=0.16; off=np.linspace(-2,2,5)*w
fig,ax=plt.subplots(figsize=(14,5.9))
fig.subplots_adjust(left=0.06,right=0.985,top=0.82,bottom=0.24)
for (name,vals,col,m),o in zip(series,off):
    lbl=name if m is None else f"{name}  (MAE {m:.0f})" if m>=10 else f"{name}  (MAE {m:.1f})"
    ax.bar(x+o,vals,w,color=col,label=lbl)
ax.axhline(0,color="#999",lw=.8)
ax.set_xticks(x); ax.set_xticklabels(labs,fontsize=10)
ax.set_ylabel("Δ$_r$G'° (kJ/mol)")
fig.text(0.5,0.955,"Naive ModelSEED-retrained dGP overfits these reactions; the GNN stays robust",
         ha="center",fontsize=13,fontweight="bold")
fig.text(0.5,0.905,"Each cluster is one reaction. Bars = the reaction energy Δ$_r$G'° each method predicts, "
         "next to the black experiment bar — the closer a bar matches black, the more accurate.",
         ha="center",fontsize=9.5,color="#555")
ax.legend(frameon=False,fontsize=9.5,loc="lower left",ncol=2)
for s in ("top","right"): ax.spines[s].set_visible(False)
fig.text(0.012,0.075,"Subset = reactions where the naively ModelSEED-retrained dGP breaks. eQ & original-dGP have these TECRDB reactions "
         "IN training (in-sample → 3.0/3.5); the GNN is HELD-OUT (6.2).",fontsize=8,color="#666")
fig.text(0.012,0.030,"Point: the GNN is the regularized version of that retrain — it matches curated incumbents held-out where the naive retrain "
         "flies off by 80–100 kJ. rxn32133/rxn34788 (n=1) excluded.",fontsize=8,color="#666")
fig.savefig(f"{FIG}/gnn_vs_dgpretrained_hard.png",dpi=300,bbox_inches="tight"); plt.close(fig)
print("wrote", f"{FIG}/gnn_vs_dgpretrained_hard.png")
print(f"hard-subset MAE (n={len(labs)}): eQ {mae_eq:.1f}  dGP-orig {mae_do:.1f}  "
      f"dGP-retrained {mae_dr:.1f}  GNN {mae_g:.1f}")
