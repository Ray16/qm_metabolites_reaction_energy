"""Build slide-specific PNGs from pinned local evidence; run from any directory.

All text is >= 18 pt (project figure convention). Inputs are read-only snapshots;
no model is rerun.
"""
from pathlib import Path
import csv
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent/'figures'
OUT.mkdir(exist_ok=True)
TEAL='#087F8C';INK='#173640';GRAY='#7C8D94';AMBER='#CF8A22';PURPLE='#9467A6';BLUE='#0072B2';RUST='#B16936'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':18,'axes.labelsize':18,
 'axes.titlesize':20,'xtick.labelsize':18,'ytick.labelsize':18,'legend.fontsize':18,'savefig.dpi':300,
 'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK})
def save(fig,name):
 fig.savefig(OUT/name,dpi=300,bbox_inches='tight',pad_inches=.14,facecolor='white');plt.close(fig)
def tsv(path):
 with open(path) as f:return list(csv.DictReader(f,delimiter='\t'))

# --- Reference tiers -------------------------------------------------------------
report=json.loads((ROOT/'artifacts/benchmark/curated/reference_report.json').read_text())
rows=[r for r in report['measurements'] if r['selection']=='used']
sets=[set(json.loads((ROOT/'src/metag/data/reactions_opentecr_std.json').read_text())),
 {r['reaction'] for r in rows},{r['reaction'] for r in rows if abs(r['temperature_K']-298.15)<1e-6},
 {r['reaction'] for r in rows if abs(r['temperature_K']-298.15)<1e-6 and not r['ionic_strength_imputed']}]
assert [len(s) for s in sets]==[364,54,18,8]
fig,ax=plt.subplots(figsize=(13,5.2))
labels=['Reaction structures','Retained reference label','Row at 298.15 K','Row at 298.15 K + reported I']
for i,(values,color) in enumerate(zip(sets,[GRAY,TEAL,TEAL,AMBER])):
 ax.barh(i,len(values),color=color,height=.58)
 ax.text(len(values)+5,i,str(len(values)),va='center',fontweight='bold')
ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_xlim(0,405)
ax.set_xlabel('Number of reactions');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
ax.grid(axis='x',alpha=.13);ax.set_axisbelow(True);save(fig,'reference_coverage.png')

# --- Frozen release 2026-10-01c --------------------------------------------------
records=[json.loads(p.read_text()) for p in sorted((ROOT/'artifacts/results/metag_opentecr_calibrated').glob('*.json'))]
x=np.array([r['exp'] for r in records]);y=np.array([r['dG'] for r in records])
legacy=json.loads((ROOT/'artifacts/benchmark/curated/legacy_reactions_opentecr_std.json').read_text())
assert all(abs(r['exp']-statistics.median(legacy[r['reaction']]['exp']))<1e-8 for r in records)
fig,ax=plt.subplots(figsize=(7.5,6.5));lims=(min(x.min(),y.min())-5,max(x.max(),y.max())+5)
ax.plot(lims,lims,'--',c=GRAY,lw=1.2);ax.scatter(x,y,s=30,color=TEAL,alpha=.64,edgecolors='none')
ax.set(xlim=lims,ylim=lims,xlabel='Archived reference (kJ/mol)',ylabel='MetaG (kJ/mol)')
ax.text(.04,.96,f'n = {len(x)}\nMAE {np.abs(y-x).mean():.2f}\nMedian {np.median(np.abs(y-x)):.2f}',
 transform=ax.transAxes,va='top')
save(fig,'historical_accuracy.png')

# --- ModelSEED preflight ---------------------------------------------------------
pre=json.loads((ROOT/'analysis/code_review/modelseed_preflight.json').read_text())
counts=[pre['status_counts'][k] for k in ['review','no_known_gap_detected','blocked']]
fig,ax=plt.subplots(figsize=(12,3.0));start=0
for v,c in zip(counts,[AMBER,TEAL,INK]):
 ax.barh(0,v,left=start,color=c,height=.5)
 if v>10:ax.text(start+v/2,0,str(v),ha='center',va='center',color='white',fontweight='bold',fontsize=22)
 start+=v
ax.set(xlim=(0,300),ylim=(-.5,.5),xlabel='300-reaction ModelSEED panel',yticks=[])
ax.spines['left'].set_visible(False)
ax.legend(handles=[plt.Rectangle((0,0),1,1,color=c) for c in [AMBER,TEAL,INK]],
 labels=[f'{v}: {l}' for v,l in zip(counts,['Require review','No known gap','Blocked'])],
 loc='upper center',bbox_to_anchor=(.5,-.55),ncol=3,frameon=False)
save(fig,'modelseed_preflight.png')

# --- CoA core crop from the verified truncation figure ---------------------------
im=Image.open(ROOT/'manuscript/figures/truncation_scheme.png')
im.crop((0,0,im.width,round(im.height*338/900))).save(OUT/'coa_core_example.png',dpi=(600,600))

# --- Matched MetaG (development) vs held-out dGbyG -------------------------------
pairs=tsv(ROOT/'artifacts/benchmark/paper_reproduction/metag_vs_dgbyg_pairs.tsv')
ref=np.array([float(p['reference']) for p in pairs])
pred={'MetaG (development)':np.array([float(p['metag']) for p in pairs]),
      'dGbyG (out-of-fold)':np.array([float(p['dgbyg']) for p in pairs])}
assert len(ref)==164
fig,axes=plt.subplots(1,3,figsize=(17,6),gridspec_kw={'wspace':.42})
lim=(-65,62)
for ax,(lab,p),c in zip(axes[:2],pred.items(),[TEAL,PURPLE]):
 ax.plot(lim,lim,'--',color=GRAY,lw=1.2);ax.scatter(ref,p,s=30,color=c,alpha=.7,edgecolors='none')
 e=np.abs(p-ref);ax.text(.05,.95,f'MAE {e.mean():.2f}\nMedian {np.median(e):.2f}',transform=ax.transAxes,va='top')
 ax.set(xlim=lim,ylim=lim,xlabel='Paper reference (kJ/mol)',ylabel='Prediction (kJ/mol)');ax.set_title(lab,pad=12)
m=[[np.abs(p-ref).mean(),np.median(np.abs(p-ref)),np.sqrt(((p-ref)**2).mean())] for p in pred.values()]
for j,(row,c,lab) in enumerate(zip(m,[TEAL,PURPLE],pred)):
 xs=np.arange(3)+(j-.5)*.38;axes[2].bar(xs,row,.36,color=c,label=lab.split(' ')[0])
 for xx,v in zip(xs,row):axes[2].text(xx,v+.25,f'{v:.1f}',ha='center',va='bottom')
axes[2].set_xticks(range(3),['MAE','Median','RMSE']);axes[2].set_ylabel('Error (kJ/mol)');axes[2].set_ylim(0,15)
axes[2].legend(frameon=False,loc='upper left');axes[2].set_title('Same 164 targets',pad=12)
save(fig,'matched_dgbyg.png')

# --- Gas-phase stereo check ------------------------------------------------------
sq=ROOT/'analysis/state_qc_20261007'
raw=json.loads((sq/'stereo_enthalpy.json').read_text());v=json.loads((sq/'stereo_reference_verified.json').read_text())['electronic']
fig,axes=plt.subplots(1,2,figsize=(14,6),gridspec_kw={'wspace':.4})
settings=[('2-Butene: H(298 K)',[-3.1,raw['pairs'][0]['uma_enthalpy_E_minus_Z_kj']],[1.64,0],['Experiment','UMA + RRHO'],(-8,2)),
 ('1,2-Difluoroethylene: E$_{el}$',[v['reference_kj'],v['uma_kj'],v['mmff_kj']],[v['reference_uncertainty_kj'],0,0],['Exp.-derived','UMA','MMFF94'],(-10,8))]
for ax,(title,ys,err,labs,yl) in zip(axes,settings):
 ax.axhline(0,color=GRAY,lw=.8)
 for i,(yv,e) in enumerate(zip(ys,err)):
  ax.errorbar(i,yv,yerr=e,fmt='o',ms=11,capsize=6,color=['#333333',TEAL,RUST][i])
  ax.annotate(f'{yv:+.2f}',(i,yv),xytext=(16,0),textcoords='offset points',va='center')
 ax.set_xticks(range(len(ys)),labs);ax.set_xlim(-.5,len(ys)-.3);ax.set_ylim(*yl)
 ax.set_title(title,pad=12);ax.set_ylabel('E − Z (kJ/mol)');ax.grid(axis='y',alpha=.16)
save(fig,'stereo_check.png')
print('Generated 6 slide PNGs.')
