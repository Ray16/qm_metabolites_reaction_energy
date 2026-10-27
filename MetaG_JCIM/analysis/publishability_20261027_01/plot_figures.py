"""Main-text figures for the Mg-bounded benchmark and the novelty tests (18-pt project style)."""
import json, os, statistics as st, sys
from pathlib import Path
sys.path.insert(0, '/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/_lib')
import _style  # noqa: F401  (18-pt rcParams, dpi 300)
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT = PROJECT / 'manuscript/figures'
BLUE, ORANGE, GREEN, GRAY, PURPLE = '#0072B2', '#D55E00', '#009E73', '#777777', '#CC79A7'

exec(compile((HERE / 'score.py').read_text().split('# ---- secondary')[0], 'score.py', 'exec'))  # pred, ref, metrics
lab = {r: v['exp'] for r, v in ref['2.0']['labels'].items()}
tiers = {d['id']: d['tier'] for d in json.loads((PROJECT / 'analysis/modelseed_scope/scope_per_reaction.json').read_text())}
M = pred['MetaG']

# ---- Figure: benchmark parity + error ECDF ----
fig, (a, b) = plt.subplots(1, 2, figsize=(16, 7.5))
rs = sorted(r for r in lab if r in M)
x, y = np.array([lab[r] for r in rs]), np.array([M[r] for r in rs])
lim = (min(x.min(), y.min()) - 5, max(x.max(), y.max()) + 5)
a.plot(lim, lim, '--', color=GRAY, lw=1.2, zorder=0)
a.fill_between(lim, [l - 10 for l in lim], [l + 10 for l in lim], color=GRAY, alpha=.12, lw=0, zorder=0)
a.scatter(x, y, s=36, color=BLUE, alpha=.75, edgecolor='none')
a.set_xlim(lim); a.set_ylim(lim)
a.set_xlabel(r"Reference $\Delta_rG'^\circ$ (kJ mol$^{-1}$)"); a.set_ylabel(r"MetaG $\Delta_rG'^\circ$ (kJ mol$^{-1}$)")
a.set_title(f'A   n = {len(rs)}', loc='left')
for vals, label, color, ls in [(np.abs(y - x), 'MetaG', BLUE, '-'), (np.abs(x), 'Zero predictor', GRAY, '--')]:
    v = np.sort(vals)
    b.step(v, np.arange(1, len(v) + 1) / len(v), where='post', color=color, ls=ls, lw=2.2,
           label=f'{label} (MAE {v.mean():.1f})')
b.set_xlim(0, 40); b.set_ylim(0, 1)
b.set_xlabel(r'Absolute error (kJ mol$^{-1}$)'); b.set_ylabel('Cumulative fraction')
b.set_title('B', loc='left'); b.legend(loc='lower right', frameon=False)
fig.tight_layout(); fig.savefig(OUT / 'benchmark_mg_bounded.pdf'); fig.savefig(OUT / 'benchmark_mg_bounded.png'); plt.close(fig)

# ---- Figure: accuracy vs novelty ----
fig, (a, b) = plt.subplots(1, 2, figsize=(18, 7.5), gridspec_kw={'width_ratios': [1, 1.5]})
D = pred['dGbyG_eQ_bridge']
groups = [('Seen\nreaction', {'T1'}), ('Unseen\ncompound', {'T3', 'T4'})]
w = .38
for i, (name, ts) in enumerate(groups):
    rr = [r for r in lab if r in M and r in D and tiers.get(r) in ts]
    for j, (src, color, lbl) in enumerate([(M, BLUE, 'MetaG'), (D, ORANGE, 'dGbyG (published)')]):
        mae = st.mean(abs(src[r] - lab[r]) for r in rr)
        a.bar(i + (j - .5) * w, mae, w, color=color, label=lbl if i == 0 else None)
        a.text(i + (j - .5) * w, mae + .2, f'{mae:.1f}', ha='center', va='bottom')
    a.text(i, -1.9, f'n = {len(rr)}', ha='center', va='top')
a.set_xticks(range(len(groups))); a.set_xticklabels([g[0] for g in groups])
a.set_ylabel(r'MAE (kJ mol$^{-1}$)'); a.set_title('A   novelty relative to dGbyG training', loc='left')
a.set_ylim(0, 12); a.legend(loc='upper left', frameon=False)
if 'CC_novel_all' in pred:
    regs = [('CC_published', 'In-sample'), ('CC_leave_reaction_out', 'Reaction\nheld out'),
            ('CC_novel_substrates', 'Substrates\nheld out'), ('CC_novel_all', 'All compounds\nheld out')]
    for i, (k, name) in enumerate(regs):
        cand = [r for r in lab if r in M and r in nov and nov[r].get('status') == 'ok']
        common3 = [r for r in cand if all(r in pred[q] for q, _ in regs[:3])]
        rr = common3 if k != 'CC_novel_all' else [r for r in cand if r in pred[k]]
        mc = st.mean(abs(pred[k][r] - lab[r]) for r in rr)
        mm = st.mean(abs(M[r] - lab[r]) for r in rr)
        b.bar(i - w / 2, mm, w, color=BLUE, label='MetaG (no training)' if i == 0 else None)
        b.bar(i + w / 2, mc, w, color=GREEN, label='Component contribution' if i == 0 else None)
        b.text(i - w / 2, mm + .3, f'{mm:.1f}', ha='center', va='bottom')
        b.text(i + w / 2, mc + .3, f'{mc:.1f}', ha='center', va='bottom')
        b.text(i, -2.4, f'n = {len(rr)}', ha='center', va='top')
    b.set_xticks(range(len(regs))); b.set_xticklabels([n for _, n in regs])
    b.set_ylabel(r'MAE (kJ mol$^{-1}$)'); b.set_title('B   CC retrained without the test compounds', loc='left')
    top = max(p.get_height() for p in b.patches)
    b.set_ylim(0, top * 1.25); b.legend(loc='upper left', frameon=False)
fig.tight_layout(); fig.savefig(OUT / 'novelty_tests.pdf'); fig.savefig(OUT / 'novelty_tests.png'); plt.close(fig)
print('figures written')
