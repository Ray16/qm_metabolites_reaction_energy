"""Score MetaG and baselines on the Mg-bounded reference (identical intersections)."""
import csv, json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ref = json.loads((HERE / 'reference_mg_bounded.json').read_text())
panel = json.loads((PROJECT / 'analysis/official_default_20261024_01/current_panel_metrics.json').read_text())
metag = {r: o['dG_kj'] for r, o in panel['all_requested_outcomes'].items() if o['status'] == 'estimated'}

pred = {'MetaG': metag}
for row in csv.DictReader(open(PROJECT / 'artifacts/benchmark/comparison/all_predictions.tsv'), delimiter='\t'):
    if row['status'] == 'ok' and row['method'] in ('eQuilibrator_CC', 'eQuilibrator_GC_only', 'dGPredictor_published', 'dGbyG_eQ_bridge'):
        pred.setdefault(row['method'], {})[row['reaction']] = float(row['dG_kJ'])
oof = {}
for row in csv.DictReader(open(PROJECT / 'artifacts/benchmark/paper_reproduction/metag_vs_dgbyg_pairs.tsv'), delimiter='\t'):
    oof[row['reaction']] = float(row['dgbyg'])   # archived 10-fold out-of-fold mean, I=0 bridged
pred['dGbyG_OOF'] = oof
nov_path = HERE / 'cc_novelty.json'
if nov_path.exists():
    nov = json.loads(nov_path.read_text())['predictions']
    for regime in ('published', 'leave_reaction_out', 'novel_substrates', 'novel_all'):
        pred['CC_' + regime] = {r: v[regime]['dG'] for r, v in nov.items()
                                if v.get('status') == 'ok' and v[regime]['dG'] is not None}


def metrics(pairs):
    e = [abs(p - y) for p, y in pairs]
    s = [p - y for p, y in pairs]
    return dict(n=len(e), MAE=round(st.mean(e), 2), median=round(st.median(e), 2),
                RMSE=round((st.mean(x * x for x in e)) ** .5, 2), bias=round(st.mean(s), 2),
                frac_gt20=round(sum(x > 20 for x in e) / len(e), 3)) if e else dict(n=0)


out = {}
for thr, block in ref.items():
    lab = {r: v['exp'] for r, v in block['labels'].items()}
    res = {'n_labels': len(lab)}
    for name, p in pred.items():
        res[name] = metrics([(p[r], y) for r, y in lab.items() if r in p])
    res['zero_on_MetaG'] = metrics([(0.0, y) for r, y in lab.items() if r in metag])
    common = [r for r in lab if all(r in pred[m] for m in ('MetaG', 'eQuilibrator_CC', 'dGbyG_eQ_bridge', 'dGPredictor_published'))]
    res['common_main'] = {m: metrics([(pred[m][r], lab[r]) for r in common]) for m in ('MetaG', 'eQuilibrator_CC', 'dGbyG_eQ_bridge', 'dGPredictor_published')}
    res['common_main']['zero'] = metrics([(0.0, lab[r]) for r in common])
    o = [r for r in lab if r in oof and r in metag]
    res['common_OOF'] = {'MetaG': metrics([(metag[r], lab[r]) for r in o]), 'dGbyG_OOF': metrics([(oof[r], lab[r]) for r in o]),
                         'zero': metrics([(0.0, lab[r]) for r in o])}
    if 'CC_novel_all' in pred:
        nv = [r for r in lab if r in metag and all(r in pred['CC_' + k] for k in ('published', 'leave_reaction_out', 'novel_substrates'))]
        res['common_CC_regimes'] = {m: metrics([(pred[m][r], lab[r]) for r in nv]) for m in
                                    ('MetaG', 'CC_published', 'CC_leave_reaction_out', 'CC_novel_substrates')}
        res['common_CC_regimes']['zero'] = metrics([(0.0, lab[r]) for r in nv])
        na = [r for r in lab if r in metag and r in pred['CC_novel_all']]
        res['common_CC_novel_all'] = {'MetaG': metrics([(metag[r], lab[r]) for r in na]),
                                      'CC_novel_all': metrics([(pred['CC_novel_all'][r], lab[r]) for r in na]),
                                      'zero': metrics([(0.0, lab[r]) for r in na])}
        for k in ('leave_reaction_out', 'novel_substrates', 'novel_all'):
            labd = [r for r in lab if r in metag and r in nov and nov[r].get('status') == 'ok']
            res['CC_' + k + '_coverage'] = f"{sum(r in pred['CC_' + k] for r in labd)}/{len(labd)}"
    out[thr] = res
(HERE / 'scores.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out['2.0'], indent=1))
for t in ('1.0', '4.0'):
    print(t, out[t]['n_labels'], out[t]['MetaG'], out[t]['zero_on_MetaG'])

# ---- secondary descriptors on the primary (2.0 kJ/mol) reference ----
lab = {r: v['exp'] for r, v in ref['2.0']['labels'].items()}
tiers = {d['id']: d['tier'] for d in json.loads((PROJECT / 'analysis/modelseed_scope/scope_per_reaction.json').read_text())}
extra = {}
def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return round(num / (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** .5, 3)
for m in [k for k in pred if pred[k]]:
    rs = [r for r in lab if r in pred[m]]
    if len(rs) < 3:
        continue
    big = [r for r in rs if abs(lab[r]) > 10]
    extra[m] = dict(pearson=pearson([pred[m][r] for r in rs], [lab[r] for r in rs]),
                    n_big=len(big), sign_acc_big=round(sum((pred[m][r] > 0) == (lab[r] > 0) for r in big) / len(big), 3) if big else None,
                    by_tier={t: metrics([(pred[m][r], lab[r]) for r in rs if tiers.get(r) == t]) for t in ('T1', 'T3', 'T4')})
out['descriptors_2.0'] = extra
(HERE / 'scores.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps({m: {k: v for k, v in e.items()} for m, e in extra.items()}, indent=1))
