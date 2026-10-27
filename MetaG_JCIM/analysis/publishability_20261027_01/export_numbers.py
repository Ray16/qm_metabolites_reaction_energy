"""Write manuscript/bench_numbers.tex from the reference, CC-novelty and scoring outputs."""
import json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
exec(compile((HERE / 'score.py').read_text(), 'score.py', 'exec'))   # defines ref, pred, nov, metrics, out, tiers, lab
S = out['2.0']
M = pred['MetaG']
macros = {}
def put(name, value, fmt='{:.1f}'):
    macros[name] = fmt.format(value) if isinstance(value, float) else str(value)

put('benchLabels', S['n_labels']); put('benchMeasurements', f"{ref['2.0']['n_measurements']:,}")
put('benchLabelsOne', out['1.0']['n_labels']); put('benchLabelsFour', out['4.0']['n_labels'])
put('benchOldLabels', 39)
_ex = ref['2.0']['excluded']
put('mgSensitiveExcluded', f"{_ex.get('Mg_sensitive_beyond_threshold', 0):,}"); put('mgAmbiguousExcluded', _ex.get('ambiguous_reaction_mapping', 0))
put('mgOtherExcluded', sum(_ex.values()) - _ex.get('Mg_sensitive_beyond_threshold', 0) - _ex.get('ambiguous_reaction_mapping', 0))
m = S['MetaG']
put('benchScored', m['n']); put('benchMAE', m['MAE']); put('benchMedian', m['median']); put('benchRMSE', m['RMSE'])
put('benchBias', m['bias']); put('benchZeroMAE', S['zero_on_MetaG']['MAE'])
put('benchMAEOne', out['1.0']['MetaG']['MAE']); put('benchMAEFour', out['4.0']['MetaG']['MAE'])
d = out['descriptors_2.0']['MetaG']
put('benchPearson', d['pearson'], '{:.2f}'); put('benchNBig', d['n_big']); put('benchSignAccPct', 100 * d['sign_acc_big'], '{:.0f}')
put('benchFracGtTwenty', 100 * m['frac_gt20'], '{:.0f}')
c = S['common_main']
put('commonN', c['MetaG']['n'])
for k, n in [('MetaG', 'commonMetaG'), ('eQuilibrator_CC', 'commonCC'), ('dGbyG_eQ_bridge', 'commonDgbyg'),
             ('dGPredictor_published', 'commonDgp'), ('zero', 'commonZero')]:
    put(n, c[k]['MAE'])
put('commonGCN', S['eQuilibrator_GC_only']['n']); put('commonGC', S['eQuilibrator_GC_only']['MAE'])
o = S['common_OOF']
put('oofN', o['MetaG']['n']); put('oofMetaG', o['MetaG']['MAE']); put('oofDgbyg', o['dGbyG_OOF']['MAE']); put('oofZero', o['zero']['MAE'])
put('oofMetaGMedian', o['MetaG']['median']); put('oofDgbygMedian', o['dGbyG_OOF']['median'])
D = pred['dGbyG_eQ_bridge']
for name, ts in [('Seen', {'T1'}), ('Novel', {'T3', 'T4'})]:
    rr = [r for r in lab if r in M and r in D and tiers.get(r) in ts]
    put(f'tier{name}N', len(rr))
    put(f'tier{name}MetaG', st.mean(abs(M[r] - lab[r]) for r in rr))
    put(f'tier{name}Dgbyg', st.mean(abs(D[r] - lab[r]) for r in rr))
    put(f'tier{name}Zero', st.mean(abs(lab[r]) for r in rr))
    put(f'tier{name}Gap', st.mean(abs(M[r] - lab[r]) for r in rr) - st.mean(abs(D[r] - lab[r]) for r in rr))
    put(f'tier{name}FarN', sum(tiers.get(r) == 'T4' for r in rr))
if 'common_CC_regimes' in S:
    cr = S['common_CC_regimes']
    put('ccSetN', cr['MetaG']['n']); put('ccSetMetaG', cr['MetaG']['MAE']); put('ccSetZero', cr['zero']['MAE'])
    put('ccPubMAE', cr['CC_published']['MAE']); put('ccLROMAE', cr['CC_leave_reaction_out']['MAE'])
    put('ccNSMAE', cr['CC_novel_substrates']['MAE'])
    put('ccPubMedian', cr['CC_published']['median']); put('ccNSMedian', cr['CC_novel_substrates']['median'])
    put('ccSetMetaGMedian', cr['MetaG']['median'])
    put('ccNSGtTwenty', 100 * cr['CC_novel_substrates']['frac_gt20'], '{:.0f}')
    put('ccSetMetaGGtTwenty', 100 * cr['MetaG']['frac_gt20'], '{:.0f}')
    na = S['common_CC_novel_all']
    put('ccNAN', na['MetaG']['n']); put('ccNAMAE', na['CC_novel_all']['MAE']); put('ccNAMetaG', na['MetaG']['MAE'])
    put('ccNAZero', na['zero']['MAE'])
    for k, n in [('leave_reaction_out', 'LRO'), ('novel_substrates', 'NS'), ('novel_all', 'NA')]:
        a, b = S[f'CC_{k}_coverage'].split('/')
        put(f'ccCov{n}', a); put('ccCovDen', b); put(f'ccCov{n}Pct', 100 * int(a) / int(b), '{:.0f}')
wc = json.loads((HERE / 'within_class.json').read_text())
for key, tag in [('MetaG', 'All'), ('CC_leave_reaction_out', 'LRO'), ('CC_novel_substrates', 'NS'), ('CC_novel_all', 'NA'), ('dGbyG_OOF', 'OOF')]:
    w = wc[key]
    put(f'wc{tag}N', w['n']); put(f'wc{tag}MetaGr', w['r_within_MetaG'], '{:.2f}'); put(f'wc{tag}Compr', w['r_within_comparator'], '{:.2f}')
    put(f'wc{tag}ECmean', w['MAE_ECmean_LOO']); put(f'wc{tag}MetaGMAE', w['MAE_MetaG']); put(f'wc{tag}CompMAE', w['MAE_comparator'])
uq = json.loads((HERE / 'uq_conformal.json').read_text())
for lvl, tag in [('80%', 'Eighty'), ('90%', 'Ninety'), ('95%', 'NinetyFive')]:
    g = uq[lvl]['global']
    put(f'uq{tag}Half', g['halfwidth']); put(f'uq{tag}LOO', 100 * g['loo_coverage'], '{:.0f}')
    put(f'uq{tag}Ext', 100 * g['external_coverage'], '{:.0f}')
    put(f'uq{tag}ExtHits', round(g['external_coverage'] * uq['n_external']))
    put(f'uq{tag}ECMean', uq[lvl]['ec_mondrian']['mean_halfwidth'])
put('uqNExt', uq['n_external'])
# per-EC (MetaG, primary reference)
inputs = json.loads((PROJECT / 'analysis/reviewer_revision_20261022_01/reference_rebuild_current_01/prediction_structures.json').read_text())
ec_rows = []
for ec in '123456':
    rr = [r for r in lab if r in M and str(inputs[r].get('EC', '')).split('.')[0] == ec]
    if rr:
        ec_rows.append((ec, len(rr), st.mean(abs(M[r] - lab[r]) for r in rr), st.mean(abs(lab[r]) for r in rr),
                        st.mean(M[r] - lab[r] for r in rr)))
names = {'1': 'Oxidoreductases', '2': 'Transferases', '3': 'Hydrolases', '4': 'Lyases', '5': 'Isomerases', '6': 'Ligases'}
lines = ['% generated by analysis/publishability_20261027_01/export_numbers.py -- do not edit by hand']
lines += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(macros.items())]
(PROJECT / 'manuscript/bench_numbers.tex').write_text('\n'.join(lines) + '\n')
tab = ['% generated by export_numbers.py', '\\begin{tabular}{lrrrr}', '\\toprule EC class & $n$ & MetaG MAE & Zero MAE & MetaG bias\\\\', '\\midrule']
tab += [f'{ec} {names[ec]} & {n} & {a:.1f} & {z:.1f} & {b:+.1f}\\\\' for ec, n, a, z, b in ec_rows]
tab += ['\\bottomrule', '\\end{tabular}']
(PROJECT / 'manuscript/bench_ec_table.tex').write_text('\n'.join(tab) + '\n')
json.dump(macros, open(HERE / 'bench_numbers.json', 'w'), indent=1)
print(json.dumps(macros, indent=1)); print('\n'.join(tab))
