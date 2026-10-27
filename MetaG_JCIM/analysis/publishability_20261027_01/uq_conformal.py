"""Split-conformal prediction intervals for MetaG (point estimates unchanged).

Calibration residuals: MetaG vs the Mg-bounded labels. Interval = prediction +/- q, q the
ceil((n+1)(1-alpha))-th smallest calibration |residual|. Coverage is checked leave-one-out on the
benchmark (exchangeable case) and on the external phosphorolysis set (distribution shift).
Variants: global; Mondrian by top-level EC class (classes with < 20 calibration reactions pool).
"""
import contextlib, io, json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile((HERE / 'score.py').read_text().split('# ---- secondary')[0], 'score.py', 'exec'))
lab = {r: v['exp'] for r, v in ref['2.0']['labels'].items()}
M = pred['MetaG']
inp = json.loads((PROJECT / 'analysis/reviewer_revision_20261022_01/reference_rebuild_current_01/prediction_structures.json').read_text())
ec = lambda r: str(inp[r].get('EC', '')).split('.')[0]
res = {r: abs(M[r] - lab[r]) for r in lab if r in M}
ext = json.loads((PROJECT / 'analysis/official_default_20261024_01/prospective_native_evaluation.json').read_text())['all24_outcomes']
ext_err = [abs(o['nominal_native_error_kj']) for o in ext.values() if o.get('evaluation_status') == 'conditional_native_recipe_scored']
EXT_EC = '2'   # nucleoside phosphorylases, EC 2.4.2.x
MIN_CLASS = 20


def q(values, alpha):
    v = sorted(values); k = math.ceil((len(v) + 1) * (1 - alpha))
    return v[min(k, len(v)) - 1]


def calib(rs, cls):
    same = [res[r] for r in rs if ec(r) == cls]
    return same if len(same) >= MIN_CLASS else [res[r] for r in rs]


out = {'n_calibration': len(res), 'n_external': len(ext_err), 'min_class_size': MIN_CLASS}
for alpha in (0.2, 0.1, 0.05):
    row = {}
    # global
    loo = [res[r] <= q([res[s] for s in res if s != r], alpha) for r in res]
    qg = q(list(res.values()), alpha)
    row['global'] = dict(halfwidth=round(qg, 1), loo_coverage=round(sum(loo) / len(loo), 3),
                         external_coverage=round(sum(e <= qg for e in ext_err) / len(ext_err), 3))
    # Mondrian EC
    loo, widths = [], []
    for r in res:
        qq = q(calib([s for s in res if s != r], ec(r)), alpha); loo.append(res[r] <= qq); widths.append(qq)
    qe = q(calib(list(res), EXT_EC), alpha)
    row['ec_mondrian'] = dict(mean_halfwidth=round(sum(widths) / len(widths), 1), loo_coverage=round(sum(loo) / len(loo), 3),
                              by_class={c: round(q(calib(list(res), c), alpha), 1) for c in sorted(set(map(ec, res)))},
                              external_halfwidth=round(qe, 1),
                              external_coverage=round(sum(e <= qe for e in ext_err) / len(ext_err), 3))
    out[f'{int(100 * (1 - alpha))}%'] = row
(HERE / 'uq_conformal.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out, indent=1))
