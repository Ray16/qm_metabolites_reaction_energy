"""Within-EC-class correlation and the leave-one-out EC-mean baseline, on identical targets."""
import contextlib, io, json, statistics as st
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(open(__file__.replace('within_class.py', 'score.py')).read().split('# ---- secondary')[0], 's', 'exec'))
lab = {r: v['exp'] for r, v in ref['2.0']['labels'].items()}; M = pred['MetaG']
inp = json.load(open(PROJECT / 'analysis/reviewer_revision_20261022_01/reference_rebuild_current_01/prediction_structures.json'))
ec = lambda r: str(inp[r].get('EC', '')).split('.')[0]
def pear(a, b):
    ma, mb = st.mean(a), st.mean(b); n = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return n / (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** .5
def within(P, rs):
    cm = {e: st.mean(lab[q] for q in rs if ec(q) == e) for e in set(map(ec, rs))}
    pm = {e: st.mean(P[q] for q in rs if ec(q) == e) for e in set(map(ec, rs))}
    return round(pear([P[r] - pm[ec(r)] for r in rs], [lab[r] - cm[ec(r)] for r in rs]), 2)
res = {}
for name in ('CC_leave_reaction_out', 'CC_novel_substrates', 'CC_novel_all', 'dGbyG_OOF', 'MetaG'):
    P = pred[name]; rs = [r for r in lab if r in M and r in P]
    loo = st.mean(abs(st.mean([lab[q] for q in rs if q != r and ec(q) == ec(r)] or [0]) - lab[r]) for r in rs)
    res[name] = dict(n=len(rs), r_within_comparator=within(P, rs), r_within_MetaG=within(M, rs),
                     MAE_comparator=round(st.mean(abs(P[r] - lab[r]) for r in rs), 2),
                     MAE_MetaG=round(st.mean(abs(M[r] - lab[r]) for r in rs), 2), MAE_ECmean_LOO=round(loo, 2))
json.dump(res, open(HERE / 'within_class.json', 'w'), indent=1); print(json.dumps(res, indent=1))
