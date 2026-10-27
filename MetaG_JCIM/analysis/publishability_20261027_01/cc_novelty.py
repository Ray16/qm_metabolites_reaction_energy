"""Component contribution retrained with the test reaction's compounds held out.

Uses the published eQuilibrator CC training matrices (cc_params.npz: train_S, train_G,
train_b, train_w) and the published CC algebra (Noor 2013; component_contribution 0.6
Trainer.train_from_matrices), solved with economy SVDs. Regimes:
  published            : the released parameters (in-sample for TECRDB reactions)
  leave_reaction_out   : drop training columns with the same compound support as the test reaction
  novel_substrates     : drop every training column containing a non-currency compound of the reaction
  novel_all            : drop every training column containing any compound of the reaction (except water)
A prediction outside the RC+GC span (infinite CC uncertainty) is reported as unavailable.
No MetaG value or experimental label is read here.
"""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
FROZEN = PROJECT / 'analysis/reviewer_revision_20261022_01/reference_rebuild_current_01'
CURRENCY = ['cpd00001', 'cpd00002', 'cpd00008', 'cpd00018', 'cpd00003', 'cpd00004', 'cpd00005', 'cpd00006',
            'cpd00009', 'cpd00012', 'cpd00011', 'cpd00013', 'cpd00010', 'cpd00007', 'cpd00242']
TOL = 1e-10
_G = {}


def solve(S, G, b, w):
    """dG0_cc, dG0_gc and the GC left-null projector, as in Trainer.train_from_matrices."""
    def inv_project(A):
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        r = int((s > TOL).sum())
        inv = Vh[:r].T @ np.diag(1 / s[:r]) @ U[:, :r].T
        P_R = U[:, :r] @ U[:, :r].T
        return inv, P_R, np.eye(A.shape[0]) - P_R
    inv_S, P_R_rc, P_N_rc = inv_project(S * w)
    GS = G.T @ S
    inv_GS, _, P_N_gc = inv_project(GS * w)
    dG0_gc = inv_GS.T @ (w * b)
    dG0_rc = inv_S.T @ (w * b)
    dG0_cc = P_R_rc @ dG0_rc + P_N_rc @ G @ dG0_gc
    return dG0_cc, dG0_gc, P_N_rc, P_N_gc


def init():
    from component_contribution.parameters import CCModelParameters
    from component_contribution.preprocessor import Preprocessor
    from equilibrator_api import ComponentContribution, Q_
    from metag.validation.reference_transform import Transformer, load_modelseed_reactions
    p = CCModelParameters.from_npz(Path.home() / '.cache/equilibrator/cc_params.npz')
    tr = Transformer(load_modelseed_reactions(FROZEN / 'sources/modelseed'))
    cc = tr.cc
    ids = p.train_G.index.tolist()
    _G.update(p=p, tr=tr, Q_=Q_, pre=Preprocessor(p), S=p.train_S.values, G=p.train_G.values,
              b=np.ravel(p.train_b), w=np.ravel(p.train_w), Nc=len(ids), Ng=int(p.dimensions.at['Ng', 'number']),
              row={c: i for i, c in enumerate(ids)},
              currency={cc.ccache.get_compound('seed:' + c).id for c in CURRENCY},
              water=cc.ccache.get_compound('seed:cpd00001').id)


def predict(rid):
    g = _G
    rxn = g['tr'].reaction(rid)
    if rxn is None:
        return rid, {'status': 'no_reaction'}
    x = np.zeros(g['Nc'] + g['Ng'])
    support = set()
    for cpd, coeff in rxn.items(protons=False):
        v = g['pre'].get_compound_vector(cpd)
        if v is None:
            return rid, {'status': 'undecomposable_compound'}
        x += coeff * v
        if cpd.id != g['water']:
            support.add(cpd.id)
    Q_ = g['Q_']
    transform = float(rxn.transform(p_h=Q_(7.0), ionic_strength=Q_('0M'), temperature=Q_('298.15K'),
                                    p_mg=Q_(14.0)).m_as('kJ/mol'))
    S, G, b, w = g['S'], g['G'], g['b'], g['w']
    nz = [set(np.nonzero(S[:, j])[0]) for j in range(S.shape[1])]
    rows = {g['row'][c] for c in support if c in g['row']}
    held = {'novel_all': rows,
            'novel_substrates': {g['row'][c] for c in support - g['currency'] if c in g['row']} or rows}
    out = {'status': 'ok', 'transform': transform, 'n_train_cols': S.shape[1]}
    regimes = {'published': np.ones(S.shape[1], bool),
               'leave_reaction_out': np.array([(c - {g['row'].get(g['water'])}) != rows for c in nz])}
    for name, h in held.items():
        regimes[name] = np.array([not (c & h) for c in nz])
    for name, keep in regimes.items():
        cc_, gc_, P_N_rc, P_N_gc = solve(S[:, keep], G, b[keep], w[keep])
        mu = float(np.hstack([cc_, gc_[:g['Ng']]]) @ x)
        xc, xg = x[:g['Nc']], np.concatenate([x[g['Nc']:], np.zeros(G.shape[1] - g['Ng'])])
        resid = G.T @ (P_N_rc @ xc) + xg
        uncovered = float(np.linalg.norm(P_N_gc @ resid)) > 1e-6
        out[name] = {'dG': None if uncovered else mu + transform, 'n_dropped': int((~keep).sum()),
                     'uncovered': uncovered}
    return rid, out


def main():
    init()
    g = _G
    cc_, gc_, _, _ = solve(g['S'], g['G'], g['b'], g['w'])
    err = float(np.abs(cc_ - np.ravel(g['p'].dG0_cc)).max())
    print('reproduces published dG0_cc to', err, flush=True)
    assert err < 1e-4
    with Pool(int(os.environ.get('NPROC', 8)), initializer=init) as pool:
        targets = json.loads((HERE / 'cc_targets.json').read_text())
        res = dict(pool.imap_unordered(predict, targets, chunksize=2))
    (HERE / 'cc_novelty.json').write_text(json.dumps({'reproduction_max_abs_kJ': err, 'currency_seed_ids': CURRENCY,
                                                      'predictions': res}, indent=1) + '\n')
    print('done', sum(r['status'] == 'ok' for r in res.values()), flush=True)


if __name__ == '__main__':
    main()
