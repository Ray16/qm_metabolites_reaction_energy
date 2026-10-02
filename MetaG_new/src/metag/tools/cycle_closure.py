"""Cycle-closure (state-function) validation of scored reactions -- ground-truth-free.

ΔrG'° is a state function: any linear combination of reactions whose stoichiometry cancels (a cycle,
v with S·v = 0) must have Σ v_j ΔG_j = 0. A pure per-compound method closes exactly (up to conformer
sampling noise). MetaG's reaction-context routing (truncation, pH-0 speciation, cofactor cores,
anchors) can give the SAME compound different effective energies in different reactions; non-closure
beyond sampling noise is the signature of that inconsistency.

Method: weighted least squares for per-compound formation energies f,
    min_f Σ_j ((Sᵀf − ΔG)_j / σ_j)²,
residual r = ΔG − Sᵀf̂ is exactly the part of ΔG that no assignment of compound energies can explain
(the projection onto the cycle space). χ² = Σ (r_j/σ_j)², dof = n_reactions − rank(S). With σ_j the
sampling-only uncertainty (U_samp), χ²/dof ≈ 1 means "closes like a state function"; ≫ 1 means routing
inconsistency. An explicit integer cycle basis (sympy) is reported for small networks.

Compound identity = InChIKey of the charge-neutralized parent, so the same metabolite written in two
protonation states is one compound (ΔrG'° is a transformed quantity over pseudo-isomer groups). H+ is
not a compound (transformed ΔG); water is.

    from metag.tools.cycle_closure import closure_report
    rep = closure_report([{"rid":..., "species": {name: [coeff, q, smi]}, "dG":..., "sigma":...}, ...])

Release criterion (suggested): chi2_per_dof on the deployed configuration with σ = U_samp, and the list
of worst cycles, reviewed before database-scale deployment.
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
_UNCHARGER = rdMolStandardize.Uncharger()
_KEYS = {}


def compound_key(smi):
    """Protonation-invariant compound identity (InChIKey of the neutralized parent); SMILES fallback."""
    if smi in _KEYS:
        return _KEYS[smi]
    k = smi
    m = Chem.MolFromSmiles(smi)
    if m is not None:
        try:
            k = Chem.MolToInchiKey(_UNCHARGER.uncharge(m)) or Chem.MolToSmiles(m)
        except Exception:
            k = Chem.MolToSmiles(m)
    _KEYS[smi] = k
    return k


def stoich_matrix(records):
    """(S, compound_keys): S[i, j] = net coefficient of compound i in reaction j (products +)."""
    keys, idx = [], {}
    cols = []
    for r in records:
        col = {}
        for c, q, smi in (tuple(v)[:3] for v in r["species"].values()):
            k = compound_key(smi)
            col[k] = col.get(k, 0) + c
        cols.append({k: v for k, v in col.items() if v != 0})
        for k in cols[-1]:
            if k not in idx:
                idx[k] = len(keys); keys.append(k)
    S = np.zeros((len(keys), len(records)))
    for j, col in enumerate(cols):
        for k, v in col.items():
            S[idx[k], j] = v
    return S, keys


def _prune_to_cycle_support(S):
    """Indices of reactions that can participate in a cycle: iteratively drop reactions that contain a
    compound appearing in no other remaining reaction (such a reaction has a zero weight in every cycle)."""
    alive = np.ones(S.shape[1], bool)
    while True:
        occ = (S[:, alive] != 0).sum(axis=1)
        drop = alive & np.any((S != 0) & (occ[:, None] == 1), axis=0)
        if not drop.any():
            return np.where(alive)[0]
        alive &= ~drop


def cycle_basis(S, max_reactions=150):
    """Integer basis of the cycle space (null space of S) over the cycle-support reactions, or None if
    the support is too large for exact rational arithmetic. Returns list of {reaction_index: coeff}."""
    support = _prune_to_cycle_support(S)
    if len(support) == 0:
        return []
    if len(support) > max_reactions:
        return None
    import sympy
    sub = S[:, support]
    sub = sub[np.any(sub != 0, axis=1)]
    # exact rationals: astype(int) truncated fractional coefficients (ModelSEED 0.5 O2) to 0
    M = sympy.Matrix([[sympy.nsimplify(float(x), rational=True) for x in row] for row in sub.tolist()])
    out = []
    for v in M.nullspace():
        den = sympy.ilcm(*[x.q for x in v]) if len(v) else 1
        iv = [int(x * den) for x in v]
        g = int(np.gcd.reduce([abs(x) for x in iv if x])) or 1
        out.append({int(support[i]): x // g for i, x in enumerate(iv) if x})
    return out


def closure_report(records, sigma_key="sigma", sigma_floor=1.0, max_cycles_listed=20):
    """records: [{"rid", "species", "dG", sigma_key}] -> dict(chi2, dof, chi2_per_dof, rms_residual,
    per-reaction residuals (worst first), cycles with their closure error)."""
    recs = [r for r in records if r.get("dG") is not None]
    S, keys = stoich_matrix(recs)
    dG = np.array([float(r["dG"]) for r in recs])
    sig = np.array([max(float(r.get(sigma_key) or 0.0), sigma_floor) for r in recs])
    W = 1.0 / sig
    A = (S.T * W[:, None])                                  # weighted design: rows = reactions
    f, *_ = np.linalg.lstsq(A, dG * W, rcond=None)
    resid = dG - S.T @ f
    rank = int(np.linalg.matrix_rank(S)) if S.size else 0
    dof = len(recs) - rank
    chi2 = float(np.sum((resid / sig) ** 2))
    # a reaction lies in a cycle iff its ΔG is constrained by the others, i.e. its leverage in the weighted
    # fit is < 1 (leverage = row norm² of the left singular vectors of A). The old |resid| > 1e-6 test
    # missed reactions whose cycles happen to close exactly.
    if A.size:
        U, sv, _ = np.linalg.svd(A, full_matrices=False)
        r = int(np.sum(sv > sv.max() * max(A.shape) * np.finfo(float).eps)) if sv.size else 0
        leverage = np.sum(U[:, :r] ** 2, axis=1)
    else:
        leverage = np.ones(len(recs))
    in_cycle = leverage < 1.0 - 1e-8
    per = sorted(({"rid": recs[j]["rid"], "residual": round(float(resid[j]), 2),
                   "z": round(float(resid[j] / sig[j]), 2)} for j in range(len(recs)) if in_cycle[j]),
                 key=lambda d: -abs(d["z"]))
    cyc = cycle_basis(S)
    cycles = None
    if cyc is not None:
        cycles = []
        for v in cyc:
            err = float(sum(c * dG[j] for j, c in v.items()))
            s = float(np.sqrt(sum((c * sig[j]) ** 2 for j, c in v.items())))
            cycles.append({"reactions": {recs[j]["rid"]: c for j, c in v.items()},
                           "closure_error": round(err, 2), "sigma": round(s, 2), "z": round(err / s, 2)})
        cycles.sort(key=lambda d: -abs(d["z"]))
    return {"n_reactions": len(recs), "n_compounds": len(keys), "rank": rank, "dof": dof,
            "chi2": round(chi2, 2), "chi2_per_dof": round(chi2 / dof, 3) if dof > 0 else None,
            "rms_residual": round(float(np.sqrt(np.mean(resid[in_cycle] ** 2))), 2) if in_cycle.any() else 0.0,
            "n_in_cycles": int(in_cycle.sum()), "worst_reactions": per[:max_cycles_listed],
            "n_cycles": (len(cycles) if cycles is not None else None),
            "worst_cycles": (cycles[:max_cycles_listed] if cycles is not None else None)}
