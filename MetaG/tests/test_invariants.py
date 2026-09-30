"""Thermodynamic invariants of the assembled estimator + failure injection (mocked QM, no GPU).

  * reversal:      score(-R) = -score(R)                    (anchors included)
  * extensivity:   score(k R) = k score(R)                  (no fixed magnitude ceiling; anchors scale)
  * cancellation:  adding the same spectator to both sides changes nothing
  * failure injection at the external boundaries: an xtb failure on a low-energy conformer, or a species
    whose every conformer rearranges, never yields a normal result under the requested species' identity."""
import numpy as np
import pytest

P = pytest.importorskip("metag.pipeline")

G = {"CCO": -100.0, "CC=O": -50.0, "[HH]": -5.0, "CC(=O)O": -300.0, "CCOC(C)=O": -350.0, "C": -30.0,
     "O=C=O": -350.0, "CCCCCC": -600.0}


@pytest.fixture
def mocked(monkeypatch):
    canon = lambda s: P.Chem.MolToSmiles(P.Chem.MolFromSmiles(s))
    monkeypatch.setattr(P, "implicit_G", lambda pu, q, smi, *a, **k: (G[canon(smi)], 0.5))
    monkeypatch.setattr(P, "water_ref_G", lambda pu, log=None: -200.0)
    for f in ("COFACTOR_RING", "AUTO_TRUNCATE", "PH0_AUTO", "ALDEHYDE_HYDRATION", "WATER_REF_HYDROLYASE",
              "TRUNC_VALIDATE"):
        monkeypatch.setenv(f, "0")
    return monkeypatch


def _score(sp, n_h=0):
    return P.score_reaction(None, {"note": "t", "n_Hplus": n_h, "species": sp}, log=lambda *_: None)


EST = {"A": [-1, 0, "CC(=O)O"], "B": [-1, 0, "CCO"], "E": [1, 0, "CCOC(C)=O"], "W": [1, 0, "O"]}
DECARB = {"A": [-1, 0, "CC(=O)O"], "B": [1, 0, "C"], "C": [1, 0, "O=C=O"]}


def _rev(sp):
    return {k: [-v[0], v[1], v[2]] for k, v in sp.items()}


def _mul(sp, k):
    return {n: [k * v[0], v[1], v[2]] for n, v in sp.items()}


@pytest.mark.parametrize("sp", [EST, DECARB])
def test_reversal_antisymmetric(mocked, sp):
    f, r = _score(sp), _score(_rev(sp))
    assert f["dG"] is not None and abs(f["dG"] + r["dG"]) < 0.11 and abs(f["dG_raw"] + r["dG_raw"]) < 0.11


@pytest.mark.parametrize("k", [0.5, 2, 5])
def test_extensive_in_stoichiometry(mocked, k):
    base = _score(DECARB)["dG_raw"]
    big = _score(_mul(DECARB, k))
    assert big["suspect"] is None and abs(big["dG_raw"] - k * base) < 0.11 * max(1, k)


def test_large_extensive_reaction_not_rejected(mocked, monkeypatch):
    # a physically valid ~ -1000 kJ reaction must be scored, not failed by a magnitude ceiling
    monkeypatch.setitem(G, "CCOC(C)=O", -1350.0)
    r = _score(EST)
    assert r["suspect"] is None and r["dG_raw"] < -900


def test_spectator_cancels(mocked):
    base = _score(DECARB)["dG_raw"]
    with_spec = _score(dict(DECARB, S1=[-1, 0, "CCCCCC"], S2=[1, 0, "CCCCCC"]))
    assert abs(with_spec["dG_raw"] - base) < 1e-6


def test_stages_add_up(mocked):
    r = _score(EST)
    st = r["stages"]
    assert abs(st["species_sum"] + st["proton"] + st["pka_transform"] - st["unanchored"]) < 0.02
    assert abs(st["unanchored"] + st.get("water_ref", 0) + st.get("anchor", 0) - st["reported"]) < 0.02
    assert r["conditions"]["pH"] == 7.0 and r["conditions"]["ionic_strength_M"] == 0.0


def test_unresolved_means_ci_contains_zero(mocked):
    r = _score(EST)
    lo, hi = r["ci95"]
    assert r["unresolved"] == (lo <= 0 <= hi)


def test_anchor_extensive():
    from metag.routing import anchor as A
    ck = {"ATP": [-1, -4, "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O"],
          "Cr": [-1, 0, "CN(CC(=O)[O-])C(N)=[NH2+]"],
          "ADP": [1, -3, "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O"],
          "PCr": [1, -2, "CN(CC(=O)[O-])C(N)=[NH+]P(=O)([O-])[O-]"]}
    a1 = A.anchor_correct(40.0, ck)[0]
    a2 = A.anchor_correct(80.0, _mul(ck, 2))[0]
    assert abs(a2 - 2 * a1) < 1e-9


# ---------------------------------------------------------------- failure injection (implicit_G internals)
class _PU:
    metag_model = P._MODEL


def _fake_backend(monkeypatch, solv=lambda i: -10.0, rearrange=False):
    from ase import Atoms
    calls = {"n": 0}
    def pool(smi, q, seed, pool, spin=1):
        m = P.Chem.AddHs(P.Chem.MolFromSmiles(smi))
        from rdkit.Chem import AllChem
        AllChem.EmbedMolecule(m, randomSeed=seed)
        x = m.GetConformer().GetPositions()
        return [Atoms([a.GetSymbol() for a in m.GetAtoms()], x + 0.01 * i, info={"charge": q, "spin": spin})
                for i in range(3)]
    monkeypatch.setattr(P, "pool_confs", pool)
    monkeypatch.setattr(P, "batched_energies", lambda pu, c: np.arange(len(c), dtype=float))
    monkeypatch.setattr(P, "batched_fire", lambda pu, sel, **k: (sel, np.arange(len(sel)) * 0.1 - 1000.0,
                                                                np.ones(len(sel), bool)))
    def dg(sym, pos, q, model, mult=1):
        calls["n"] += 1
        return solv(calls["n"])
    monkeypatch.setattr(P, "dgsolv", dg)
    monkeypatch.setattr(P, "same_connectivity", lambda a, g: not rearrange)
    monkeypatch.setattr(P, "uma_gibbs_corr", lambda *a, **k: (5.0, {"n_imag": 0, "max_imag_cm": 0.0}))
    monkeypatch.setenv("SPECIES_CACHE", "0")
    monkeypatch.setattr(P._sc, "_ENABLED", False)


def test_xtb_failure_on_low_energy_conformer_fails_species(monkeypatch):
    _fake_backend(monkeypatch, solv=lambda i: None)          # every solvation call fails (incl. retry)
    w = []
    g, s = P.implicit_G(_PU(), 0, "CCO", None, None, None, lambda *_: None, "x", w)
    assert g is None and s is None


def test_all_conformers_rearranged_raises(monkeypatch):
    _fake_backend(monkeypatch, rearrange=True)
    with pytest.raises(P.SpeciesRearranged):
        P.implicit_G(_PU(), 0, "CCO", None, None, None, lambda *_: None, "x", [])


def test_rearranged_species_reroutes_then_fails_closed(monkeypatch):
    for f in ("COFACTOR_RING", "AUTO_TRUNCATE", "PH0_AUTO", "ALDEHYDE_HYDRATION", "WATER_REF_HYDROLYASE"):
        monkeypatch.setenv(f, "0")
    monkeypatch.setattr(P, "water_ref_G", lambda pu, log=None: -200.0)
    seen = []
    def always_rearranges(pu, q, smi, *a, **k):
        seen.append(smi); raise P.SpeciesRearranged("x", smi, 3)
    monkeypatch.setattr(P, "implicit_G", always_rearranges)
    r = _score(DECARB)
    assert r is None and len(seen) >= 2                       # tried once, re-routed once, then no estimate


def test_routing_exception_fails_closed(mocked, monkeypatch):
    import metag.routing.cofactor_cores as CC
    monkeypatch.setenv("COFACTOR_RING", "1")
    def boom(sp):
        raise RuntimeError("injected")
    monkeypatch.setattr(CC, "cofactor_ring", boom)
    r = _score(DECARB)
    assert r["dG"] is None and "routing error" in r["suspect"] and r["ci_info"]["externally_calibrated"] is False


# ---------------------------------------------------------------- spin, thermal projection, calibration
def test_spin_from_radicals_and_metal_refusal():
    from metag.energetics.conformers import spin_multiplicity
    assert spin_multiplicity("O=O", 0) == 3 and spin_multiplicity("[CH3]", 0) == 2
    assert spin_multiplicity("[CH2]", 0) == 3                 # triplet methylene (2 radical electrons)
    with pytest.raises(ValueError):
        spin_multiplicity("[Fe+2]", 2)


def test_imaginary_mode_is_detected_not_floored():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from ase import Atoms
    from metag.energetics.thermal import internal_vib_energies
    m = Chem.AddHs(Chem.MolFromSmiles("CC")); AllChem.EmbedMolecule(m, randomSeed=1)
    mp = AllChem.MMFFGetMoleculeProperties(m); ff = AllChem.MMFFGetMoleculeForceField(m, mp)
    ff.MMFFAddTorsionConstraint(2, 0, 1, 5, False, 0.0, 0.0, 1e4); ff.Minimize(maxIts=5000)
    pos = m.GetConformer().GetPositions()
    def grad(x):
        c = m.GetConformer()
        for i, p in enumerate(x.reshape(-1, 3)):
            c.SetAtomPosition(i, p.tolist())
        return np.array(AllChem.MMFFGetMoleculeForceField(m, mp).CalcGrad()) * 0.0433641
    x0 = pos.ravel(); H = np.zeros((x0.size,) * 2)
    for k in range(x0.size):
        xp = x0.copy(); xp[k] += 1e-3; xm = x0.copy(); xm[k] -= 1e-3
        H[k] = (grad(xp) - grad(xm)) / 2e-3
    vib, imag = internal_vib_energies(Atoms([a.GetSymbol() for a in m.GetAtoms()], pos), 0.5 * (H + H.T),
                                      "nonlinear")
    assert len(vib) + len(imag) == 3 * m.GetNumAtoms() - 6 and len(imag) == 1 and imag[0] > 200


def test_calibration_keeps_large_errors_and_groups_folds():
    from metag.tools import calibrate as C
    recs = [{"rid": f"r{i}", "class": "c", "dG": float(i % 7), "dG_raw": float(i % 7), "exp": 0.0, "U_samp": 0}
            for i in range(40)] + [{"rid": "big", "class": "c", "dG": 500.0, "dG_raw": 500.0, "exp": 0.0}]
    c = C.calibrate(recs, groups={f"r{i}": f"g{i // 4}" for i in range(40)})
    assert c["n_reactions"] == 41 and c["n_abs_err_gt_200"] == 1 and "clusters" in c["cv_grouping"]
