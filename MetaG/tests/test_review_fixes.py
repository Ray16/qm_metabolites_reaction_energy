"""Regression tests for the 2026-09-30 code-review fixes (pure logic / mocked QM, no GPU).

Each test pins one invariant that was silently violated before: routed reactions stay balanced, invalid
reactions return no estimate, intervals are only "calibrated" for the calibrated configuration, water is
water however it is written, fractional coefficients are not truncated, the pKa ladders of phosphoryl-N /
acyl-phosphate / carbonic acid, the decarboxylase σ-class, the activated-aldehyde hydration, and the
float32-aware conformer dedup."""
import numpy as np
import pytest

from metag.chem import is_water, is_balanced, reaction_residual
from metag.routing import pka_transform as PK
from metag import uncertainty as U

P = pytest.importorskip("metag.pipeline")


# ---------------------------------------------------------------- shared helpers (metag.chem)
def test_water_any_spelling():
    for s in ("O", "[OH2]", "[H]O[H]"):
        assert is_water(s, 0)
    assert not is_water("O", -1) and not is_water("[OH-]", -1) and not is_water("OO", 0)


def test_balance_with_protons_and_fractions():
    # 2 H2O2 -> 2 H2O + O2 written with fractional coefficients
    assert is_balanced({"a": [-1, 0, "OO"], "b": [1, 0, "O"], "c": [0.5, 0, "O=O"]})
    assert not is_balanced({"a": [-1, 0, "OO"], "b": [1, 0, "O"]})
    # lactate + NAD-ring model -> pyruvate + dihydro ring + H+  (proton closes H and charge)
    lac = {"lac": [-1, -1, "CC(O)C(=O)[O-]"], "nad": [-1, 1, "C[n+]1cccc(C(N)=O)c1"],
           "pyr": [1, -1, "CC(=O)C(=O)[O-]"], "nadh": [1, 0, "CN1C=CCC(C(N)=O)=C1"]}
    assert is_balanced(lac, n_hplus=1) and not is_balanced(lac, n_hplus=0)
    res, _ = reaction_residual({"x": [-1, -1, "CCO"]})          # declared charge != SMILES charge
    assert "declared_charge_mismatch" in res


# ---------------------------------------------------------------- input validation + fail-closed
def test_validate_reaction_rejects_charge_mismatch():
    with pytest.raises(ValueError):
        P.validate_reaction({"species": {"a": [-1, -1, "CCO"], "b": [1, 0, "CC=O"]}})
    with pytest.raises(ValueError):
        P.validate_reaction({"species": {"a": [-1, 0, "CC(=O)[O-]"]}})
    with pytest.raises(ValueError):
        P.validate_reaction({"species": {"a": [-1, 0, "not a smiles"]}})
    P.validate_reaction({"species": {"a": [-1, -1, "CC(=O)[O-]"]}})   # consistent -> no error


@pytest.mark.parametrize("record", [
    [-1, 0],
    [0, 0, "CCO"],
    [float("nan"), 0, "CCO"],
    [-1, 0.5, "CCO"],
    [-1, 0, ""],
])
def test_validate_reaction_rejects_malformed_species_records(record):
    with pytest.raises(ValueError):
        P.validate_reaction({"species": {"a": record}})
    with pytest.raises(ValueError):
        P.validate_reaction({"n_Hplus": float("inf"), "species": {"a": [-1, 0, "CCO"]}})


def test_auxiliary_solvation_failure_does_not_remove_primary_minimum():
    # SOLV_ALSO is diagnostic: a failed auxiliary value must leave the primary minimum intact. Conversely, a
    # partial auxiliary list must never be filtered/renormalized into a biased auxiliary Boltzmann ensemble.
    from metag.energetics.conformers import UniqueMinima
    aux = "cosmo" if P.SOLV_MODEL != "cosmo" else "alpb"
    atoms = P.Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    uniq, also = UniqueMinima(), {aux: []}
    P._add_minimum(uniq, also, atoms, -10.0, {P.SOLV_MODEL: -2.0, aux: None})
    assert uniq.G == [-12.0] and also[aux] == [None]
    assert P._complete_auxiliary_values(also[aux]) is None
    assert P._complete_auxiliary_values([-13.0, -12.5]) == [-13.0, -12.5]


@pytest.fixture
def mocked(monkeypatch):
    G = {"CCO": -100.0, "CC=O": -50.0, "[HH]": -5.0, "CC(=O)O": -300.0, "CCOC(C)=O": -350.0}
    monkeypatch.setattr(P, "implicit_G", lambda pu, q, smi, *a, **k: (G[P.Chem.MolToSmiles(P.Chem.MolFromSmiles(smi))], 0.5))
    monkeypatch.setattr(P, "water_ref_G", lambda pu, log=None: -200.0)
    for f in ("COFACTOR_RING", "AUTO_TRUNCATE", "PH0_AUTO", "ANCHOR_CORRECT", "ALDEHYDE_HYDRATION",
              "WATER_REF_HYDROLYASE", "TRUNC_VALIDATE"):
        monkeypatch.setenv(f, "0")
    return monkeypatch


def _score(sp, n_h=0):
    return P.score_reaction(None, {"note": "test", "n_Hplus": n_h, "species": sp}, log=lambda *_: None)


def test_unbalanced_reaction_returns_no_estimate(mocked):
    # proton count wrong -> charge/element imbalance -> NO number, no interval, calibration off
    r = _score({"A": [-1, 0, "CCO"], "B": [1, 0, "CC=O"], "H2": [1, 0, "[HH]"]}, n_h=1)
    assert r["suspect"] and r["dG"] is None and r["dG_raw"] is None
    assert r["ci95"] == [None, None] and r["sigma_pred"] is None
    assert r["ci_info"]["externally_calibrated"] is False
    assert r["dG_raw_unreliable"] is not None                 # kept for diagnosis only


def test_water_spelling_does_not_change_dG(mocked):
    base = {"A": [-1, 0, "CC(=O)O"], "B": [-1, 0, "CCO"], "E": [1, 0, "CCOC(C)=O"]}
    a = _score(dict(base, W=[1, 0, "O"]))["dG_raw"]
    b = _score(dict(base, W=[1, 0, "[H]O[H]"]))["dG_raw"]
    assert a is not None and a == b                          # "[H]O[H]" used to be scored as a 1 M solute


def test_balance_guard_reverts_unbalanced_rewrite(monkeypatch):
    # a cofactor core that drops atoms (the mixed-disulfide failure) must be reverted, not scored
    import metag.routing.cofactor_cores as CC
    monkeypatch.setenv("AUTO_TRUNCATE", "0"); monkeypatch.setenv("PH0_AUTO", "0")
    rx = {"note": "t", "n_Hplus": 0, "species": {"A": [-1, 0, "CCO"], "B": [1, 0, "CC=O"], "H2": [1, 0, "[HH]"]}}
    monkeypatch.setattr(CC, "cofactor_ring", lambda sp: {"A": [-1, 0, "CCCO"], "B": [1, 0, "CC=O"], "H2": [1, 0, "[HH]"]})
    new, routes, _ = P.route_reaction(rx, log=lambda *_: None)
    assert new["species"] == rx["species"] and routes["cofactor_ring"] is False
    assert any("cofactor_ring" in e and "reverted" in e for e in routes["reverted"]) and not routes["errors"]


# ---------------------------------------------------------------- calibration fingerprint
def test_config_mismatch_disables_calibration(monkeypatch):
    cfg = P.effective_config()
    monkeypatch.setattr(U, "CALIB_CONFIG", dict(cfg))
    assert U.calibration_mismatch(cfg) == []
    assert U.calibration_mismatch(dict(cfg, solv_model="gbsa"))            # any difference is reported
    assert U.calibration_mismatch(None)                                    # unverifiable -> mismatch
    lo, hi, c, info = U.prediction_interval("fumarate hydratase", ["O"], 0.0, config=dict(cfg, solv_model="gbsa"))
    assert info["externally_calibrated"] is False and "solv_model" in info["calibration_scope"]
    monkeypatch.setattr(U, "CALIB_CONFIG", None)                           # artifact without fingerprint
    assert U.calibration_mismatch(cfg)


def test_flag_defaults_single_source():
    import inspect, re
    src = inspect.getsource(P)
    assert not re.search(r'_flag\("[A-Z0-9_]+", default=', src)          # defaults live in FLAG_DEFAULTS
    for name in re.findall(r'_flag\("([A-Z0-9_]+)"\)', src):
        assert name in P.FLAG_DEFAULTS, name


def test_config_fingerprints_numeric_acceptance_controls(monkeypatch):
    base = P.effective_config()
    for env, key, value in (
        ("DG_SANITY_KJ", "dg_sanity_kj", "321"),
        ("TRUNC_VALIDATE_TOL", "trunc_validate_tol", "7"),
        ("SMD_THRESHOLD", "smd_threshold", "9"),
    ):
        monkeypatch.setenv(env, value)
        changed = P.effective_config()
        assert changed[key] != base[key]
        monkeypatch.delenv(env)
    assert base["explicit_sampling"] == [P.N_EXPLICIT_SEEDS, P.EXPLICIT_KEEP]


# ---------------------------------------------------------------- pKa ladders
def _pkas(smi):
    return sorted(p for _, p in PK._classify_species(PK._canonicalize_maxanion(smi))[1])


def test_phosphoramidate_acylphosphate_carbonate_ladders():
    assert _pkas("CN(CC(=O)[O-])C(N)=[NH+]P(=O)([O-])[O-]") == sorted([PK.CARBOXYL_PKA] + PK.P_N_LADDER)
    assert _pkas("CC(=O)OP(=O)([O-])[O-]") == PK.ACYL_P_LADDER               # acetyl phosphate
    assert _pkas("O=C([O-])O") == PK.CARBONATE_LADDER                         # bicarbonate
    assert _pkas("COP(=O)([O-])[O-]") == PK.P_LADDER[1]                       # plain monoester unchanged
    assert _pkas("O=P([O-])([O-])O") == PK.P_LADDER[0][:3]                     # free Pi unchanged
    assert _pkas("CC(=O)[O-]") == [PK.CARBOXYL_PKA]                          # carboxylate unchanged


def test_ph0_fractional_coefficients_carry_multiplicity():
    # 1/2 acetate + 1/4 O2 -> 1/2 glycolate (balanced, fractional)
    sp = {"a": [-0.5, -1, "CC(=O)[O-]"], "o2": [-0.25, 0, "O=O"], "b": [0.5, -1, "OCC(=O)[O-]"]}
    assert is_balanced(sp)
    ns, sites, nh = PK.build_ph0_reaction(sp, 0)
    assert all(len(s) == 4 and s[3] == 0.5 for s in sites) and len(sites) == 2


def test_redox_proton_is_carried_not_refused():
    # lactate + NAD+ -> pyruvate + NADH + H+ (ring model): neutralized reaction is short one H;
    # carried as n_H+ instead of refusing pH-0 (the same metabolites are pH-0 routed elsewhere)
    lac = {"lac": [-1, -1, "CC(O)C(=O)[O-]"], "nad": [-1, 1, "C[n+]1cccc(C(N)=O)c1"],
           "pyr": [1, -1, "CC(=O)C(=O)[O-]"], "nadh": [1, 0, "CN1C=CCC(C(N)=O)=C1"]}
    why = []
    out = PK.build_ph0_reaction(lac, 1, base=False, why=why)
    assert out is not None, why
    ns, sites, nh = out
    assert is_balanced(ns, nh)


# ---------------------------------------------------------------- σ-class keywords
def test_decarboxylase_not_kinase():
    for note in ("glutamate decarboxylase", "pyruvate decarboxylase", "ornithine decarboxylase"):
        assert U.mech_class(note, []) != "kinase/phosphotransfer"
    assert U.mech_class("pyruvate carboxylase", []) == "kinase/phosphotransfer"
    assert U.mech_class("argininosuccinate lyase", []) == "ammonia-lyase"
    assert U.mech_class("fumarase", []) == "hydratase"


# ---------------------------------------------------------------- aldehyde hydration
def test_gem_diol_hydrates_the_activated_aldehyde():
    from metag.routing.aldehyde_hydration import gem_diol
    assert gem_diol("O=CCCC(=O)C=O") == "O=CCCC(=O)C(O)O"       # alpha-keto CHO, not the alkyl CHO
    assert gem_diol("O=CC(=O)[O-]") == "O=C([O-])C(O)O"          # glyoxylate
    assert gem_diol("O=CCCC") is None                            # non-activated: gated out


# ---------------------------------------------------------------- conformer dedup + connectivity
def test_dedup_energy_tolerance_exceeds_float32_step():
    from metag.energetics.conformers import UniqueMinima, EV2KJ
    u = UniqueMinima()
    e_kj = -72000.0 * EV2KJ                                        # ATP-size total energy
    step = float(np.spacing(np.float32(72000.0))) * EV2KJ
    assert step > 0.5                                              # the old 0.5 kJ tolerance was below it
    assert u._e_tol(e_kj) >= 2 * step and u._e_tol(-10.0) == u.e_tol
    assert UniqueMinima(e_tol=-1.0)._e_tol(e_kj) < 0               # legacy never-merge preserved


def test_connectivity_check_flags_proton_transfer():
    from rdkit import Chem
    from metag.energetics.conformers import pool_confs, bond_graph, same_connectivity
    smi = "[NH3+]CC(=O)[O-]"
    m = Chem.AddHs(Chem.MolFromSmiles(smi)); ref = bond_graph(m)
    a = pool_confs(smi, 0, 1, 1)[0]
    assert same_connectivity(a, ref)
    on = next(x.GetIdx() for x in m.GetAtoms() if x.GetSymbol() == "O" and x.GetFormalCharge() == -1)
    hn = next(n.GetIdx() for n in m.GetAtomWithIdx(0).GetNeighbors() if n.GetSymbol() == "H")
    x = a.get_positions(); x[hn] = x[on] + np.array([0.97, 0.0, 0.0]); a.set_positions(x)
    assert not same_connectivity(a, ref)


# ---------------------------------------------------------------- calibration: signed anchor offsets
def test_calibrate_refits_reversed_anchor_with_sign():
    from metag.tools import calibrate as C
    rows = C._normalize([
        {"rid": "f", "class": "k", "dG": 10.0, "dG_raw": 60.0, "exp": 10.0, "U_samp": 0,
         "anchor": {"subclass": "phosphagen", "offset": 50.0, "direction": 1}},
        {"rid": "r", "class": "k", "dG": -10.0, "dG_raw": -60.0, "exp": -10.0, "U_samp": 0,
         "anchor": {"subclass": "phosphagen", "offset": -50.0, "direction": -1}}])
    refs = {"phosphagen": ("tecrdb", {"f", "r"})}
    off = C._fold_offsets(rows, [0, 1], refs)
    assert abs(off["phosphagen"] - 50.0) < 1e-9                    # both readings agree on +50 canonical
    assert all(abs(C._residual(r, off, refs)) < 1e-9 for r in rows)


def test_cycle_closure_keeps_fractional_coefficients():
    from metag.tools.cycle_closure import cycle_basis
    S = np.array([[1.0, -2.0], [-0.5, 1.0]])                       # col0 = 0.5 * ... -> one cycle
    b = cycle_basis(S)
    assert b and len(b[0]) == 2


def test_gate_audit_accepts_historical_calibration_schemas():
    from analysis.gate_audit import class_map
    assert class_map([["rxn1", "hydratase", 1.2]]) == {"rxn1": "hydratase"}
    assert class_map([{"rid": "rxn2", "class": "redox"}]) == {"rxn2": "redox"}
    assert class_map({"rxn3": {"class": "transferase"}}) == {"rxn3": "transferase"}
    assert class_map({"row": {"reaction": "rxn4", "class": "lyase"}}) == {"rxn4": "lyase"}


def test_float32_precision_audit_propagates_stoichiometry():
    from analysis.electronic_precision_audit import reaction_precision
    species = {
        ("CCO", 0): {"ulp_kj": 0.5},
        ("CC=O", 0): {"ulp_kj": 1.0},
    }
    rec = {"species_scored": {
        "a": [-2, 0, "CCO"],
        "b": [1, 0, "CC=O"],
        "water": [1, 0, "O"],
    }}
    out = reaction_precision(rec, species)
    assert out["worst_bound_kj"] == pytest.approx(1.0)
    assert out["rms_scale_kj"] == pytest.approx(np.sqrt(2) / np.sqrt(12))
    assert out["n_cached_species"] == 2


def test_float32_precision_audit_reports_missing_species():
    from analysis.electronic_precision_audit import reaction_precision
    out = reaction_precision(
        {"species_scored": {"a": [-1, 0, "CCO"]}},
        {},
    )
    assert out == {"missing_species": ["a"]}


def test_pka_validation_excludes_cationic_zwitterions():
    from rdkit import Chem
    from analysis.review_fixes.pka_table_validation import acid_only
    assert acid_only(Chem.MolFromSmiles("CC(=O)[O-]"))
    assert not acid_only(Chem.MolFromSmiles("NC(=[NH2+])NCCS(=O)(=O)[O-]"))


def test_routing_consistency_audit_detects_context_dependent_species():
    from analysis.routing_consistency_audit import audit
    original = {
        "r1": {"species": {"A": [-1, -1, "CC(=O)[O-]"]}},
        "r2": {"species": {"A": [1, -1, "CC(=O)[O-]"]}},
    }
    results = {
        "r1": {"species_scored": {"A": [-1, 0, "CC(=O)O"]}, "routes": {"ph0": True}},
        "r2": {"species_scored": {"A_t": [1, 0, "CC"]}, "routes": {"truncated": True}},
    }
    report = audit(original, results)
    assert report["n_context_dependent_species"] == 1
    assert report["conflicts"][0]["n_representations"] == 2


def test_reassembler_cache_settings_require_exact_solvation_provenance():
    from analysis.review_fixes.reassemble import matching_settings
    required = {"model": "uma", "solv": "cpcmx", "physics": "v2"}
    assert matching_settings(required, required, "primary")
    assert not matching_settings(dict(required, via="solv_also"), required, "primary")
    assert matching_settings(dict(required, via="solv_also"), required, "solv_also")
    assert not matching_settings(dict(required, via="other"), required, "solv_also")
    assert not matching_settings(dict(required, physics="v1"), required, "primary")


def test_reassembler_metrics_exclude_suspect_and_missing_predictions():
    from analysis.review_fixes.reassemble import metrics
    rows = {
        "a": {"err": 2.0, "suspect": None},
        "b": {"err": -4.0, "suspect": None},
        "c": {"err": 1000.0, "suspect": "unbalanced"},
        "d": {"err": None, "suspect": None},
    }
    assert metrics(rows) == {
        "n": 2, "mae": 3.0, "median_ae": 3.0, "rmse": 3.162, "bias": -1.0
    }


def test_rescore_source_fingerprint_changes_with_scoring_source(tmp_path, monkeypatch):
    from analysis import tecrdb_rescore
    package = tmp_path / "MetaG/metag"
    package.mkdir(parents=True)
    scoring = package / "pipeline.py"
    scoring.write_text("VERSION = 1\n")
    runner = tmp_path / "runner.py"
    runner.write_text("runner = 1\n")
    monkeypatch.setattr(tecrdb_rescore, "ROOT", str(tmp_path))
    monkeypatch.setattr(tecrdb_rescore, "__file__", str(runner))
    before = tecrdb_rescore.source_fingerprint()
    scoring.write_text("VERSION = 2\n")
    assert tecrdb_rescore.source_fingerprint() != before
