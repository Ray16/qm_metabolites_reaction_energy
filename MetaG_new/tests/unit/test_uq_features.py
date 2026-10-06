"""Feature-based per-reaction uncertainty (UQ_MODEL=features)."""
import ast
import inspect
from pathlib import Path

import pytest

from metag import uncertainty as U
from metag import uq_features as F

CREATINE_KINASE = {
    "ATP": [-1, -4, "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O"],
    "Cr": [-1, 0, "CN(CC(=O)[O-])C(N)=[NH2+]"],
    "ADP": [1, -3, "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O"],
    "PCr": [1, -2, "CN(CC(=O)[O-])C(N)=[NH+]P(=O)([O-])[O-]"],
    "H": [1, 1, "[H+]"],
}
ISOMERASE = {"g6p": [-1, -2, "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1OP(=O)([O-])[O-]"],
             "f6p": [1, -2, "OC[C@]1(O)O[C@H](COP(=O)([O-])[O-])[C@@H](O)[C@@H]1O"]}
ROUTES = {"truncated": True, "ntp_core": True, "ph0": True}


def _rev(sp):
    return {k: [-c, q, s] for k, (c, q, s) in sp.items()}


def test_features_read_no_annotations():
    params = set(inspect.signature(F.reaction_features).parameters)
    assert params.isdisjoint({"note", "ec", "enzyme", "reaction_id", "rid"})
    tree = ast.parse(Path(F.__file__).read_text())
    strings = {n.value.lower() for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not strings & {"note", "ec", "ec_number", "enzyme"}


def test_features_and_width_are_reversal_invariant():
    st = {"pka_transform": -7.0}
    fwd = F.reaction_features(CREATINE_KINASE, CREATINE_KINASE, ROUTES, st, 0.9)
    rev = F.reaction_features(_rev(CREATINE_KINASE), _rev(CREATINE_KINASE), ROUTES, {"pka_transform": 7.0}, 0.9)
    assert fwd == rev
    s1, lo1, hi1, _ = U.feature_interval(fwd, 28.5, config={})
    s2, lo2, hi2, _ = U.feature_interval(rev, -28.5, config={})
    assert s1 == s2 and abs((hi1 - lo1) - (hi2 - lo2)) < 0.11


def test_phosphagen_wider_than_isomerase():
    pg = F.reaction_features(CREATINE_KINASE, CREATINE_KINASE, ROUTES, {"pka_transform": -7.0}, 0.9)
    iso = F.reaction_features(ISOMERASE, ISOMERASE, {"ph0": True}, {"pka_transform": 0.0}, 0.3)
    assert U.feature_interval(pg, 0.0, config={})[0] > U.feature_interval(iso, 0.0, config={})[0]


def test_artifact_matches_code_and_records_heldout_coverage():
    art = U._uq_artifact()
    assert art["feature_version"] == F.FEATURE_VERSION
    assert art["features"] == F.FEATURES
    assert len(art["coef"]) == len(F.FEATURES)
    assert art["nested_cv"]["coverage95"] >= 0.94
    assert art["conformal_q"]["0.95"] > art["conformal_q"]["0.68"] > 0


def test_out_of_range_or_unverified_config_is_nominal():
    raw = F.reaction_features(ISOMERASE, ISOMERASE, {"ph0": True}, {"pka_transform": 0.0}, 0.3)
    *_, info = U.feature_interval(raw, 0.0, config=None)
    assert info["coverage_calibrated"] is False
    raw = dict(raw, sum_heavy=10_000)
    *_, info = U.feature_interval(raw, 0.0, config=U._uq_artifact()["config"])
    assert info["coverage_calibrated"] is False and "sum_heavy" in info["extrapolated_features"]


def test_uq_model_switch(monkeypatch):
    monkeypatch.delenv("UQ_MODEL", raising=False)
    assert U.uq_model() == "class"
    monkeypatch.setenv("UQ_MODEL", "features")
    assert U.uq_model() == "features"
    monkeypatch.setenv("UQ_MODEL", "bogus")
    with pytest.raises(ValueError):
        U.uq_model()
