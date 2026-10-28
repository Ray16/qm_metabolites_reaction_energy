"""Regression checks for defects reproduced during the publication code review."""
import json
import sys

import pytest

from metag import pipeline as p, uncertainty as uq
from metag.reactions import stoichiometric_extent


def equation(scale=1):
    return {"species": {"ethanol": [-scale, 0, "CCO"], "ether": [scale, 0, "COC"]}}


@pytest.mark.parametrize("scale", [0.5, 2, 10, -1, -2])
def test_class_uncertainty_is_extensive(monkeypatch, scale):
    # Synthetic fully identified calibration, distinct from the historical
    # packaged artifact whose source fingerprint is unavailable.
    monkeypatch.setattr(uq, 'CALIB_CONFIG', dict(uq.CALIB_CONFIG, estimator_source_sha256='a' * 64))
    def score(n):
        sp = equation(n)["species"]
        interval = uq.prediction_interval("isomerase", [v[2] for v in sp.values()], 0,
                         species=sp, U_samp=3*abs(n), config=uq.CALIB_CONFIG)
        sigma, _ = uq.reaction_sigma("isomerase", [v[2] for v in sp.values()],
                         U_samp=3*abs(n), species=sp, config=uq.CALIB_CONFIG)
        assert abs(sigma - interval[3]["sigma"]) < 0.11
        return interval
    base, changed = score(1), score(scale)
    assert changed[1] == pytest.approx(abs(scale)*base[1], abs=0.6)
    assert changed[3]["coverage_calibrated"]


def test_benchmark_equations_have_unit_extent():
    assert all(stoichiometric_extent(rx["species"]) == 1 for rx in p.REACTIONS.values())


@pytest.mark.parametrize("scale", [0.5, 2, 10])
def test_routing_and_point_estimate_are_extent_invariant(monkeypatch, scale):
    # Without normalization, the doubled equation bypasses a core builder
    # restricted to unit stoichiometry and gets a different species energy.
    from metag.routing import truncate
    monkeypatch.setattr(p, "_flag", lambda name: name in ("AUTO_TRUNCATE", "STD_STATE_1M"))
    def build(species, **kw):
        if any(abs(v[0]) != 1 for v in species.values()):
            return None
        return {"a": [-1, 0, "CCCO"], "b": [1, 0, "CCOC"]}, 0
    monkeypatch.setattr(truncate, "build_truncated_reaction", build)
    energies = {"CCO": -10, "COC": -8, "CCCO": -30, "CCOC": -25}
    monkeypatch.setattr(p, "implicit_G", lambda pu, q, smi, *a, **kw: (energies[smi], 0.5))
    base = p.score_reaction(None, equation(), log=lambda *_: None)
    changed = p.score_reaction(None, equation(scale), log=lambda *_: None)
    assert changed["routes"]["truncated"]
    assert changed["dG"] == pytest.approx(scale * base["dG"], abs=0.11)
    assert changed['sigma_pred'] is None and base['sigma_pred'] is None
    assert changed['uncertainty_diagnostic']['sigma_kj']==pytest.approx(scale*base['uncertainty_diagnostic']['sigma_kj'],abs=.6)


def test_routing_preserves_supplied_pka_site_extent():
    rx = equation(2)
    rx["pka_sites"] = [["react", 4.5], ["prod", 6.0, "base", 2]]
    routed, _, _ = p.route_reaction(rx, allow_truncate=False, log=lambda *_: None)
    assert routed["pka_sites"] == [["react", 4.5, "acid", 1.0], ["prod", 6.0, "base", 2.0]]


@pytest.mark.parametrize("species", [equation()["species"],
    {"L": [-1, 0, "N[C@@H](C)C(=O)O"], "D": [1, 0, "N[C@H](C)C(=O)O"]}])
def test_cv_groups_hold_equivalent_equations_together(species):
    from metag.tools.calibrate import reaction_groups
    inputs = {str(n): {"species": {k: [n*c, q, s] for k, (c, q, s) in species.items()}}
              for n in (1, -1, 2, 0.5)}
    assert len(set(reaction_groups(inputs).values())) == 1


def test_feature_uncertainty_scales_with_extent():
    from metag.uq_features import reaction_features
    sp = equation()["species"]
    raw = reaction_features(sp, sp, {}, {}, 1.0)
    a = uq.feature_interval(raw, 0, species=sp, extent=1)
    b = uq.feature_interval(raw, 0, species=sp, extent=2)
    assert b[0] == pytest.approx(2*a[0], abs=0.11)
    assert b[2] == pytest.approx(2*a[2], abs=0.11)


def test_score_without_optional_note(monkeypatch):
    # Force all routes off, so none can incidentally supply a note.
    monkeypatch.setattr(p, "_flag", lambda name: name == "STD_STATE_1M")
    monkeypatch.setattr(p, "implicit_G", lambda *a, **k: (-50.0, 0.2))
    r = p.score_reaction(None, equation(), log=lambda *_: None)
    assert r["dG"] == 0 and r["note"] == ""


def test_cli_records_all_failures_and_keeps_completed_work(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "OUT", str(tmp_path))
    monkeypatch.setattr(p, "REACTIONS", {k: equation() for k in ("ok", "none", "invalid", "exception")})
    monkeypatch.setattr(p, "load_uma", lambda *a: None)
    monkeypatch.setattr(sys, "argv", ["metag"])
    good = {"reaction": "ok", "dG": 1.0, "dG_raw": 1.0, "sigma_pred": 2.0,
            "exp": None, "err": [], "explicit": False}
    def run(pu, key, *args):
        if key == "ok":
            return good
        # Results must already be persisted before the next computation begins.
        assert json.loads((tmp_path / "unified_pipeline_all.json").read_text())[0] == good
        if key == "none":
            return None
        if key == "exception":
            raise ValueError("unsupported species")
        return p._no_estimate(key, equation(), {}, "unbalanced", lambda *_: None)
    monkeypatch.setattr(p, "run_reaction", run)
    p.main()
    records = json.loads((tmp_path / "unified_pipeline_all.json").read_text())
    assert [r["reaction"] for r in records] == ["ok", "none", "invalid", "exception"]
    assert all(r["dG"] is None and not r["ci_info"]["coverage_calibrated"] for r in records[1:])


def test_failed_hydration_does_not_return_calibrated_point(monkeypatch):
    rx = {"species": {"ethanol": [-1, 0, "CCO"], "acetaldehyde": [1, 0, "CC=O"],
                       "hydrogen": [1, 0, "[H][H]"]}}
    monkeypatch.setattr(p, "implicit_G", lambda pu, q, smi, *a, **k:
                        (None, None) if smi == "CC(O)O" else (-50.0, 0.1))
    monkeypatch.setattr(p, "water_ref_G", lambda *a: -10.0)
    result = p.score_reaction(None, rx, log=lambda *_: None)
    assert result["dG"] is None
    assert "incomplete hydration" in result["suspect"]
    assert result["ci_info"]["coverage_calibrated"] is False


@pytest.mark.parametrize("energy,sigma", [(float("nan"), 1), (float("inf"), 1), (-10, -1), (-10, float("nan"))])
def test_cache_rejects_invalid_values(monkeypatch, tmp_path, energy, sigma):
    from metag.energetics import species_cache as c
    monkeypatch.setattr(c, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(c, "_ENABLED", True)
    c.put("CCO", 0, "implicit", {}, energy, sigma)
    assert c.get("CCO", 0, "implicit", {}) is None
    # Also reject bad files produced by previous versions or damaged outside this API.
    path, smi = c._path("CCO", 0, "implicit", {})
    from pathlib import Path
    Path(path).write_text(json.dumps({"smi": smi, "q": 0, "method": "implicit", "settings": {},
                                    "ver": c.CACHE_VERSION, "G": energy, "sigma": sigma}))
    assert c.get("CCO", 0, "implicit", {}) is None


def test_cache_checks_identity_and_cleans_failed_publication(monkeypatch, tmp_path):
    from metag.energetics import species_cache as c
    from pathlib import Path
    monkeypatch.setattr(c, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(c, "_ENABLED", True)
    c.put("CCO", 0, "implicit", {}, -50, 1)
    path, _ = c._path("CCO", 0, "implicit", {})
    record = json.loads(Path(path).read_text())
    record["settings"] = {"wrong_model": True}
    Path(path).write_text(json.dumps(record))
    assert c.get("CCO", 0, "implicit", {}) is None
    c.put("CO", 0, "implicit", {}, -50, 1, meta={"bad": object()})
    assert not list(tmp_path.glob("*.tmp"))
