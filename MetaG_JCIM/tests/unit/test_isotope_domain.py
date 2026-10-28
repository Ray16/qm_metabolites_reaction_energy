import pytest

from metag.chem import isotope_labels
from metag.energetics.conformers import pool_confs, spin_multiplicity
from metag import pipeline as P
from metag.tools.preflight import audit_reactions
from metag.routing.planner import route_reaction


@pytest.mark.parametrize("smiles", ["[13CH3]CO", "[2H]OC", "[1H]OC"])
def test_mass_labels_cannot_enter_unlabelled_thermal_model(smiles):
    assert isotope_labels(smiles)
    with pytest.raises(ValueError, match="Isotope"):
        spin_multiplicity(smiles, 0)
    with pytest.raises(ValueError, match="Isotope"):
        pool_confs(smiles, 0, 1, 2)


def test_unlabelled_structures_keep_existing_domain():
    assert isotope_labels("CCO") == []
    assert spin_multiplicity("CCO", 0) == 1


def isotope_reaction():
    return {"species": {"a": [-1, 0, "[13CH3]CO"], "b": [1, 0, "[13CH3]CO"]}}


def test_scoring_and_preflight_refuse_isotopes_before_backend_or_cache(monkeypatch):
    monkeypatch.setattr(P._sc, "get", lambda *a, **k: pytest.fail("unsupported isotope reached cache"))
    result = P.score_reaction(None, isotope_reaction(), key="isotope", log=lambda *_: None)
    assert result["dG"] is None
    assert "isotope" in result["suspect"]
    assert result["routes"]["unsupported_isotopes"]["a"][0]["mass_number"] == 13
    assert audit_reactions({"isotope": isotope_reaction()})["reactions"]["isotope"]["status"] == "blocked"


def test_optional_radius_wrapper_cannot_bypass_isotope_guard(monkeypatch):
    old_flag = P._flag
    monkeypatch.setattr(P, "_flag", lambda name: True if name == "TRUNC_VALIDATE" else old_flag(name))
    assert P.score_reaction(None, isotope_reaction(), log=lambda *_: None)["dG"] is None


def test_direct_explicit_energy_refuses_isotope_cache(monkeypatch):
    monkeypatch.setattr(P._sc, "get", lambda *a, **k: pytest.fail("unsupported isotope cache used"))
    with pytest.raises(ValueError, match="Isotope"):
        P.explicit_G(None, 0, "[13CH4]", (), lambda *_: None, "label")


def test_disproven_reaction_correction_is_not_a_supported_production_switch(monkeypatch):
    monkeypatch.setenv("SMD_SOLV", "1")
    with pytest.raises(ValueError, match="inconsistent water"):
        route_reaction({"species": {"a": [-1, 0, "CCO"], "b": [1, 0, "CCO"]}})
