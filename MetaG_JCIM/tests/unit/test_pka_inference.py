"""Regression checks for pKa domain, provenance, and thermodynamic bookkeeping."""
import json
import math

import pytest
from rdkit import Chem

from metag.energetics.protonation import reaction_transform, site_free_energy
from metag.reactions import validate_reaction
from metag.routing import pka_transform as pka
from metag.routing.site_pka import UnsupportedSiteError, load_site_model, site_features


@pytest.mark.parametrize("smi,group", [
    ("C[S-]", "thiolate"), ("[O-]c1ccccc1", "phenolate"),
    ("O=C([O-])[O-]", "carbonate"), ("NP(=O)([O-])[O-]", "phosphoramidate"),
    ("CC(=O)OP(=O)([O-])[O-]", "acyl phosphate"),
])
def test_site_model_rejects_unsupported_chemistry(monkeypatch, smi, group):
    monkeypatch.setenv("PKA_MODEL", "site")
    with pytest.raises(UnsupportedSiteError, match=group):
        pka._neutralize(smi)
    # The table covers additional groups, except its quarantined P-N ladder.
    monkeypatch.setenv("PKA_MODEL", "table")
    if group == "phosphoramidate":
        with pytest.raises(UnsupportedSiteError, match="quarantined"):
            pka._neutralize(smi)
    else:
        neutral, values, charge = pka._neutralize(smi)
        assert neutral and values and charge == 0


def test_model_choice_and_missing_artifact_fail_explicitly(monkeypatch, tmp_path):
    monkeypatch.setenv("PKA_MODEL", "typo")
    with pytest.raises(ValueError, match="unknown PKA_MODEL"):
        pka.pka_model_config()
    monkeypatch.setenv("PKA_MODEL", "site")
    monkeypatch.setenv("PKA_SITE_MODEL_FILE", str(tmp_path / "missing.json"))
    with pytest.raises(ValueError, match="cannot load requested"):
        pka._neutralize("CC(=O)[O-]")


def test_model_content_change_updates_parameters_and_fingerprint(monkeypatch, tmp_path):
    params = dict(load_site_model()[0])
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"params": params}))
    monkeypatch.setenv("PKA_MODEL", "site")
    monkeypatch.setenv("PKA_SITE_MODEL_FILE", str(path))
    before = pka.pka_model_config()
    value = pka._neutralize("CC(=O)[O-]")[1][0]
    params["carb_plain"] += 0.5
    path.write_text(json.dumps({"params": params}))
    assert pka.pka_model_config()["site_pka_sha256"] != before["site_pka_sha256"]
    assert pka._neutralize("CC(=O)[O-]")[1][0] == pytest.approx(value + 0.5)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, True])
def test_model_invalid_parameter_rejected(monkeypatch, tmp_path, value):
    params = dict(load_site_model()[0])
    params["carb_plain"] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"params": params}))
    monkeypatch.setenv("PKA_SITE_MODEL_FILE", str(path))
    with pytest.raises(ValueError, match="carb_plain"):
        load_site_model()


def test_no_training_dependency_in_inference():
    assert site_features.__module__ == "metag.routing.site_pka"


@pytest.mark.parametrize("neutral,anion", [
    ("CCS", "CC[S-]"), ("Oc1ccccc1", "[O-]c1ccccc1"),
    ("CC(=O)O", "CC(=O)[O-]"), ("OP(=O)(O)O", "[O-]P(=O)([O-])[O-]"),
])
def test_acid_reference_and_transform_independent_of_input_charge(monkeypatch, neutral, anion):
    monkeypatch.setenv("PKA_MODEL", "table")
    left, right = pka._neutralize(neutral), pka._neutralize(anion)
    assert left[0] == right[0]
    assert sorted(left[1]) == sorted(right[1])
    assert left[2] == right[2] == 0


@pytest.mark.parametrize("smiles", ["CCO", "CC(=O)S", "COc1ccccc1", "CSSC"])
def test_weak_acid_extension_leaves_other_groups_alone(smiles):
    assert pka._canonicalize_maxanion(smiles) == smiles


def test_monoprotic_equal_population_limit_and_mirror():
    rt_ln10 = 5.7
    assert site_free_energy(7, "acid", 7, rt_ln10) == pytest.approx(-rt_ln10 * math.log10(2))
    assert site_free_energy(4, "acid", 7, rt_ln10) == site_free_energy(10, "base", 7, rt_ln10)
    # Population-dominance limit must remain finite far beyond 10**308.
    assert site_free_energy(-1000, "acid", 7, rt_ln10) == pytest.approx(-rt_ln10 * 1007)
    assert site_free_energy(1000, "acid", 7, rt_ln10) == 0


def test_reversal_scaling_and_spectator_cancellation():
    sites = [["react", 4, "acid", 0.5], ["prod", 9, "base", 2]]
    value = reaction_transform(sites, 7, 5.7)
    assert reaction_transform([["prod" if s == "react" else "react", p, k, m]
                               for s, p, k, m in sites], 7, 5.7) == -value
    assert reaction_transform([[s, p, k, 3*m] for s, p, k, m in sites], 7, 5.7) == pytest.approx(3*value)
    assert reaction_transform([["react", 4], ["prod", 4]], 7, 5.7) == 0


@pytest.mark.parametrize("site", [[], ["wrong", 4], ["react", 4, "wrong"],
    ["prod", float("nan")], ["react", 4, "acid", -1], ["react", 4, "acid", 0],
    ["react", 4, "acid", True], ["react", 4, "acid", 1, "extra"]])
def test_bad_user_terms_rejected_before_scoring(site):
    with pytest.raises(ValueError):
        validate_reaction({"species": {"ethanol": [-1, 0, "CCO"], "ether": [1, 0, "COC"]},
                           "pka_sites": [site]})
