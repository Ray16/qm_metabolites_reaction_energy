"""Regression tests for ensemble and speciation errors; no external QM calls."""
import math

import numpy as np
import pytest
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

from metag import pipeline as P
from metag.energetics.microstates import combine
from metag.routing import aldehyde_hydration as hydration
from metag.routing import pka_transform as pk


def methane():
    mol = Chem.AddHs(Chem.MolFromSmiles("C"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    atoms = Atoms([a.GetSymbol() for a in mol.GetAtoms()], mol.GetConformer().GetPositions())
    return mol, atoms


@pytest.fixture
def species_backend(monkeypatch):
    _, atoms = methane()
    monkeypatch.setattr(P._sc, "get", lambda *a, **k: None)
    writes = []
    monkeypatch.setattr(P._sc, "put", lambda *a, **k: writes.append((a, k)))
    monkeypatch.setattr(P, "pool_confs", lambda *a, **k: [atoms.copy()])
    monkeypatch.setattr(P, "batched_energies", lambda *a, **k: np.array([-100.0 / P.EV2KJ]))
    monkeypatch.setattr(P, "batched_fire", lambda pu, ats, **k:
                        (ats, np.full(len(ats), -100.0 / P.EV2KJ), np.ones(len(ats), dtype=bool)))
    monkeypatch.setattr(P, "dgsolv", lambda *a, **k: -10.0)
    monkeypatch.setattr(P, "same_connectivity", lambda *a: True)
    monkeypatch.setattr(P, "uma_gibbs_corr", lambda *a, **k:
                        (5.0, {"n_imag": 0, "max_imag_cm": 0.0, "_imag_vecs": []}))
    monkeypatch.setattr(P, "_SOLV_RELAX", False)
    monkeypatch.setattr(P, "_DEDUP", True)
    monkeypatch.setattr(P, "_ACID_HB_FILTER", False)
    monkeypatch.setattr(P, "SOLV_ALSO", [])
    monkeypatch.setattr(P, "CONV_MAX", 3)
    return writes


@pytest.mark.parametrize("thermal_ensemble", [False, True])
def test_solvent_relaxation_survives_thermal_processing(monkeypatch, species_backend, thermal_ensemble):
    monkeypatch.setattr(P, "_THERMAL_ENSEMBLE", thermal_ensemble)
    baseline, _ = P.implicit_G(None, 0, "C", (), 1, 1, lambda *_: None, "methane")
    assert baseline == pytest.approx(-105.0)  # electronic -100, solvation -10, RRHO +5
    monkeypatch.setattr(P, "_SOLV_RELAX", True)

    def aqueous(*args):
        assert args[-1] == pytest.approx([5.0])
        return [-125.0], {}, {"n_relaxed": 1, "n_unconverged": 0, "n_rearranged": 0, "n_failed": 0}

    monkeypatch.setattr(P, "_solvent_relaxed_ensemble", aqueous)
    result, _ = P.implicit_G(None, 0, "C", (), 1, 1, lambda *_: None, "methane")
    assert result == pytest.approx(-125.0)  # includes RRHO; must neither overwrite nor add it twice
    assert species_backend[-1][0][4] == pytest.approx(-125.0)


def test_no_aqueous_minimum_fails_without_caching_vertical_result(monkeypatch, species_backend):
    monkeypatch.setattr(P, "_SOLV_RELAX", True)
    monkeypatch.setattr(P, "_solvent_relaxed_ensemble", lambda *a: None)
    warnings = []
    assert P.implicit_G(None, 0, "C", (), 1, 1, lambda *_: None, "methane", warnings) == (None, None)
    assert "no converged solution-phase minimum" in warnings[-1]
    assert species_backend == []


def test_partial_aqueous_ensemble_is_not_cached(monkeypatch, species_backend):
    monkeypatch.setattr(P, "_SOLV_RELAX", True)
    monkeypatch.setattr(P, "_solvent_relaxed_ensemble", lambda *a:
                        ([-125.0], {}, {"n_relaxed": 1, "n_unconverged": 1,
                                       "n_rearranged": 0, "n_failed": 0}))
    warnings = []
    result, _ = P.implicit_G(None, 0, "C", (), 1, 1, lambda *_: None, "methane", warnings)
    assert result == pytest.approx(-125.0)
    assert any("incomplete solution-phase ensemble" in warning for warning in warnings)
    assert species_backend == []


def test_unconverged_aqueous_structures_cannot_dominate(monkeypatch):
    from metag.energetics import solv_relax
    mol, atoms = methane()
    minima = P.UniqueMinima(template=mol, e_tol=-1)
    minima.add(atoms, -100.0, -110.0)
    minima.add(atoms, -90.0, -100.0)
    extra = lambda *a: None
    extra.failed = set()
    monkeypatch.setattr(solv_relax, "make_extra_forces", lambda *a, **k: extra)
    monkeypatch.setattr(P, "batched_fire", lambda pu, ats, **k:
                        (ats, np.array([-1000.0, -90.0]) / P.EV2KJ, np.array([False, True])))
    monkeypatch.setattr(P, "dgsolv", lambda *a, **k: -10.0)
    monkeypatch.setattr(P, "same_connectivity", lambda *a: True)
    monkeypatch.setattr(P, "SOLV_ALSO", [])
    values, _, info = P._solvent_relaxed_ensemble(
        None, minima, 0, 1, None, mol, "test", lambda *_: None, [5.0, 7.0])
    assert values == pytest.approx([-93.0])  # -90 -10 +7; reject spuriously favorable -1000
    assert info["n_unconverged"] == 1


@pytest.mark.parametrize("gap", [-3000.0, 3000.0])
def test_hydration_partition_is_stable_for_large_state_gaps(gap):
    result = hydration.mixture_G(-1e6, -1e6 + gap, 0.0)
    assert math.isfinite(result)
    assert result == pytest.approx(-1e6 + min(gap, 0))
    assert result == pytest.approx(hydration.mixture_G_states(-1e6, [(1, -1e6 + gap)], 0))


def test_duplicate_microstate_labels_are_rejected():
    with pytest.raises(ValueError, match="duplicate microstate"):
        combine([{"name": "a", "G": 0.0}, {"name": "a", "G": 10.0}])


@pytest.mark.parametrize("neutral,charged", [
    ("N", "[NH4+]"), ("CN", "C[NH3+]"), ("CNC", "C[NH2+]C"),
    ("CN(C)C", "C[NH+](C)C"), ("NCC(=O)O", "[NH3+]CC(=O)[O-]"),
    ("CN(C)CC(=O)O", "C[NH+](C)CC(=O)[O-]"),
])
def test_base_transform_independent_of_input_protonation_on_neutral_route(neutral, charged):
    assert pk._neutralize_v2(neutral) == pk._neutralize_v2(charged)


@pytest.mark.parametrize("smiles", ["Nc1ccccc1", "CC(=O)N", "C=N", "NC(N)=N", "NO", "[NH2]", "C[N+](C)(C)C"])
def test_neutral_base_extension_excludes_other_nitrogen_chemistry(smiles):
    assert pk._neutral_base_pkas(Chem.MolFromSmiles(smiles)) == []


def test_fractional_ammonia_transform_cancels_and_reverses():
    species = {"ammonia": [-0.5, 0, "N"], "ammonium": [0.5, 1, "[NH4+]"]}
    routed, sites, nh = pk.build_ph0_reaction(species, n_Hplus=-0.5, force_base=True)
    assert nh == 0
    assert routed["ammonia"][2] == routed["ammonium"][2] == "N"
    assert sites == [["react", pk.AMMONIA_PKA, "base", 0.5], ["prod", pk.AMMONIA_PKA, "base", 0.5]]


@pytest.mark.parametrize("constant", ["AMMONIA_PKA", "AAA_AMINE_PKA", "PRIMARY_AMINE_PKA", "IMIDAZOLE_PKA", "GUANIDINIUM_PKA"])
def test_base_constants_participate_in_calibration_fingerprint(monkeypatch, constant):
    before = P.effective_config()["pka_constants"]
    monkeypatch.setattr(pk, constant, getattr(pk, constant) + 0.1)
    assert P.effective_config()["pka_constants"] != before


@pytest.fixture
def tautomer_pilot():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "analysis/glycosyl_tautomers/pilot.py"
    spec = importlib.util.spec_from_file_location("tautomer_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tautomer_cap_always_preserves_input(tautomer_pilot):
    # The archived guanine run omitted its input and silently used an alternative.
    states, reference = tautomer_pilot.enumerate_tautomers(tautomer_pilot.BASES["guanine"], cap=1)
    assert states == [reference]


def test_tautomer_pilot_never_substitutes_failed_input(tautomer_pilot):
    result = tautomer_pilot.summarize({"input": None, "alternative": -100.0}, "input")
    assert result == {"status": "input_state_failed", "lowering_kJ": None}


def test_tautomer_pilot_labels_partial_state_set(tautomer_pilot):
    result = tautomer_pilot.summarize({"input": -100.0, "alternative": None}, "input")
    assert result["status"] == "partial_selected_set"
    assert result["failed_states"] == ["alternative"]
