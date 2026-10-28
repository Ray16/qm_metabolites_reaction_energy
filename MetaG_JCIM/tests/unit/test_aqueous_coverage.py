"""Incomplete state enumeration must not masquerade as an equilibrium estimate."""
from unittest.mock import Mock

import pytest

from metag.energetics.aqueous_states import hydrate_all_sites
from metag.routing.aldehyde_hydration import n_hydration_sites


@pytest.mark.parametrize("smiles,count,computed", [
    ("O=CCC(=O)CC=O", 3, 5),  # seven subsets, five distinct molecular hydrates
    ("O=CCC(=O)CC(=O)CC=O", 4, 0),
])
def test_hydration_limit_requires_complete_ensemble(smiles, count, computed):
    assert n_hydration_sites(smiles) == count
    energies, sigmas = {"x": -20.0}, {"x": 0.1}
    routes = {"warnings": []}
    implicit = Mock(return_value=(-25.0, 0.1))
    water = Mock(return_value=-10.0)
    hydrate_all_sites(None, {}, "x", 0, smiles, energies, sigmas, 0.0,
                      1, 1, 1, lambda _: None, routes, implicit=implicit,
                      water_reference=water, acid_transform=lambda _: 0.0,
                      calibrate=False)
    assert implicit.call_count == computed
    assert bool(routes.get("incomplete_hydration")) == (count > 3)
    if count > 3:
        assert energies["x"] == -20.0
        water.assert_not_called()


def test_hydration_limit_fails_closed_at_reaction_level(monkeypatch):
    from metag import pipeline as p
    monkeypatch.setattr(p, "_flag", lambda n: n in {
        "STD_STATE_1M", "ALDEHYDE_HYDRATION", "CARBONYL_HYDRATION_ALL"})
    monkeypatch.setattr(p, "implicit_G", lambda *a, **kw: (-50.0, 0.1))
    monkeypatch.setattr(p, "water_ref_G", lambda *a, **kw: -10.0)
    reaction = {"species": {
        "four-carbonyl": [-1, 0, "O=CCC(=O)CC(=O)CC=O"],
        "hydrogen": [-1, 0, "[H][H]"],
        "three-carbonyl": [1, 0, "OCCC(=O)CC(=O)CC=O"],
    }}
    result = p.score_reaction(None, reaction, log=lambda _: None)
    assert result["dG"] is None
    assert result["routes"]["input_balanced"]
    assert "incomplete hydration" in result["suspect"]
    assert result["ci_info"]["coverage_calibrated"] is False


_MULTI_RING = "O=C(CC(C(N)=O)C(=O)NC)CC(C(N)=O)C(=O)NCC"


def test_ring_enumeration_exposes_cap_without_silent_partial_list():
    from metag.routing.ring_chain import (
        IncompleteRingChainError, enumerate_ring_forms, ring_forms)
    enumeration = enumerate_ring_forms(_MULTI_RING)
    assert enumeration.total_states == 8
    assert enumeration.state_limit == 4
    assert len(enumeration.forms) == 4
    assert not enumeration.complete
    with pytest.raises(IncompleteRingChainError, match="8 states, limit 4"):
        ring_forms(_MULTI_RING)


@pytest.mark.parametrize("smiles,computed", [
    ("NC(=O)CCC(=O)C(=O)O", 2), (_MULTI_RING, 0),
])
def test_ring_coverage_checked_before_state_qm(smiles, computed):
    from metag.energetics.aqueous_states import fold_ring_chain
    energies, sigmas = {"x": -20.0}, {"x": 0.1}
    routes = {"warnings": []}
    implicit = Mock(return_value=(-25.0, 0.1))
    fold_ring_chain(None, {}, "x", 0, smiles, energies, sigmas,
                    1, 1, 1, lambda _: None, routes, implicit=implicit,
                    acid_transform=lambda _: 0.0, standard_state=0.0,
                    temperature=298.15)
    assert implicit.call_count == computed
    assert bool(routes.get("incomplete_ring_chain")) == (computed == 0)
    if not computed:
        assert energies["x"] == -20.0
    else:
        assert energies["x"] < -25.0


def test_ring_limit_fails_closed_at_reaction_level(monkeypatch):
    from metag import pipeline as p
    monkeypatch.setattr(p, "_flag", lambda n: n in {"STD_STATE_1M", "RING_CHAIN"})
    calls = []
    def implicit(pu, q, smi, seeds, keep, pool, log, name, warnings=None, diagnostics=None):
        calls.append(name)
        return -50.0, 0.1
    monkeypatch.setattr(p, "implicit_G", implicit)
    # Declare the fixture's stereocenters so this test reaches the ring-state
    # coverage gate rather than the earlier chemical-identity guard.
    ketone = "O=C(C[C@H](C(N)=O)C(=O)NC)C[C@H](C(N)=O)C(=O)NCC"
    reaction = {"species": {
        "ketone": [-1, 0, ketone], "hydrogen": [-1, 0, "[H][H]"],
        "alcohol": [1, 0, ketone.replace("O=C", "O[C@H]", 1)],
    }}
    result = p.score_reaction(None, reaction, log=lambda _: None)
    assert result["dG"] is None
    assert result["routes"]["input_balanced"]
    assert "incomplete ring-chain" in result["suspect"]
    assert result["ci_info"]["coverage_calibrated"] is False
    assert not any("(ring" in name for name in calls)
