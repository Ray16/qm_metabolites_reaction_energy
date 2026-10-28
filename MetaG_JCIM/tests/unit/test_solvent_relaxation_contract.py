"""Regression cases for destination-force consistency and incomplete ensembles."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from metag.energetics import solv_relax as S
from metag.energetics.sampling import species_diagnostics


def test_frozen_aqueous_conformer_keeps_solvent_force_while_another_relaxes(monkeypatch):
    atoms = [Atoms("H", positions=[[1, 0, 0]]), Atoms("H", positions=[[0, 0, 0]])]
    calls = []
    def force(symbols, positions, *args):
        calls.append(positions.copy())
        return np.array([[1., 0, 0]])
    monkeypatch.setattr(S, "solvation_forces", force)
    extra = S.make_extra_forces(0, 1, "alpb", workers=1)
    first = extra(atoms, np.array([False, False]))
    second = extra(atoms, np.array([True, False]))
    np.testing.assert_allclose(second[0], first[0])
    gas = np.array([[-1., 0, 0], [0, 0, 0]])
    np.testing.assert_allclose((gas + second)[0], 0)
    assert len(calls) == 3
    atoms[0].positions[0, 0] += 0.1
    extra(atoms, np.array([True, False]))
    assert len(calls) == 5  # changed geometry invalidates frozen-force reuse


@pytest.mark.parametrize("force", [None, np.array([[1., 2.]]), np.array([[float("nan"), 0, 0]])])
def test_failed_extra_force_is_recorded_and_never_cached(monkeypatch, force):
    monkeypatch.setattr(S, "solvation_forces", lambda *a: force)
    extra = S.make_extra_forces(0, 1, "alpb", workers=1)
    result = extra([Atoms("H", positions=[[0, 0, 0]])], [False])
    np.testing.assert_allclose(result, [[0, 0, 0]])
    assert extra.failed == {0}
    assert "gradient" in extra.failures[0]


@pytest.mark.parametrize("line", ["1.0 2.0", "nan 0 0", "unparseable", ""])
def test_malformed_xtb_gradient_is_not_a_valid_force(monkeypatch, tmp_path, line):
    xyz = tmp_path / "m.xyz"
    xyz.write_text("1\n\nH 0 0 0\n")
    def run(*args, **kwargs):
        (tmp_path / "gradient").write_text("$grad\ncycle 1\n0 0 0 h\n" + line + "\n$end\n")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(S.subprocess, "run", run)
    assert S._xtb_grad(str(tmp_path), str(xyz), 0, 0, []) is None


def test_cache_diagnostics_preserve_destination_scope_without_aliasing():
    metadata = {"solv_relax": {"n_failed": 1, "sampling_sigma_scope": "gas only"}}
    result = species_diagnostics(metadata, "cache")
    assert result["solv_relax"]["n_failed"] == 1
    result["solv_relax"]["n_failed"] = 9
    assert metadata["solv_relax"]["n_failed"] == 1


def test_unimplemented_solvent_gradient_fails_explicitly():
    with pytest.raises(ValueError, match="No implemented"):
        S.solvation_forces(["H"], [[0, 0, 0]], 0, model="cpcmx")
