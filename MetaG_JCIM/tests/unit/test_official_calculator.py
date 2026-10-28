"""Actual mode routing and endpoint correctness for the default calculator."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import random
import torch
from ase import Atoms

from metag.config import uma_runtime_profile
from metag.energetics import uma
from metag.energetics.official_uma import OfficialUMACalculator, trace_backend
from metag.energetics.provenance import file_hash


class Unit:
    def __init__(self, nitro=False, disable=False):
        self.inference_settings = SimpleNamespace(nitro=nitro, tf32=nitro,
                                                  merge_mole=nitro, compile=nitro)
        self.calls = 0
        self.disable = disable

    def predict(self, batch):
        self.calls += 1
        if self.disable:
            self.inference_settings.nitro = False
        return {"unit": self, "call": self.calls}


def calculator(*, eligible=True, disable=False):
    # Exercise routing without touching CUDA or loading any checkpoint.
    value = object.__new__(OfficialUMACalculator)
    value._reference = Unit()
    value._last_unit = value._reference
    value._nitro = None
    value._nitro_key = None
    value._counts = Counter()
    value._fallbacks = Counter()
    value._nitro_available = eligible
    value.metag_runtime_profile = {"requested_calculator": "nitro"}
    loaded = []

    def load(settings):
        unit = Unit(nitro=True, disable=disable)
        loaded.append(unit)
        return unit
    value._load = load
    return value, loaded


def batch(n=1, charge=0, spin=1, numbers=(8, 1, 1)):
    return SimpleNamespace(natoms=torch.tensor([len(numbers)] * n),
                           atomic_numbers=torch.tensor(list(numbers) * n),
                           charge=torch.tensor([charge] * n), spin=torch.tensor([spin] * n))


def test_default_is_nitro_and_explicit_profiles_are_strict(monkeypatch):
    monkeypatch.delenv("UMA_RUNTIME_PROFILE", raising=False)
    assert uma_runtime_profile() == "nitro"
    monkeypatch.setenv("UMA_RUNTIME_PROFILE", "official_fp32")
    assert uma_runtime_profile() == "official_fp32"
    assert uma_runtime_profile("legacy") == "legacy"
    for value in ["unknown", "", True, 5]:
        with pytest.raises(ValueError):
            uma_runtime_profile(value)


def test_v100_and_batches_use_fp32_without_nitro_loading():
    value, loaded = calculator(eligible=False)
    assert value.predict(batch())["unit"] is value._reference
    assert not loaded and not value.inference_settings.nitro
    value, loaded = calculator()
    assert value.predict(batch(n=2))["unit"] is value._reference
    assert not loaded and value._fallbacks["multiple_systems"] == 1


def test_charge_spin_and_composition_changes_create_fresh_nitro_units():
    value, loaded = calculator()
    first = value.predict(batch())["unit"]
    assert value.predict(batch())["unit"] is first
    value.predict(batch(charge=1))
    value.predict(batch(charge=1, spin=3))
    value.predict(batch(numbers=(6, 1, 1)))
    assert len(loaded) == 4 and value._counts["actual_nitro_calls"] == 5
    assert value.predict_precise(batch())["unit"] is value._reference
    assert not value.inference_settings.nitro


def test_vendor_reduced_precision_fallback_is_discarded():
    value, loaded = calculator(disable=True)
    assert value.predict(batch())["unit"] is value._reference
    assert value.predict(batch())["unit"] is value._reference
    assert len(loaded) == 1 and loaded[0].calls == 1
    assert value._counts["actual_nitro_calls"] == 0
    assert value._fallbacks["vendor_disabled_nitro"] == 2


def test_trace_is_per_invocation_and_does_not_change_scientific_config():
    value, _ = calculator()

    @trace_backend
    def score(pu):
        pu.predict(batch())
        pu.predict_precise(batch())
        return {"config": {"source": "fixed"}}

    for _ in range(2):
        result = score(value)
        assert result["config"] == {"source": "fixed"}
        assert result["backend_execution"]["counts"]["actual_nitro_calls"] == 1
        assert result["backend_execution"]["counts"]["official_fp32_calls"] == 1


def test_approximate_false_convergence_requires_fp32_refinement(monkeypatch):
    monkeypatch.setattr(uma, "DEV", "cpu")
    value, _ = calculator()
    exact_positions = []

    def approximate(pu, atoms, chunk):
        pu.record("actual_nitro_calls")
        return torch.tensor([-99.]), torch.zeros((1, 3)), torch.zeros(1, dtype=torch.long)

    def precise(pu, atoms):
        p = atoms[0].positions.copy()
        exact_positions.append(p)
        return (torch.tensor([.5 * np.square(p).sum()], dtype=torch.float64),
                torch.tensor(-p, dtype=torch.float32), torch.zeros(1, dtype=torch.long))

    monkeypatch.setattr(uma, "_predict_chunked", approximate)
    monkeypatch.setattr(uma, "_predict", precise)
    atom = Atoms("H", positions=[[1., 0, 0]])
    relaxed, energy, done = uma.batched_fire(value, [atom], steps=1, fmax=.999,
                                             return_converged=True)
    assert done.tolist() == [True]
    assert relaxed[0].positions[0, 0] < 1.
    assert energy[0] == pytest.approx(.5 * np.square(relaxed[0].positions).sum())
    assert value._counts["fp32_endpoint_refinements"] == 1
    # One allowed move was used; the extra work is evaluation, not a hidden
    # extension of the optimization budget.
    changed = [x for x in exact_positions if x[0, 0] < 1.]
    assert len(changed) == 1


@pytest.mark.parametrize("fail", [False, True])
def test_secondary_sdk_initialization_preserves_sampling_rng(tmp_path, fail):
    value = object.__new__(OfficialUMACalculator)
    value._checkpoint = tmp_path / "weights.pt"
    value._checkpoint.write_bytes(b"immutable weights")
    value._checkpoint_hash = file_hash(value._checkpoint)
    value._resolved = SimpleNamespace(atom_refs={"H": 1.}, form_elem_refs=None)
    value._device = "cpu"

    def seed_and_load(path, **kwargs):
        assert kwargs["seed"] == 41
        random.seed(41)
        np.random.seed(41)
        torch.manual_seed(41)
        if fail:
            raise RuntimeError("initialization failed")
        return "loaded"

    value._sdk = SimpleNamespace(load_predict_unit=seed_and_load)
    random.seed(17)
    np.random.seed(23)
    torch.manual_seed(29)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    if fail:
        with pytest.raises(RuntimeError, match="initialization failed"):
            value._load("nitro")
    else:
        assert value._load("nitro") == "loaded"
    assert random.getstate() == before[0]
    after = np.random.get_state()
    assert after[0] == before[1][0] and after[2:] == before[1][2:]
    np.testing.assert_array_equal(after[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])
