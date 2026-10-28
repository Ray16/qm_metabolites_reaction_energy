"""Prevent stale energies and falsely identical estimator configurations (CPU only)."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from metag.energetics import provenance as pr
from metag.energetics import species_cache as cache


def trusted_model(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"first weights")
    identity = pr.model_identity(checkpoint, atom_refs={"H": 1.2}, form_elem_refs=None,
                                 inference_settings={"precision": "float32"})
    return SimpleNamespace(metag_provenance=identity)


@pytest.fixture
def producer_env(tmp_path, monkeypatch):
    xtb = tmp_path / "xtb"
    xtb.write_text("#!/bin/sh\nexit 0\n")
    xtb.chmod(0o755)
    monkeypatch.setattr(pr, "software_versions", lambda: {"fairchem-core": "test-v1"})
    monkeypatch.setattr(pr, "species_source_identity", lambda: "source-v1")
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cache, "_ENABLED", True)
    return xtb


@pytest.mark.parametrize("change", ["software", "xtb", "model", "references", "inference", "source"])
def test_changed_producer_cannot_reuse_old_energy(tmp_path, monkeypatch, producer_env, change):
    predictor = trusted_model(tmp_path)
    old = {"physics": "same", "producer": pr.producer_identity(predictor, producer_env)}
    assert old["producer"]["cacheable"]
    cache.put("CCO", 0, "implicit", old, -123, 0.5)
    assert cache.get("OCC", 0, "implicit", old) == (-123, 0.5)
    if change == "software":
        monkeypatch.setattr(pr, "software_versions", lambda: {"fairchem-core": "test-v2"})
    elif change == "xtb":
        producer_env.write_text("#!/bin/sh\nexit 1\n")
    elif change == "source":
        monkeypatch.setattr(pr, "species_source_identity", lambda: "source-v2")
    else:
        field = {"model": "checkpoint_sha256", "references": "references_sha256",
                 "inference": "inference_sha256"}[change]
        predictor.metag_provenance[field] = "f" * 64
    new = {"physics": "same", "producer": pr.producer_identity(predictor, producer_env)}
    assert cache.get("CCO", 0, "implicit", new) is None
    assert cache.get("CCO", 0, "implicit", old) == (-123, 0.5)


def test_unknown_producer_never_persists_or_reads_energies(tmp_path, producer_env):
    settings = {"producer": pr.producer_identity(SimpleNamespace(metag_model="named"), producer_env)}
    assert settings["producer"]["cacheable"] is False
    cache.put("CCO", 0, "implicit", settings, -123, 0.5)
    assert cache.get("CCO", 0, "implicit", settings) is None
    assert not (tmp_path / "cache").exists()


def test_missing_executable_disables_persistent_cache(tmp_path, producer_env):
    assert not pr.producer_identity(trusted_model(tmp_path), tmp_path / "missing")["cacheable"]


def test_additional_solvent_binary_is_part_of_producer_identity(tmp_path, producer_env):
    predictor = trusted_model(tmp_path)
    cpcmx = tmp_path/'xtbcpx'
    cpcmx.write_text('#!/bin/sh\nexit 0\n'); cpcmx.chmod(0o755)
    first = pr.producer_identity(predictor, producer_env, additional_executables={'cpcmx':cpcmx})
    assert first['cacheable']
    cpcmx.write_text('#!/bin/sh\nexit 1\n')
    second = pr.producer_identity(predictor, producer_env, additional_executables={'cpcmx':cpcmx})
    assert first != second
    cpcmx.unlink()
    assert not pr.producer_identity(predictor, producer_env,
                                    additional_executables={'cpcmx':cpcmx})['cacheable']


def test_pipeline_records_active_auxiliary_solvent_backend(monkeypatch):
    from metag import pipeline as p
    monkeypatch.setattr(p, 'SOLV_MODEL', 'alpb')
    monkeypatch.setattr(p, 'SOLV_ALSO', ['cpcmx'])
    monkeypatch.setattr(p, 'producer_identity', lambda pu, xtb, **kwargs: kwargs)
    assert p._producer_identity(None)['additional_executables'] == {'xtbcpx':p.thermal_mod.XTBCPX}


def test_file_identity_changes_when_replaced_and_survives_relocation(tmp_path):
    path = tmp_path / "model"
    path.write_bytes(b"weights-1")
    before = pr.file_hash(path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_bytes(path.read_bytes())
    assert pr.file_hash(elsewhere) == before
    path.write_bytes(b"weights-2")
    assert pr.file_hash(path) != before


def test_fairchem_reference_containers_have_content_based_identity():
    from omegaconf import OmegaConf
    plain = {'refs': {'omol': [1.0, 2.0]}, 'metadata': {'description': 'atomic references'}}
    configured = OmegaConf.create(plain)
    assert pr.content_hash(configured) == pr.content_hash(plain)
    configured.refs.omol[0] = 3.
    assert pr.content_hash(configured) != pr.content_hash(plain)


def test_reference_only_edits_reuse_raw_energies_but_change_estimator_identity(tmp_path):
    from metag.validation.provenance import source_tree_fingerprint
    root = tmp_path/'metag'
    (root/'energetics').mkdir(parents=True)
    (root/'validation').mkdir()
    for name in ('pipeline.py', 'chem.py', 'symmetry.py', 'water_count.py', 'config.py', 'profile.py', 'conditions.py'):
        (root/name).write_text('# numerical dependency\n')
    (root/'energetics/thermal.py').write_text('temperature = 298.15\n')
    reference = root/'validation/chemical_reference.py'
    reference.write_text('policy = 1\n')
    species_before = pr.species_source_fingerprint(root)
    estimator_before = source_tree_fingerprint(root)
    reference.write_text('policy = 2\n')
    assert pr.species_source_fingerprint(root) == species_before
    assert source_tree_fingerprint(root) != estimator_before
    (root/'conditions.py').write_text('T = 310.15\n')
    assert pr.species_source_fingerprint(root) != species_before
    (root/'conditions.py').write_text('# numerical dependency\n')
    assert pr.species_source_fingerprint(root) == species_before
    (root/'energetics/thermal.py').write_text('temperature = 310.15\n')
    assert pr.species_source_fingerprint(root) != species_before


def test_per_call_policy_and_auto_radius_have_distinct_identities(monkeypatch):
    from metag import pipeline as p
    from metag import uncertainty as uq
    monkeypatch.delenv("TRUNC_RADIUS", raising=False)
    automatic = p.effective_config()
    explicit_two = p.effective_config(trunc_radius=2)
    explicit_four = p.effective_config(trunc_radius=4)
    full = p.effective_config(allow_truncate=False)
    assert automatic != explicit_two != explicit_four
    assert automatic != full
    assert automatic["routing_policy"]["trunc_radius_explicit"] is False
    monkeypatch.setenv("TRUNC_RADIUS", "2")
    assert p.effective_config() == explicit_two
    monkeypatch.setattr(uq, "CALIB_CONFIG", deepcopy(automatic))
    for different in (explicit_two, explicit_four, full):
        assert uq.calibration_mismatch(different)
        info = uq.prediction_interval("fumarate hydratase", ["O"], 0, config=different)[3]
        assert not info["coverage_calibrated"]


@pytest.mark.parametrize("invalid", [False, True])
def test_result_retains_call_policy_even_on_failed_estimate(monkeypatch, invalid):
    from metag import pipeline as p
    monkeypatch.setattr(p, "_flag", lambda name: name == "STD_STATE_1M")
    monkeypatch.setattr(p, "implicit_G", lambda *a, **kw: (-50, 0.2))
    reaction = {"species": {"ethanol": [-1, 0, "CCO"], "ether": [1, 0, "COC"]}}
    if invalid:
        reaction["n_Hplus"] = 1
    result = p.score_reaction(None, reaction, allow_truncate=False, trunc_radius=4, log=lambda *_: None)
    assert result["config"] == p.effective_config(allow_truncate=False, trunc_radius=4)
    assert (result["dG"] is None) == invalid


def test_loader_hashes_the_checkpoint_and_references_it_actually_loads(tmp_path, monkeypatch):
    from metag.energetics import uma
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test model")
    monkeypatch.setattr(uma, "_ensure_registered", lambda _: None)
    backend = uma.pretrained_mlip
    monkeypatch.setattr(backend, "pretrained_checkpoint_path_from_name", lambda _: checkpoint)
    monkeypatch.setattr(backend, "get_reference_energies", lambda *a: {"H": 1.2})
    monkeypatch.setitem(backend._MODEL_CKPTS.checkpoints, "test", SimpleNamespace(form_elem_refs=None))
    loaded = []
    def load(path, **kwargs):
        loaded.append((path, kwargs))
        return SimpleNamespace(inference_settings={"precision": "float32"})
    monkeypatch.setattr(backend, "load_predict_unit", load)
    model = uma.load_uma("test", profile="legacy")
    assert loaded == [(checkpoint, {"device": uma.DEV, "atom_refs": {"H": 1.2}, "form_elem_refs": None})]
    assert model.metag_model == "test"
    assert model.metag_provenance["checkpoint_sha256"] == pr.file_hash(checkpoint)
    # Exercise actual fairchem datatypes without loading a model or touching CUDA.
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    assert len(pr.model_identity(checkpoint, atom_refs=None, form_elem_refs=None,
                                 inference_settings=InferenceSettings())["inference_sha256"]) == 64


def test_equivalent_smiles_produce_identical_fresh_sampling_inputs(monkeypatch):
    import numpy as np
    from metag import pipeline as p
    from metag.energetics.conformers import pool_confs
    class SamplingObserved(Exception):
        pass
    observed = []
    def sample(smiles, q, seed, pool, spin):
        atoms = pool_confs(smiles, q, seed, pool, spin=spin)
        observed.append((smiles, [(a.get_chemical_symbols(), a.positions.copy()) for a in atoms]))
        raise SamplingObserved
    monkeypatch.setattr(p._sc, 'get', lambda *a, **kw: None)
    monkeypatch.setattr(p, 'sampling_budget', lambda smi: (1, 1, 3))
    monkeypatch.setattr(p, 'pool_confs', sample)
    for smi in ('CCO', 'OCC'):
        with pytest.raises(SamplingObserved):
            p.implicit_G(None, 0, smi, [1], 1, 3, lambda *_: None, 'ethanol')
    assert observed[0][0] == observed[1][0]
    assert len(observed[0][1]) == len(observed[1][1])
    for (sa, xa), (sb, xb) in zip(observed[0][1], observed[1][1]):
        assert sa == sb
        np.testing.assert_array_equal(xa, xb)


def test_explicit_geometry_uses_canonical_identity_without_merging_stereo(monkeypatch):
    from metag import pipeline as p
    class GeometryObserved(Exception):
        pass
    observed = []
    def geometry(pu, charge, smiles):
        observed.append(smiles)
        raise GeometryObserved
    monkeypatch.setattr(p._sc, 'get', lambda *a, **kw: None)
    monkeypatch.setattr(p, 'bare_geom', geometry)
    for smi in ('CCO', 'OCC', 'C[C@H](O)C(=O)O', 'C[C@@H](O)C(=O)O'):
        with pytest.raises(GeometryObserved):
            p.explicit_G(None, 0, smi, [1], lambda *_: None, 'test')
    assert observed[0] == observed[1]
    assert observed[2] != observed[3]


def test_loaded_module_versions_are_distinct_from_distribution_metadata(monkeypatch):
    import sys
    from types import SimpleNamespace
    from metag.energetics import provenance as pr
    monkeypatch.setitem(sys.modules,'numpy',SimpleNamespace(__version__='actual-numerical-build'))
    monkeypatch.setattr(pr,'software_versions',lambda:{'numpy':'metadata-build'})
    identity=pr.runtime_identity('unavailable-test-xtb')
    assert identity['software']['numpy']=='metadata-build'
    assert identity['imported_module_versions']['numpy']=='actual-numerical-build'
