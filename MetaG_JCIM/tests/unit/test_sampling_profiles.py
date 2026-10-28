"""Search presets cannot silently change a loaded estimator or erase user settings."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from metag.sampling_profiles import sampling_environment, apply_sampling_profile


def test_current_preserves_custom_scientific_and_resource_settings():
    original = {"CONV_TOL": "1.3", "QRRHO": "1", "OMP_NUM_THREADS": "2"}
    result = sampling_environment("current", original)
    assert result == original and result is not original


def test_thorough_changes_only_search_effort():
    original = {"QRRHO": "1", "SOLV_MODEL": "alpb", "CONV_TOL": "2.5", "OMP_NUM_THREADS": "2"}
    result = sampling_environment("thorough", original)
    assert original["CONV_TOL"] == "2.5"
    for name in ("QRRHO", "SOLV_MODEL", "OMP_NUM_THREADS"):
        assert result[name] == original[name]


def test_loaded_pipeline_cannot_receive_an_ineffective_profile(monkeypatch):
    monkeypatch.setitem(sys.modules, "metag.pipeline", object())
    monkeypatch.setenv("CONV_TOL", "2.5")
    with pytest.raises(RuntimeError, match="before importing"):
        apply_sampling_profile("thorough")
    assert os.environ["CONV_TOL"] == "2.5"


def test_profile_reaches_actual_sampler_and_distinguishes_cache_settings():
    command = '''import json
from metag.sampling_profiles import apply_sampling_profile
apply_sampling_profile('thorough')
from metag import pipeline as p
print(json.dumps({'tol':p.CONV_TOL,'hits':p.CONV_HITS,'cap':p.CONV_MAX,'settings':p._IMPLICIT_SETTINGS}))
'''
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(Path(__file__).parents[2] / 'src')
    environment.update(OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", command], env=environment, text=True, capture_output=True, check=True)
    record = json.loads(result.stdout.splitlines()[-1])
    assert record['tol'] == .5 and record['hits'] == 3 and record['cap'] == 12
    assert record['settings']['sample_scale'] == 2
    assert record['settings']['keep_scale'] == 2
