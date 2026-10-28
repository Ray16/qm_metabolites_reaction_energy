"""Architectural contracts: routing and CLI parsing must not load QM libraries."""
import json
import os
from pathlib import Path
import subprocess
import sys

from metag.cli import run_cli


def test_routing_and_cli_help_work_when_qm_imports_are_forbidden():
    code = '''
import importlib.abc
import sys
class NoQM(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'fairchem', 'ase'}:
            raise AssertionError('QM dependency imported: ' + fullname)
sys.meta_path.insert(0, NoQM())
from metag.assembly import assemble_reaction
from metag.energetics.sampling import sampling_budget
from metag.energetics.thermal_ensemble import ThermalTrack
from metag.energetics.aqueous_states import hydrate_all_sites, fold_ring_chain
from metag.routing.planner import route_reaction
reaction = {'species': {'ethanol': [-1, 0, 'CCO'], 'ether': [1, 0, 'COC']}}
rx, routes, truncated = route_reaction(reaction, log=lambda *_: None)
assert routes['input_balanced'] and not routes['errors']
assert sorted(map(tuple, rx['species'].values())) == sorted(map(tuple, reaction['species'].values()))
from metag.tools.preflight import audit_reactions
assert audit_reactions({'test': reaction})['n_reactions'] == 1
from metag.cli import main
try:
    main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError('help did not exit')
assert 'metag.pipeline' not in sys.modules
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[2] / "src"))
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--only" in result.stdout


def test_cli_checkpoints_failure_before_scoring_next_reaction(tmp_path):
    reactions = {"first": {}, "second": {}}
    output = tmp_path / "unified_pipeline_all.json"
    loaded = []

    def load(name):
        loaded.append(name)
        return object()

    def score(model, key, seeds, keep, pool, log):
        assert model is not None
        assert seeds == [3, 5] and keep == 2 and pool == 8
        if key == "first":
            raise RuntimeError("backend failure")
        saved = json.loads(output.read_text())
        assert len(saved) == 1 and saved[0]["reaction"] == "first"
        assert "backend failure" in saved[0]["suspect"]
        return {"reaction": key, "dG": 1.0, "explicit": False, "exp": [], "err": []}

    def failed(key, rx, routes, reason, log):
        return {"reaction": key, "dG": None, "suspect": reason}

    run_cli(reactions=reactions, model_name="model", load_model=load, score=score,
            no_estimate=failed, out=tmp_path, argv=["--seeds", "3,5", "--keep", "2", "--pool", "8"])
    assert loaded == ["model"]
    assert len(json.loads(output.read_text())) == 2


def test_assembly_preserves_stoichiometry_and_separates_energy_terms():
    import pytest
    from metag.assembly import assemble_reaction
    species = {"a": [-2, 0, "C"], "b": [1, 0, "CC"]}
    result = assemble_reaction(species, {"a": -100, "b": -180}, {"a": 3, "b": 8},
                               n_hplus=0.5, proton_energy=-10, pka_sites=[], ph=7, rt_ln10=5.7)
    assert (result.species_sum, result.proton, result.protonation, result.total) == (20, -5, 0, 15)
    assert result.sampling_sigma == 10
    reverse = {n: [-c, q, s] for n, (c, q, s) in species.items()}
    reversed_result = assemble_reaction(reverse, {"a": -100, "b": -180}, {"a": 3, "b": 8},
                                        n_hplus=-0.5, proton_energy=-10, pka_sites=[], ph=7, rt_ln10=5.7)
    assert reversed_result.total == -result.total
    assert reversed_result.sampling_sigma == pytest.approx(result.sampling_sigma)
