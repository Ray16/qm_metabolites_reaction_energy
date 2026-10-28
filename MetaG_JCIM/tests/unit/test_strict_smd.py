from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from metag.energetics import strict_smd as S

SYMBOLS = ["H", "H"]
COORDS = [[0., 0., 0.], [0.74, 0., 0.]]
BACKEND = {"versions": {"gpu4pyscf": "test"}, "producer_sha256": "test",
           "cds_total_convention_verified": True}


def complete(request, backend):
    return {"status": "complete", "request": request, "backend": backend, "gas_Eh": -1.0,
            "smd_Eh": -1.01, "smd_cds_Eh": 0.001, "gas_converged": True,
            "smd_converged": True, "dgsolv": (-1.01 + 1.0) * S.HARTREE_KJ}


def test_cache_distinguishes_sub_milliangstrom_geometry_spin_and_method():
    key = S._key(SYMBOLS, COORDS, 0, "def2-svp", backend=BACKEND)
    moved = deepcopy(COORDS); moved[1][0] += 0.00001
    assert key != S._key(SYMBOLS, moved, 0, "def2-svp", backend=BACKEND)
    assert key != S._key(SYMBOLS, COORDS, 0, "def2-svp", mult=3, backend=BACKEND)
    assert key != S._key(SYMBOLS, COORDS, 0, "def2-svp", xc="wb97m-v", backend=BACKEND)
    assert key != S._key(SYMBOLS, COORDS, 0, "def2-svp", backend={**BACKEND, "producer_sha256": "new"})


@pytest.mark.parametrize("fault", ["gas_failed", "smd_failed", "nan", "missing_cds", "string_convergence", "boolean_charge"])
def test_bad_calculations_and_equal_but_wrong_types_are_not_cacheable(fault):
    request = S.request_record(SYMBOLS, COORDS, 0, "def2-svp")
    record = complete(request, BACKEND)
    if fault == "gas_failed": record["gas_converged"] = False
    elif fault == "smd_failed": record["smd_converged"] = False
    elif fault == "nan": record["smd_Eh"] = float("nan")
    elif fault == "missing_cds": record.pop("smd_cds_Eh")
    elif fault == "string_convergence": record["gas_converged"] = "true"
    else:
        record["request"] = {**request, "charge": False}
    assert not S._valid_record(record, request, BACKEND)


def test_complete_records_reuse_cache_and_failed_runs_publish_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(S, "backend_identity", lambda: BACKEND)
    calls = []
    def compute(request, backend):
        calls.append(request)
        return complete(request, backend)
    monkeypatch.setattr(S, "compute_record", compute)
    first = S.smd_dgsolv(SYMBOLS, COORDS, 0)
    assert S.smd_dgsolv(SYMBOLS, COORDS, 0) == first
    assert len(calls) == 1 and len(list(tmp_path.glob('*.json'))) == 1
    monkeypatch.setattr(S, "compute_record", lambda request, backend: {**complete(request, backend), "smd_converged": False})
    diagnostics = {}
    assert S.smd_dgsolv(SYMBOLS, COORDS, 0, xc="wb97m-v", diagnostics=diagnostics) is None
    assert diagnostics["status"] == "failed"
    assert len(list(tmp_path.glob('*.json'))) == 1


def test_bad_existing_cache_is_preserved_before_valid_replacement(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(S, "backend_identity", lambda: BACKEND)
    monkeypatch.setattr(S, "compute_record", complete)
    path = tmp_path/(S._key(SYMBOLS, COORDS, 0, "def2-svp", backend=BACKEND)+'.json')
    path.write_text('partial invalid evidence')
    assert S.smd_dgsolv(SYMBOLS, COORDS, 0) is not None
    preserved = list(tmp_path.glob('*.invalid-*.json'))
    assert len(preserved) == 1 and preserved[0].read_text() == 'partial invalid evidence'


@pytest.mark.parametrize('key,value', [('gas_Eh', '-1'), ('smd_cds_Eh', True), ('dgsolv', '-26.25499639')])
def test_numeric_strings_and_boolean_energies_cannot_enter_cache(key, value):
    request = S.request_record(SYMBOLS, COORDS, 0, 'def2-svp')
    record = complete(request, BACKEND);record[key] = value
    assert not S._valid_record(record, request, BACKEND)


def test_json_list_cache_is_preserved_and_recomputed(monkeypatch, tmp_path):
    monkeypatch.setattr(S, '_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(S, 'backend_identity', lambda: BACKEND)
    monkeypatch.setattr(S, 'compute_record', complete)
    path=tmp_path/(S._key(SYMBOLS, COORDS, 0, 'def2-svp', backend=BACKEND)+'.json')
    path.write_text('[]')
    assert isinstance(S.smd_dgsolv(SYMBOLS, COORDS, 0), float)
    assert any(p.read_text()=='[]' for p in tmp_path.glob('*.invalid-*.json'))


def test_legacy_plain_stdout_and_failed_return_code_cannot_supply_energy(monkeypatch):
    result = SimpleNamespace(returncode=0, stdout='DGSOLV -20\n', stderr='')
    monkeypatch.setattr(S.subprocess, "run", lambda *a, **k: result)
    with pytest.raises(ValueError, match="structured"):
        S._rpc([], 'SMD_RECORD ')
    result.returncode = 1
    result.stdout = 'SMD_RECORD '+json.dumps({'status': 'complete'})
    with pytest.raises(RuntimeError):
        S._rpc([], 'SMD_RECORD ')


def install_fake_engines(monkeypatch, gas_ok=True, smd_ok=True):
    calls = []
    class MF:
        def __init__(self):
            self.grids = SimpleNamespace(level=None); self.nlcgrids = SimpleNamespace(level=None)
            self.converged = gas_ok
        def density_fit(self): return self
        def do_nlc(self): return False
        def kernel(self): calls.append('gas'); return -1.0
        def make_rdm1(self): return 'density'
        def SMD(self):
            calls.append('SMD')
            return SimpleNamespace(converged=smd_ok, with_solvent=SimpleNamespace(solvent=None, e_cds=0.001),
                                   kernel=lambda **kw: -1.01)
    def factory(kind):
        def make(*a, **kw): calls.append(kind); return MF()
        return make
    monkeypatch.setitem(sys.modules, 'pyscf', SimpleNamespace(gto=SimpleNamespace(M=lambda **kw: calls.append(kw['spin']))))
    monkeypatch.setitem(sys.modules, 'gpu4pyscf', SimpleNamespace(dft=SimpleNamespace(RKS=factory('RKS'), UKS=factory('UKS'))))
    monkeypatch.setitem(sys.modules, 'gpu4pyscf.scf', SimpleNamespace(hf=SimpleNamespace()))
    return calls


def test_specified_open_shell_uses_uks_and_failed_gas_never_runs_smd(monkeypatch):
    calls = install_fake_engines(monkeypatch)
    request = S.request_record(SYMBOLS, COORDS, -1, 'def2-svp', mult=2)
    assert S.compute_record(request, BACKEND)['status'] == 'complete'
    assert 'UKS' in calls and 'RKS' not in calls and 1 in calls
    calls = install_fake_engines(monkeypatch, gas_ok=False)
    with pytest.raises(ValueError, match='Gas SCF'):
        S.compute_record(S.request_record(SYMBOLS, COORDS, 0, 'def2-svp'), BACKEND)
    assert 'SMD' not in calls
