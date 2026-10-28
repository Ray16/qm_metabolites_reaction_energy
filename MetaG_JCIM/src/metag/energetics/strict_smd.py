"""Auditable gas/SMD single-point potential differences at identical coordinates.

No reaction correction, thermal free energy or validated accuracy is implied.
GPU calls inherit the caller's reserved device and native thread allocation.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from numbers import Real
from collections.abc import Mapping
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

import numpy as np
from rdkit import Chem

HARTREE_KJ = 2625.499639
REDOX_PY = os.environ.get("SMD_PY", "/homes/rzhu/miniforge3/envs/redox/bin/python")
_CACHE_DIR = os.environ.get("SMD_CACHE", str(Path.cwd() / ".metag_cache/smd_cache"))
SOURCE_ROOT = Path(__file__).resolve().parents[2]


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def request_record(symbols, coords, q, basis, *, xc="PBE0", mult=1):
    positions = np.asarray(coords, dtype=float)
    symbols = list(symbols)
    if not symbols or positions.shape != (len(symbols), 3) or not np.isfinite(positions).all():
        raise ValueError("Finite (natoms,3) Angstrom coordinates are required")
    if isinstance(q, bool) or not math.isfinite(float(q)) or int(q) != float(q):
        raise ValueError("Charge must be a finite integer")
    if isinstance(mult, bool) or not math.isfinite(float(mult)) or int(mult) != float(mult) or mult < 1:
        raise ValueError("Spin multiplicity must be a positive integer")
    table = Chem.GetPeriodicTable()
    numbers = []
    for symbol in symbols:
        number = table.GetAtomicNumber(symbol)
        if not number or table.GetElementSymbol(number) != symbol:
            raise ValueError("Element symbols are required; isotope-specific solvent is unsupported")
        numbers.append(number)
    electrons = sum(numbers) - int(q)
    if electrons < int(mult)-1 or (electrons - (int(mult)-1)) % 2:
        raise ValueError("Electron count and specified spin multiplicity are inconsistent")
    if not isinstance(basis, str) or not basis or not isinstance(xc, str) or not xc:
        raise ValueError("Explicit functional and basis names are required")
    return {"schema": "matched-smd-v2", "symbols": symbols, "coordinates_A": positions.tolist(),
            "charge": int(q), "spin_multiplicity": int(mult), "basis": basis, "functional": xc,
            "solvent": "water", "grid_level": 3, "nlc_grid_level": 3, "density_fit": True,
            "scf_tolerance_Eh": 1e-9, "max_scf_cycles": 100, "overlap_cutoff": 1e-6,
            "quantity": "matched_geometry_smd_minus_gas_potential_kj_mol"}


def backend_identity():
    import pyscf
    import gpu4pyscf
    import cupy
    modules = {"pyscf": pyscf, "gpu4pyscf": gpu4pyscf, "cupy": cupy, "numpy": np}
    versions = {}
    for name, module in modules.items():
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = getattr(module, "__version__", "unknown")
    paths = {"gpu4pyscf/"+name: Path(gpu4pyscf.__file__).resolve().parent/name
             for name in ("scf/hf.py", "dft/rks.py", "dft/uks.py", "solvent/smd.py", "solvent/_attach_solvent.py")}
    paths["pyscf/solvent/smd.py"] = Path(pyscf.__file__).resolve().parent/"solvent/smd.py"
    attach = paths["gpu4pyscf/solvent/_attach_solvent.py"].read_text()
    return {"versions": versions, "python": list(sys.version_info[:2]),
            "producer_sha256": _hash(Path(__file__).read_bytes()),
            "sources_sha256": {name: _hash(path.read_bytes()) for name, path in paths.items()},
            "cds_total_convention_verified": "e_tot += e_cds" in attach}


def _key(symbols, coords, q, basis, *, xc="PBE0", mult=1, backend=None):
    payload = {"request": request_record(symbols, coords, q, basis, xc=xc, mult=mult), "backend": backend}
    return _hash(json.dumps(payload, sort_keys=True, allow_nan=False).encode())


def validated_difference(gas_Eh, smd_Eh, *, gas_converged, smd_converged, cds_Eh):
    if not all(isinstance(flag, (bool, np.bool_)) and bool(flag) for flag in (gas_converged, smd_converged)):
        raise ValueError("Both gas and SMD SCFs must converge")
    if not all(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
               and math.isfinite(float(value)) for value in (gas_Eh, smd_Eh, cds_Eh)):
        raise ValueError("Gas, SMD and CDS energies must be finite")
    value = (float(smd_Eh)-float(gas_Eh))*HARTREE_KJ
    if not math.isfinite(value):
        raise ValueError("Potential difference overflowed")
    return value


def compute_record(request, backend):
    from pyscf import gto
    from gpu4pyscf import dft
    from gpu4pyscf.scf import hf
    if not backend.get("cds_total_convention_verified"):
        raise ValueError("SMD total/CDS convention has not been established for this implementation")
    hf.remove_overlap_zero_eigenvalue = True
    hf.overlap_zero_eigenvalue_threshold = request["overlap_cutoff"]
    mol = gto.M(atom=list(zip(request["symbols"], request["coordinates_A"])), unit="Angstrom",
                basis=request["basis"], charge=request["charge"], spin=request["spin_multiplicity"]-1,
                verbose=0)
    factory = dft.RKS if request["spin_multiplicity"] == 1 else dft.UKS
    mf = factory(mol, xc=request["functional"]).density_fit()
    mf.grids.level = request["grid_level"]
    mf.nlcgrids.level = request["nlc_grid_level"]
    mf.conv_tol = request["scf_tolerance_Eh"]
    mf.max_cycle = request["max_scf_cycles"]
    if request["functional"].lower().endswith("-v") and not mf.do_nlc():
        raise ValueError("Requested VV10 functional has no active nonlocal correlation")
    gas = float(mf.kernel())
    if not mf.converged or not math.isfinite(gas):
        raise ValueError("Gas SCF failed; no solvent calculation or cache publication")
    ms = mf.SMD()
    ms.with_solvent.solvent = request["solvent"]
    solution = float(ms.kernel(dm0=mf.make_rdm1()))
    cds = ms.with_solvent.e_cds
    cds = float(cds.get() if hasattr(cds, "get") else cds)
    difference = validated_difference(gas, solution, gas_converged=mf.converged,
                                      smd_converged=ms.converged, cds_Eh=cds)
    return {"status": "complete", "request": request, "backend": backend,
            "gas_Eh": gas, "smd_Eh": solution, "smd_cds_Eh": cds,
            "gas_converged": True, "smd_converged": True, "dgsolv": difference,
            "scope": "single-point potential difference; specified spin not proven ground state; no thermal/free-energy or isotope-solvent validation"}


def _rpc(arguments, marker):
    environment = {**os.environ, "PYTHONPATH": str(SOURCE_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run([REDOX_PY, "-m", "metag.energetics.smd_solv", *arguments],
                            cwd=SOURCE_ROOT, env=environment, capture_output=True, text=True, timeout=900)
    if result.returncode:
        raise RuntimeError("SMD backend process failed: " + result.stderr[-500:])
    rows = [line[len(marker):] for line in result.stdout.splitlines() if line.startswith(marker)]
    if len(rows) != 1:
        raise ValueError("Exactly one structured backend result is required")
    return json.loads(rows[0])


def _valid_record(record, request, backend):
    if not isinstance(record, Mapping) or not isinstance(backend, Mapping):
        return False
    try:
        canonical = lambda value: json.dumps(value, sort_keys=True, allow_nan=False)
        if (record.get("status") != "complete" or not backend.get("cds_total_convention_verified")
                or canonical(record.get("request")) != canonical(request)
                or canonical(record.get("backend")) != canonical(backend)):
            return False
        value = validated_difference(record["gas_Eh"], record["smd_Eh"], gas_converged=record["gas_converged"],
                                     smd_converged=record["smd_converged"], cds_Eh=record["smd_cds_Eh"])
        difference = record["dgsolv"]
        return (isinstance(difference, Real) and not isinstance(difference, (bool, np.bool_))
                and math.isfinite(float(difference))
                and math.isclose(value, float(difference), abs_tol=1e-8, rel_tol=1e-12))
    except (KeyError, ValueError, TypeError):
        return False


def smd_dgsolv(symbols, coords, q, basis="def2-svp", *, xc="PBE0", mult=1, diagnostics=None):
    """Complete potential difference or None, with producer-identified atomic cache."""
    diagnostics = diagnostics if diagnostics is not None else {}
    try:
        request = request_record(symbols, coords, q, basis, xc=xc, mult=mult)
        try:
            backend = backend_identity()
            local = True
        except (ImportError, OSError):
            backend = _rpc(["--backend-info"], "SMD_BACKEND ")
            local = False
        if not backend.get("cds_total_convention_verified"):
            raise ValueError("SMD total/CDS convention is unverified")
        key = _key(symbols, coords, q, basis, xc=xc, mult=mult, backend=backend)
        cache = Path(_CACHE_DIR)/f"{key}.json"
        invalid_cache = False
        if cache.exists():
            try:
                cached = json.loads(cache.read_text())
                if _valid_record(cached, request, backend):
                    diagnostics.update(status="complete", source="cache", record=cached)
                    return float(cached["dgsolv"])
            except (ValueError, OSError):
                pass
            invalid_cache = True
        if local:
            record = compute_record(request, backend)
        else:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"request.json"
                path.write_text(json.dumps(request, allow_nan=False))
                record = _rpc(["--request", str(path)], "SMD_RECORD ")
        if not _valid_record(record, request, backend):
            raise ValueError("Incomplete or mismatched producer record; not cached")
        cache.parent.mkdir(parents=True, exist_ok=True)
        if invalid_cache:
            cache.replace(cache.with_name(cache.stem+".invalid-"+uuid.uuid4().hex+".json"))
        with tempfile.NamedTemporaryFile("w", dir=cache.parent, suffix=".tmp", delete=False) as handle:
            json.dump(record, handle, allow_nan=False)
            temporary = Path(handle.name)
        temporary.replace(cache)
        diagnostics.update(status="complete", source="fresh", record=record)
        return float(record["dgsolv"])
    except Exception as exc:
        diagnostics.update(status="failed", reason=repr(exc))
        return None


def _compute_smd(symbols, coords, q, basis, xc="PBE0", mult=1):
    request = request_record(symbols, coords, q, basis, xc=xc, mult=mult)
    return compute_record(request, backend_identity())["dgsolv"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-info", action="store_true")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--xyz", type=Path)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--xc", default="PBE0")
    parser.add_argument("--spin-multiplicity", type=int, default=1)
    args = parser.parse_args()
    backend = backend_identity()
    if args.backend_info:
        print("SMD_BACKEND " + json.dumps(backend))
        return
    if args.request:
        raw = json.loads(args.request.read_text())
        request = request_record(raw["symbols"], raw["coordinates_A"], raw["charge"], raw["basis"],
                                 xc=raw["functional"], mult=raw["spin_multiplicity"])
        if request != raw:
            raise ValueError("Unsupported or altered request settings")
    else:
        if args.xyz is None:
            parser.error("Supply --request or --xyz")
        lines = args.xyz.read_text().splitlines()
        count = int(lines[0])
        rows = [line.split() for line in lines[2:2+count]]
        if len(rows) != count or any(len(row) != 4 for row in rows):
            raise ValueError("Incomplete XYZ")
        request = request_record([r[0] for r in rows], [[float(x) for x in r[1:]] for r in rows],
                                 args.charge, args.basis, xc=args.xc, mult=args.spin_multiplicity)
    record = compute_record(request, backend)
    print("SMD_RECORD " + json.dumps(record, allow_nan=False))
    print("DGSOLV " + str(record["dgsolv"]))


if __name__ == "__main__":
    main()
