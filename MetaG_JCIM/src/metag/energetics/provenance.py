"""Producer identities for energies; importing this module does not load QM models."""

from dataclasses import asdict, is_dataclass
from collections.abc import Mapping, Sequence
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import math


def ordered_geometry_record(symbols, positions, charge, spin_multiplicity=1):
    """Export an ordered geometry and its content identity, without invoking QM.

    This records coordinates, not a minimum or chemical-state validation claim.
    Callers must associate it with the energies evaluated at that same geometry.
    """
    import numpy as np
    symbols = list(symbols)
    xyz = np.asarray(positions, dtype=float)
    if (not symbols or any(not isinstance(s, str) or not s for s in symbols)
            or xyz.shape != (len(symbols), 3) or not np.isfinite(xyz).all()):
        raise ValueError("finite ordered (natoms,3) Angstrom coordinates and symbols required")
    integers = []
    for name, value in (("charge", charge), ("spin_multiplicity", spin_multiplicity)):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be an integer")
        try:
            valid = math.isfinite(value) and int(value) == value
        except (ValueError, TypeError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(f"{name} must be a finite integer")
        integers.append(int(value))
    if integers[1] < 1:
        raise ValueError("spin_multiplicity must be positive")
    record = {"schema": "ordered-geometry-v1", "symbols": symbols,
              "positions_angstrom": xyz.tolist(), "charge": integers[0],
              "spin_multiplicity": integers[1]}
    return dict(record, sha256=content_hash(record))


def content_hash(value):
    """Hash a JSON-compatible scientific setting without path-dependent identifiers."""
    def encode(obj):
        # fairchem reference YAMLs are OmegaConf containers, not plain dicts.
        if isinstance(obj, Mapping):
            return dict(obj)
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            return list(obj)
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
        if type(obj).__module__ == "torch" and type(obj).__name__ == "dtype":
            return str(obj)
        raise TypeError(f"unsupported provenance value: {type(obj).__name__}")
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, default=encode).encode()).hexdigest()


def file_hash(path):
    """Hash bytes, including same-size edits within a filesystem timestamp tick."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def software_versions():
    versions = {}
    for name in ("fairchem-core", "torch", "rdkit", "ase", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@lru_cache(maxsize=1)
def source_identity():
    from metag.validation.provenance import source_tree_fingerprint
    return source_tree_fingerprint(Path(__file__).parents[1])


def species_source_fingerprint(package_root):
    """Hash raw-energy dependencies, excluding downstream reference/evaluation code.

    A new pKa/reference/assembly policy may reuse unchanged molecular energies;
    the full estimator identity is recorded separately in each reaction result.
    """
    root = Path(package_root)
    paths = list((root / "energetics").rglob("*.py"))
    paths.extend(root / name for name in (
        "pipeline.py", "chem.py", "symmetry.py", "water_count.py", "config.py", "profile.py",
        "conditions.py"))
    digest = hashlib.sha256(b"metag-raw-species-source-v1\0")
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        contents = path.read_bytes()
        digest.update(str(len(contents)).encode() + b"\0" + contents)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def species_source_identity():
    return species_source_fingerprint(Path(__file__).parents[1])


def runtime_identity(xtb):
    """Identify the running installation and executable without launching xTB."""
    resolved = shutil.which(str(xtb))
    try:
        xtb_sha256 = file_hash(resolved) if resolved else None
    except OSError:
        xtb_sha256 = None
    return {"schema": 2, "source_sha256": species_source_identity(), "source_scope": "raw_species_v1",
            "software": dict(software_versions()), "xtb_sha256": xtb_sha256,
            "imported_module_versions": {
                name: str(getattr(sys.modules.get(module), "__version__", "unavailable"))
                for name, module in (("numpy", "numpy"), ("torch", "torch"), ("rdkit", "rdkit"),
                                     ("ase", "ase"), ("fairchem-core", "fairchem.core"))}}


def model_identity(checkpoint, *, atom_refs, form_elem_refs, inference_settings):
    """Identity of the exact checkpoint and reference data used by the loader."""
    settings = asdict(inference_settings) if is_dataclass(inference_settings) else inference_settings
    return {"checkpoint_sha256": file_hash(checkpoint),
            "references_sha256": content_hash({"atom_refs": atom_refs, "form_elem_refs": form_elem_refs}),
            "inference_sha256": content_hash(settings)}


def producer_identity(predictor, xtb, *, additional_executables=None):
    """Unknown/custom models remain usable, but must not share persistent energies."""
    runtime = runtime_identity(xtb)
    extra = {}
    for name, binary in sorted((additional_executables or {}).items()):
        resolved = shutil.which(str(binary))
        try:
            extra[name] = file_hash(resolved) if resolved else None
        except OSError:
            extra[name] = None
    runtime["additional_executable_sha256"] = extra
    model = getattr(predictor, "metag_provenance", None)
    fields = ("checkpoint_sha256", "references_sha256", "inference_sha256")
    known_model = isinstance(model, dict) and all(
        isinstance(model.get(key), str) and len(model[key]) == 64
        and all(char in "0123456789abcdef" for char in model[key]) for key in fields)
    return dict(runtime, model={key: model[key] for key in fields} if known_model else None,
                cacheable=bool(known_model and runtime["xtb_sha256"]
                               and all(runtime["software"].values()) and all(extra.values())))
