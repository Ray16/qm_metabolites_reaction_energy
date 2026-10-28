"""Deterministic source fingerprints for production records."""

import hashlib
from pathlib import Path


def source_tree_fingerprint(package_root, additional_files=()):
    """Hash relative paths and contents of Python source files deterministically."""
    package_root = Path(package_root).resolve()
    paths = list(package_root.rglob("*.py"))
    paths.extend(Path(path).resolve() for path in additional_files)
    common_root = package_root.parent

    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        try:
            relative = path.relative_to(common_root)
        except ValueError:
            relative = path
        digest.update(str(relative).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()

