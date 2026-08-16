#!/usr/bin/env python
"""Lightweight results cache for the QM-solvation exploration.

Rationale (user): during exploration we recompute the same species/reactions
repeatedly. Cache expensive results keyed by a METHOD TAG so that (a) reusing the
same method is free, and (b) improving the method (e.g. swapping xtb --ohess thermal
for a batched UMA Hessian, or changing the solvation model) invalidates ONLY the
affected entries — bump/branch the method tag, old entries stay for comparison.

Layout: artifacts/cache/<method_tag>/<key>.json  (one file per entry -> concurrency-safe
atomic writes; delete a method_tag subdir to invalidate that method wholesale).

Key = identity of the physical quantity, canonicalised. Typical identities:
  species free energy : (canonical_smiles, charge, n_water, scheme)
  water-cluster ref   : ("H2O_cluster", n)
The METHOD TAG encodes engine + thermal + solvation + scheme version, e.g.
  "umaS1p2_xtbOhessCosmo_clustercycle_v1".  When you change any of those, change the tag.

Usage:
    from cache import Cache
    C = Cache("umaS1p2_xtbOhessCosmo_clustercycle_v1")
    hit = C.get(smiles, charge, n_water)          # None if absent
    if hit is None:
        val = compute(...)                         # {"G_aq":..., "E_uma":..., "corr":...}
        C.put(val, smiles, charge, n_water)        # identity args after the value
"""
import hashlib
import json
import os
import tempfile

try:
    from rdkit import Chem
    _HAVE_RDKIT = True
except Exception:
    _HAVE_RDKIT = False

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "cache")


def canon_smiles(smi):
    """Canonical SMILES so equivalent inputs hit the same cache entry."""
    if not _HAVE_RDKIT:
        return smi
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else smi


def _key(parts):
    s = "||".join(str(p) for p in parts)
    return hashlib.sha1(s.encode()).hexdigest()[:20]


class Cache:
    def __init__(self, method_tag, root=_ROOT, enabled=True):
        self.tag = method_tag
        self.dir = os.path.join(root, method_tag)
        self.enabled = enabled
        if enabled:
            os.makedirs(self.dir, exist_ok=True)
        self.hits = self.misses = 0

    def _path(self, identity):
        # canonicalise a leading SMILES-looking identity element
        ident = list(identity)
        if ident and isinstance(ident[0], str):
            ident[0] = canon_smiles(ident[0])
        return os.path.join(self.dir, _key(ident) + ".json"), ident

    def get(self, *identity):
        if not self.enabled:
            return None
        path, _ = self._path(identity)
        if os.path.isfile(path):
            try:
                v = json.load(open(path))["value"]
                self.hits += 1
                return v
            except Exception:
                return None
        self.misses += 1
        return None

    def put(self, value, *identity):
        if not self.enabled:
            return value
        path, ident = self._path(identity)
        rec = {"method_tag": self.tag, "identity": ident, "value": value}
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(rec, fh)
        os.replace(tmp, path)          # atomic
        return value

    def stats(self):
        return f"cache[{self.tag}] hits={self.hits} misses={self.misses}"
