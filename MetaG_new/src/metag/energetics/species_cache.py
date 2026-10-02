"""Content-addressed per-species QM cache.

The expensive work (UMA conformer sampling + Hessian thermal + xtb solvation) depends ONLY on
the species (canonical SMILES + charge) and the METHOD SETTINGS that change the number -- NOT on
which reaction the species appears in. So we cache the computed (G, sigma) at the species level,
keyed by a hash of (canonical_SMILES, charge, method, settings, cache_version).

Payoff (the point of this module):
  * A correction that only changes how species COMBINE (pKa transform, isodesmic referencing,
    gating) recomputes NO QM -- every species is a cache hit; the reaction ΔG is reassembled in ms.
  * A correction that introduces NEW species for one class (cofactor cores, explicit waters,
    truncated cores) computes only the new species; the ~90% unchanged species hit cache.
  * Species that recur across reactions (water, NAD/ATP/CoA) are computed ONCE, reused everywhere.

Correctness = the key. `settings` MUST contain everything that changes the number (engine model,
solvation model, convergence tol, sampling budget knobs, explicit-cluster knobs). Change any of
them (or bump CACHE_VERSION) -> new key -> clean recompute. No stale results. Chunk sizes are NOT
in the key (they don't change the result, only GPU memory).

Storage: one small JSON per key under artifacts/species_cache/, written atomically (tmp+rename) so
the parallel multi-GPU sweep is race-safe (distinct keys -> distinct files; same key -> idempotent).
Disable with SPECIES_CACHE=0.
"""
import os
import json
import hashlib
import tempfile

try:
    from rdkit import Chem
except Exception:                                            # RDKit absent -> canonicalization is a no-op
    Chem = None

# cache lives OUTSIDE the package (never write into an installed package). Override with METAG_CACHE.
CACHE_DIR = os.environ.get("METAG_CACHE", os.path.join(os.getcwd(), ".metag_cache", "species_cache"))
CACHE_VERSION = "v1"                                          # bump to invalidate ALL cached species
_ENABLED = os.environ.get("SPECIES_CACHE", "1").lower() not in ("0", "off", "false", "no")

_hits = _misses = _writes = 0                                # per-process stats (for logging)


def canonical(smi):
    if Chem is None:
        return smi
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else smi


def _path(smi, q, method, settings):
    cs = canonical(smi)
    payload = json.dumps({"smi": cs, "q": int(q), "method": method,
                          "settings": settings, "ver": CACHE_VERSION}, sort_keys=True)
    h = hashlib.sha256(payload.encode()).hexdigest()[:32]
    return os.path.join(CACHE_DIR, h + ".json"), cs


def get(smi, q, method, settings, with_meta=False):
    """Return cached (G, sigma) -- or (G, sigma, meta) with with_meta=True -- or None. Never raises."""
    global _hits, _misses
    if not _ENABLED:
        return None
    try:
        p, _ = _path(smi, q, method, settings)
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            _hits += 1
            return (d["G"], d["sigma"], d.get("meta") or {}) if with_meta else (d["G"], d["sigma"])
    except Exception:
        pass
    _misses += 1
    return None


def put(smi, q, method, settings, G, sigma, meta=None):
    """Store (G, sigma) for a species, plus optional provenance `meta` (warnings, minima count, ...).
    Atomic; never raises (a failed write is reported once on stderr). No-op on G is None."""
    global _writes, _write_error_reported
    if not _ENABLED or G is None:
        return
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        p, cs = _path(smi, q, method, settings)
        rec = {"smi": cs, "q": int(q), "method": method, "settings": settings,
               "G": float(G), "sigma": None if sigma is None else float(sigma), "ver": CACHE_VERSION,
               "meta": meta or {}}
        fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(rec, f)
        os.replace(tmp, p)                                    # atomic on POSIX
        _writes += 1
    except Exception as e:
        if not _write_error_reported:
            import sys
            print(f"[species_cache] write failed ({e}); continuing without caching", file=sys.stderr)
            _write_error_reported = True


_write_error_reported = False


def stats():
    return {"hits": _hits, "misses": _misses, "writes": _writes, "enabled": _ENABLED}
