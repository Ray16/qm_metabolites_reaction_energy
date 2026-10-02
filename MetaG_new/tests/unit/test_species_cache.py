import importlib
import json
from pathlib import Path


def _reload_cache(monkeypatch, cache_dir):
    monkeypatch.setenv("METAG_CACHE", str(cache_dir))
    monkeypatch.setenv("SPECIES_CACHE", "1")
    import metag.energetics.species_cache as species_cache

    return importlib.reload(species_cache)


def test_cache_key_is_canonical_and_settings_sensitive(tmp_path, monkeypatch):
    cache = _reload_cache(monkeypatch, tmp_path)
    settings = {"model": "uma-s-1p2p1", "solv": "alpb", "physics": "2026-10-01c"}

    ethanol, canonical = cache._path("C(C)O", 0, "implicit", settings)
    equivalent, _ = cache._path("CCO", 0, "implicit", settings)
    different_charge, _ = cache._path("CCO", 1, "implicit", settings)
    different_settings, _ = cache._path(
        "CCO", 0, "implicit", dict(settings, solv="cosmo")
    )

    assert canonical == "CCO"
    assert ethanol == equivalent
    assert ethanol != different_charge
    assert ethanol != different_settings


def test_cache_write_is_complete_and_resumable(tmp_path, monkeypatch):
    cache = _reload_cache(monkeypatch, tmp_path)
    settings = {"model": "uma-s-1p2p1", "solv": "alpb", "physics": "2026-10-01c"}

    cache.put("CCO", 0, "implicit", settings, -123.5, 1.25, {"minima": 4})

    path, _ = cache._path("CCO", 0, "implicit", settings)
    assert Path(path).is_file()
    assert not list(tmp_path.glob("*.tmp"))
    assert cache.get("C(C)O", 0, "implicit", settings, with_meta=True) == (
        -123.5,
        1.25,
        {"minima": 4},
    )
    record = json.loads(Path(path).read_text())
    assert record["ver"] == cache.CACHE_VERSION
    assert record["settings"] == settings


def test_disabled_cache_does_not_read_or_write(tmp_path, monkeypatch):
    monkeypatch.setenv("METAG_CACHE", str(tmp_path))
    monkeypatch.setenv("SPECIES_CACHE", "0")
    import metag.energetics.species_cache as species_cache

    cache = importlib.reload(species_cache)
    settings = {"physics": "2026-10-01c"}
    cache.put("O", 0, "implicit", settings, -1.0, 0.1)

    assert cache.get("O", 0, "implicit", settings) is None
    assert list(tmp_path.iterdir()) == []

