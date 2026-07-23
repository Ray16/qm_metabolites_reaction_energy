"""Per-compound driver: structure -> conformers -> DFT -> ensemble G(aq).

Ties the pipeline stages together for one `Metabolite`, with JSON caching so a
re-run skips compounds already completed (resumable batch runs).
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from . import config
from .conformers import generate_screened_conformers
from .geometry import write_xyz
from .qm_backend import ORCABackend, QMBackend, QMResult
from .structures import Metabolite, load_metabolites
from .thermo import CompoundEnergy, assemble_compound_energy


def _cache_path(cpd_id: str) -> str:
    return os.path.join(config.COMPOUND_CACHE_DIR, f"{cpd_id}.json")


def load_cached(cpd_id: str) -> CompoundEnergy | None:
    path = _cache_path(cpd_id)
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    return CompoundEnergy(
        cpd_id=d["cpd_id"],
        gibbs_kJ=d["gibbs_kJ"],
        n_conformers=d["n_conformers"],
        min_conformer_kJ=d["min_conformer_kJ"],
        method=d["method"],
        all_converged=d["all_converged"],
        gibbs_highlevel_kJ=d.get("gibbs_highlevel_kJ"),
        sp_method=d.get("sp_method"),
        wall_seconds=d.get("wall_seconds"),
    )


_TIMINGS_HEADER = ["cpd_id", "name", "n_atoms", "label", "stage",
                   "wall_s", "orca_s", "ok"]


def _append_timings(meta: Metabolite, job_times: list[dict]) -> None:
    """Append one row per ORCA job to the tracking CSV (results/timings.csv).

    Opened in append mode (O_APPEND): each row is small (<4 KB) so concurrent
    batch workers append atomically without clobbering each other. The header is
    written only when the file is new (batch pre-creates it; see compute_batch)."""
    if not job_times:
        return
    os.makedirs(os.path.dirname(config.TIMINGS_CSV), exist_ok=True)
    write_header = not os.path.exists(config.TIMINGS_CSV) or \
        os.path.getsize(config.TIMINGS_CSV) == 0
    with open(config.TIMINGS_CSV, "a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(_TIMINGS_HEADER)
        for j in job_times:
            w.writerow([meta.cpd_id, meta.name, meta.n_atoms, j["label"],
                        j["stage"], j["wall_s"], j.get("orca_s"), j["ok"]])


def _save_cache(energy: CompoundEnergy) -> None:
    os.makedirs(config.COMPOUND_CACHE_DIR, exist_ok=True)
    with open(_cache_path(energy.cpd_id), "w") as fh:
        json.dump(energy.to_dict(), fh, indent=2)


def _save_conformer_records(cpd_id: str, records: list[dict]) -> None:
    """Persist per-conformer geometries/energies so a later SP needs no re-opt."""
    os.makedirs(config.CONFORMER_DIR, exist_ok=True)
    path = os.path.join(config.CONFORMER_DIR, f"{cpd_id}.json")
    with open(path, "w") as fh:
        json.dump(records, fh, indent=2)


def _backend_sp_keywords(backend: QMBackend) -> str | None:
    """The configured tier-2 single-point keywords for this backend, if any."""
    level = getattr(backend, "level", None)
    return getattr(level, "sp_keywords", None)


def compute_compound(
    meta: Metabolite,
    backend: QMBackend,
    *,
    conf_settings: config.ConformerSettings = config.DEFAULT_CONFORMERS,
    parallel: config.ParallelSettings = config.DEFAULT_PARALLEL,
    conditions: config.Conditions = config.DEFAULT_CONDITIONS,
    use_cache: bool = True,
) -> CompoundEnergy:
    """Full structure -> G(aq) for one compound. Cached by compound id.

    Runs tier-1 (opt+freq) per conformer, persists the optimised geometry, and --
    if the backend has tier-2 single-point keywords -- refines the electronic
    energy and forms a second ensemble G. A cached tier-1-only result is treated
    as incomplete when tier-2 is requested.
    """
    sp_keywords = _backend_sp_keywords(backend)
    if use_cache:
        cached = load_cached(meta.cpd_id)
        if cached is not None and (
            sp_keywords is None or cached.gibbs_highlevel_kJ is not None
        ):
            print(f"[compute] {meta.cpd_id} cached -> {cached.best_gibbs_kJ():.1f} kJ/mol")
            return cached

    print(f"[compute] {meta.cpd_id} ({meta.name}): screening conformers...")
    screened = generate_screened_conformers(
        meta, conf_settings=conf_settings, parallel=parallel
    )
    print(f"[compute] {meta.cpd_id}: {len(screened)} conformer(s) -> DFT")

    geom_dir = os.path.join(config.GEOMETRY_DIR, meta.cpd_id)
    os.makedirs(geom_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"orca_{meta.cpd_id}_", dir=config.SCRATCH_ROOT)

    results: list[QMResult] = []
    highlevel_gibbs: list[float] | None = [] if sp_keywords else None
    records: list[dict] = []
    for i, conf in enumerate(screened):
        label = f"{meta.cpd_id}_c{i}"
        wd = os.path.join(work, label)
        res = backend.compute_gibbs(conf.geometry, label, wd)
        results.append(res)

        # Persist the optimised geometry for future re-use.
        geom_path = os.path.join(geom_dir, f"conf{i}.xyz")
        write_xyz(res.geometry, geom_path, comment=f"{meta.cpd_id} conf{i} {res.method}")
        thermal_corr = res.gibbs_hartree - res.electronic_hartree
        record = {
            "conformer": i,
            "geometry_xyz": geom_path,
            "gibbs_hartree": res.gibbs_hartree,
            "electronic_hartree": res.electronic_hartree,
            "thermal_corr_hartree": thermal_corr,
            "converged": res.converged,
            "method": res.method,
        }
        print(
            f"[compute] {meta.cpd_id} conf {i}: G(tier1)={res.gibbs_kJ:.1f} kJ/mol "
            f"(converged={res.converged})"
        )

        # Tier 2: higher-level single point on the optimised geometry.
        if sp_keywords:
            e_sp = backend.single_point_energy(
                res.geometry, sp_keywords, f"{label}_sp", os.path.join(wd, "sp")
            )
            g_highlevel = e_sp + thermal_corr   # SP electronic + tier-1 RRHO
            highlevel_gibbs.append(g_highlevel)
            record.update(
                sp_electronic_hartree=e_sp,
                sp_gibbs_hartree=g_highlevel,
                sp_method=sp_keywords,
            )
            print(
                f"[compute] {meta.cpd_id} conf {i}: G(tier2)="
                f"{g_highlevel * 2625.499639:.1f} kJ/mol [{sp_keywords.split()[0]}]"
            )
        records.append(record)

    wall_s = getattr(backend, "total_wall_s", None)
    energy = assemble_compound_energy(
        meta.cpd_id,
        results,
        highlevel_gibbs_hartree=highlevel_gibbs,
        sp_method=sp_keywords,
        wall_seconds=wall_s,
        conditions=conditions,
    )
    _save_cache(energy)
    _save_conformer_records(meta.cpd_id, records)
    _append_timings(meta, getattr(backend, "job_times", []))
    wall_str = "n/a" if wall_s is None else f"{wall_s:.0f}s ({wall_s / 60:.1f} min)"
    print(
        f"[compute] {meta.cpd_id} DONE -> G(aq)={energy.best_gibbs_kJ():.1f} kJ/mol "
        f"in {wall_str}"
    )
    return energy


# Above this many atoms, conformational averaging buys little relative to its
# (steeply scaling) DFT cost, so large cofactors use only the single best conformer.
LARGE_ATOM_THRESHOLD = 50
LARGE_MAX_CONFS = 1


def conformer_settings_for(
    meta: Metabolite, base: config.ConformerSettings = config.DEFAULT_CONFORMERS
) -> config.ConformerSettings:
    """Size-tuned conformer budget: cap large molecules to one DFT conformer."""
    import dataclasses

    if meta.n_atoms > LARGE_ATOM_THRESHOLD:
        return dataclasses.replace(base, max_qm_confs=LARGE_MAX_CONFS)
    return base


def _worker(cpd_id: str) -> tuple[str, str | None]:
    """Process-pool worker: compute one compound by id. Returns (id, error|None).

    Runs in a fresh process so a single ORCA crash cannot take down the batch.
    """
    config.ensure_dirs()
    try:
        (meta,) = [m for m in load_metabolites() if m.cpd_id == cpd_id]
        backend = ORCABackend()
        compute_compound(meta, backend, conf_settings=conformer_settings_for(meta))
        return cpd_id, None
    except Exception:  # noqa: BLE001 - record any failure, keep the batch alive
        return cpd_id, traceback.format_exc()


def compute_batch(
    metabolites: list[Metabolite],
    *,
    parallel: config.ParallelSettings = config.DEFAULT_PARALLEL,
) -> dict[str, CompoundEnergy]:
    """Compute many compounds concurrently (one ORCA job per process).

    Concurrency is `parallel.max_concurrent_jobs`; each job internally uses
    `parallel.orca_nprocs` cores.
    """
    config.ensure_dirs()
    # Pre-create the timings CSV (with header) in the parent process so the
    # concurrent workers below only ever append data rows -- no header race.
    if not os.path.exists(config.TIMINGS_CSV) or os.path.getsize(config.TIMINGS_CSV) == 0:
        with open(config.TIMINGS_CSV, "w", newline="") as fh:
            csv.writer(fh).writerow(_TIMINGS_HEADER)
    # Smallest first: the benchmark fills in quickly while large cofactors grind.
    todo = sorted(
        (m for m in metabolites if load_cached(m.cpd_id) is None),
        key=lambda m: m.n_atoms,
    )
    print(
        f"[batch] {len(metabolites)} compounds, {len(todo)} to compute "
        f"({parallel.max_concurrent_jobs} concurrent x {parallel.orca_nprocs} cores)"
    )

    with ProcessPoolExecutor(max_workers=parallel.max_concurrent_jobs) as ex:
        futures = {ex.submit(_worker, m.cpd_id): m.cpd_id for m in todo}
        for fut in as_completed(futures):
            cpd_id, err = fut.result()
            if err:
                print(f"[batch] FAILED {cpd_id}:\n{err}")

    energies: dict[str, CompoundEnergy] = {}
    for m in metabolites:
        cached = load_cached(m.cpd_id)
        if cached is not None:
            energies[m.cpd_id] = cached
    print(f"[batch] complete: {len(energies)}/{len(metabolites)} succeeded")
    return energies
