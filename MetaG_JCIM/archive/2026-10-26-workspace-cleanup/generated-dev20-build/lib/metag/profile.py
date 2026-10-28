"""Step-0 wall-clock profiler — the GPU-vs-CPU split per reaction.

Purpose: decide (with numbers, not guesses) whether the `batched_fire` per-step CPU
round-trip (item A: positions GPU->CPU->ASE + AtomicData.from_ase rebuild every step) is a
big enough fraction of per-reaction GPU time to be worth optimizing, and how large the xtb
solvation CPU cost is relative to the UMA GPU forward.

Enable with METAG_PROFILE=1. When disabled it is a near-zero-overhead no-op (a single dict
lookup returning a shared null context), so it is safe to leave in the hot path. It NEVER
changes any number — pure timing.

Buckets (leaf-level, so they nest inside the pipeline phases rather than tile it):
  uma.gpu        pu.predict forward. CUDA-synced while profiling so wall == real GPU time.
                 Covers ALL forward passes: conformer ranking + FIRE relax + Hessian.
  uma.build      AtomicData.from_ase + batch assembly + host->device copy  (item-A, half 1)
  uma.writeback  per-FIRE-step positions GPU->CPU->ASE set_positions loop  (item-A, half 2)
  xtb.solv       xtb solvation single points (subprocess). SUMMED across the solvation thread
                 pool, so this is CPU-seconds; divide by METAG_XTB_WORKERS for wall-time.
  hessian        uma_gibbs_corr body (its forward passes are also counted under uma.gpu).

Read it as: item-A fraction = (uma.build + uma.writeback) / (uma.gpu + uma.build + uma.writeback).
If that is small, item A (and ALCHEMI's GPU-resident batch) buys little; if large, it is worth it.
"""
import atexit
import os
import threading
import time

ENABLED = os.environ.get("METAG_PROFILE", "0").lower() not in ("0", "", "false", "no", "off")

_lock = threading.Lock()
_acc: dict[str, list] = {}          # name -> [total_seconds, count]


class _Timer:
    __slots__ = ("name", "sync", "_t0")

    def __init__(self, name, sync):
        self.name = name
        self.sync = sync

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.sync:                               # make wall == real GPU time (CUDA is async)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
        dt = time.perf_counter() - self._t0
        with _lock:
            e = _acc.get(self.name)
            if e is None:
                _acc[self.name] = [dt, 1]
            else:
                e[0] += dt
                e[1] += 1
        return False


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL = _Null()


def timed(name, sync=False):
    """Context manager accumulating wall time into bucket `name`. No-op unless METAG_PROFILE=1.
    sync=True inserts torch.cuda.synchronize() on exit so a GPU-forward bucket measures real
    kernel time rather than the async launch."""
    return _Timer(name, sync) if ENABLED else _NULL


def reset():
    with _lock:
        _acc.clear()


def report(log=print, header=""):
    if not ENABLED:
        return
    with _lock:
        snap = sorted(((n, t, c) for n, (t, c) in _acc.items()), key=lambda x: -x[1])
    if not snap:
        return
    total = sum(t for _, t, _ in snap)
    workers = os.environ.get("METAG_XTB_WORKERS", "8")
    log(f"[profile] {header}  (METAG_XTB_WORKERS={workers}; uma.*/hessian are single-thread wall, "
        f"xtb.solv is summed CPU-seconds across the pool)")
    for n, t, c in snap:
        log(f"[profile]   {n:14s} {t:9.2f}s  n={c}")
    gpu = _acc.get("uma.gpu", [0, 0])[0]
    io = _acc.get("uma.build", [0, 0])[0] + _acc.get("uma.writeback", [0, 0])[0]
    if gpu + io > 0:
        log(f"[profile]   -> item-A CPU round-trip = {100 * io / (gpu + io):.1f}% of UMA relax/forward time")


if ENABLED:
    atexit.register(lambda: report(header="ATEXIT cumulative (last reset window)"))
