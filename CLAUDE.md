# Repo conventions & performance rules (thermodynamic_calc)

## NEVER use `conda run -n <env> <cmd>` in a loop or hot path
`conda run` re-activates the environment on **every** call — measured **~30 s of
pure overhead per invocation** (vs **0.57 s** for the same `xtb` single point via
the direct binary). In a threaded/parallel loop this compounds into a hang
(e.g. 5 seeds × 8 threads × 30 s = machine thrash). This has burned us once
(2026-08-14, step4e xtb solvation stalled 10+ min at 0% GPU).

**Instead, call the binary by its full path** and set thread limits:
```python
XTB_BIN = "/homes/rzhu/miniforge3/envs/xtb/bin/xtb"   # or f"{os.environ['HOME']}/miniforge3/envs/xtb/bin/xtb"
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
subprocess.run([XTB_BIN, xyz, "--gfn", "2", "--chrg", str(q), "--sp", "--alpb", "water"],
               cwd=tmpdir, env=ENV, capture_output=True, text=True, timeout=120)
```
`conda run` is fine ONCE at the top of a shell script; never per-item in Python.

## Set thread limits for CPU tools when parallelizing
CPU QM tools (xtb, etc.) default to multi-threaded (all cores). Running N of them
concurrently oversubscribes the CPU and everything crawls. Always set
`OMP_NUM_THREADS=1` (and `MKL_NUM_THREADS=1`) on each worker when you fan out, and
keep total workers ≤ physical cores.

## UMA / batching (see PROGRESS.md, memory)
- Always BATCH UMA inference (`batched_relax.py`); never sequential per-structure BFGS.
- `AtomicData.from_ase` MUST get `r_data_keys=["spin","charge"]` or every structure
  is computed NEUTRAL. charge/spin in `atoms.info` must be `int`.

## Repo
`thermodynamic_calc/` is its own git repo (remote `qm_metabolites_reaction_energy`,
branch `master`, SSH). Commit + push after each meaningful step; the daily cron
(`daily_commit.sh`, 23:47) now also pushes.
