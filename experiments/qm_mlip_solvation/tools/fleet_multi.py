"""Multi-node fleet dispatcher: fan unified_pipeline --only KEYS across ALL free GPUs on ALL reachable
lambda nodes, ONE job per GPU at a time (each slot runs its keys sequentially -> no 2-per-GPU OOM, no CPU
oversubscription with OMP=1). Collection is via shared-NFS logs (LOGDIR). Non-blocking: launches one
detached ssh per (node,gpu) slot and returns.

Usage: python tools/fleet_multi.py --rxn_file scripts/reactions_tecrdb_all.json --logdir logs/xxx \
         --flags "AUTO_TRUNCATE=1 ..." --seeds 1,2,3,4 --keys "rxn1 rxn2 ..." [--exclude lambda3,lambda6]
"""
import argparse, os, re, subprocess, sys

DIR = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/experiments/qm_mlip_solvation"
PY = "/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/uma/bin/python"


def parse_slots(exclude):
    """Run check_gpu, return list of (node, gpu_index) for every free GPU on reachable nodes."""
    out = subprocess.run(["check_gpu"], capture_output=True, text=True).stdout
    slots = []
    for line in out.splitlines():
        m = re.match(r"\s*(lambda\d+):\s+([0-9,\-]+)\s+free", line)
        if not m:
            continue
        node, spec = m.group(1), m.group(2)
        if node in exclude:
            continue
        for part in spec.split(","):
            if "-" in part:
                a, b = part.split("-"); slots += [(node, g) for g in range(int(a), int(b) + 1)]
            else:
                slots.append((node, int(part)))
    return slots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rxn_file", required=True); ap.add_argument("--logdir", required=True)
    ap.add_argument("--flags", default=""); ap.add_argument("--seeds", default="1,2,3,4")
    ap.add_argument("--keep", default="10"); ap.add_argument("--pool", default="48")
    ap.add_argument("--keys", required=True); ap.add_argument("--exclude", default="lambda3")
    a = ap.parse_args()
    exclude = set(a.exclude.split(",")) if a.exclude else set()
    keys = a.keys.split()
    slots = parse_slots(exclude)
    if not slots:
        print("no free slots"); sys.exit(1)
    os.makedirs(os.path.join(DIR, a.logdir), exist_ok=True)
    # round-robin keys -> slots
    buckets = {i: [] for i in range(len(slots))}
    todo = [k for k in keys if not _done(os.path.join(DIR, a.logdir, k + ".log"))]
    for i, k in enumerate(todo):
        buckets[i % len(slots)].append(k)
    print(f"{len(todo)} pending keys across {len(slots)} GPU slots on "
          f"{len(set(n for n,_ in slots))} nodes ({sorted(set(n for n,_ in slots))})")
    procs = []
    for i, (node, gpu) in enumerate(slots):
        ks = buckets[i]
        if not ks:
            continue
        inner = " ; ".join(
            f"CUDA_VISIBLE_DEVICES={gpu} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RXN_FILE={a.rxn_file} {a.flags} "
            f"{PY} scripts/unified_pipeline.py --only {k} --seeds {a.seeds} --keep {a.keep} "
            f"--pool {a.pool} > {a.logdir}/{k}.log 2>&1" for k in ks)
        remote = f"cd {DIR} && ( {inner} ) < /dev/null > /dev/null 2>&1 &"
        procs.append(subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", node, remote]))
    for p in procs:
        try: p.wait(timeout=30)
        except Exception: pass
    print(f"launched {sum(1 for b in buckets.values() if b)} slot-loops; logs -> {a.logdir}/")


def _done(path):
    if not os.path.exists(path):
        return False
    try:
        return "err=" in open(path).read() or "ΔG =" in open(path).read()
    except Exception:
        return False


if __name__ == "__main__":
    main()
