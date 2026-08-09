#!/usr/bin/env python
"""CPCM-X dGsolv at fixed geometry for the full-TECRDB conformer ensemble.

Physics-first solvation routing experiment: swap ALPB -> CPCM-X per species so
we can evaluate, post-hoc, whether routing the CPCM-X-favourable groups
(carboxylate/phenol/thiol) improves reaction dG. Phosphates are skipped by
default (CPCM-X collapses on them; see FINDINGS.md).

Writes {cpd: [dGsolv_kJ per conformer]} so the scorer can substitute per policy.
"""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

HERE = os.path.dirname(os.path.abspath(__file__))
XTB = os.environ.get("XTB_CPCMX", "/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/xtbcpx/bin/xtb")
SCRATCH = "/tmp/qm_thermo_scratch/cpcmx_tecrdb"
HARTREE_TO_KJ = 2625.499639
_LOCK = Lock()


def cpcmx_dgsolv_kJ(xyz_path, chg, omp=2):
    wd = tempfile.mkdtemp(prefix="cpx_", dir=SCRATCH)
    try:
        shutil.copy(xyz_path, os.path.join(wd, "in.xyz"))
        env = {**os.environ, "OMP_NUM_THREADS": str(omp), "OMP_STACKSIZE": "4G",
               "OPENBLAS_NUM_THREADS": "1"}
        subprocess.run([XTB, "in.xyz", "--gfn", "2", "--chrg", str(int(chg)),
                        "--uhf", "0", "--cpcmx", "water"],
                       cwd=wd, capture_output=True, text=True, env=env, timeout=600)
        f6 = os.path.join(wd, "fort.6")
        if not os.path.isfile(f6):
            return None
        m = re.search(r"solvation free energy \(dG_solv\):\s+(-?\d+\.\d+E[+-]\d+)",
                      open(f6, errors="replace").read())
        return float(m.group(1)) * HARTREE_TO_KJ if m else None
    except Exception:
        return None
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens", default=os.path.join(HERE, "ensemble_tecrdb_full.json"))
    ap.add_argument("--mets", default=os.path.join(HERE, "tecrdb_full_metabolites.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "cpcmx_dgsolv_tecrdb_full.json"))
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--omp", type=int, default=1)
    ap.add_argument("--skip-phosphorus", action="store_true", default=True)
    args = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)

    ens = json.load(open(args.ens))
    mets = {m["id"]: m for m in json.load(open(args.mets))}
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    done = json.load(open(args.out)) if os.path.isfile(args.out) else {}
    jobs = []
    for c, confs in ens.items():
        if c in done:
            continue
        smi = mets.get(c, {}).get("smiles", "")
        mol = Chem.MolFromSmiles(smi) if smi else None
        if args.skip_phosphorus and mol is not None and any(a.GetSymbol() == "P" for a in mol.GetAtoms()):
            continue
        chg = int(mets[c]["charge"])
        for i, cf in enumerate(confs):
            p = cf["xyz"] if os.path.isabs(cf["xyz"]) else os.path.join(HERE, cf["xyz"])
            jobs.append((c, i, p, chg, len(confs)))

    print(f"=== CPCM-X | {len(jobs)} conformers to run | {args.workers}w x {args.omp}omp ===", flush=True)
    results = {}
    for c, i, p, q, n in jobs:
        results.setdefault(c, [None] * n)
    n_done = [0]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(cpcmx_dgsolv_kJ, p, q, args.omp): (c, i) for c, i, p, q, n in jobs}
        for fut in as_completed(futs):
            c, i = futs[fut]
            results[c][i] = fut.result()
            with _LOCK:
                n_done[0] += 1
                if n_done[0] % 100 == 0:
                    print(f"  {n_done[0]}/{len(jobs)}", flush=True)
    done.update(results)
    json.dump(done, open(args.out, "w"), indent=1)
    nfail = sum(1 for v in done.values() for x in v if x is None)
    print(f"wrote {args.out}: {len(done)} species, {nfail} failed conformers", flush=True)


if __name__ == "__main__":
    main()
