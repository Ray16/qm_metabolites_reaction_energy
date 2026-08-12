#!/usr/bin/env python
"""GFN2-xTB single points -> QC features (base/xtb env, needs rdkit + xtb).

Per compound: Mulliken partial charges (heavy atoms, xyz order == rdkit order,
verified), HOMO/LUMO/gap. Writes artifacts/qm_features.json.
Run:  conda run -n xtb python scripts/extract_xtb.py
"""
import _bootstrap  # noqa: F401
import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor

from rdkit import Chem, RDLogger

from gnn import paths

RDLogger.DisableLog("rdApp.*")
XTB = shutil.which("xtb") or "xtb"


def run_one(args):
    cid, smiles, charge = args
    xyz = f"{paths.PIPE}/geometries_tecrdb_full/{cid}/conf_000.xyz"
    if not os.path.exists(xyz):
        return cid, None
    nh = Chem.MolFromSmiles(smiles).GetNumAtoms()
    try:
        with tempfile.TemporaryDirectory() as dtmp:
            shutil.copy(xyz, f"{dtmp}/m.xyz")
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            p = subprocess.run([XTB, "m.xyz", "--gfn", "2", "--chrg", str(int(charge)),
                                "--sp", "--iterations", "500"],
                               cwd=dtmp, env=env, capture_output=True, text=True, timeout=300)
            out = p.stdout
            chg = [float(x) for x in open(f"{dtmp}/charges").read().split()] \
                if os.path.exists(f"{dtmp}/charges") else []
            h = re.findall(r"([\-\d.]+)\s*\(HOMO\)", out)
            l = re.findall(r"([\-\d.]+)\s*\(LUMO\)", out)
            g = re.search(r"HOMO-LUMO GAP\s+([\-\d.]+)\s+eV", out)
            return cid, {"mulliken": chg[:nh], "homo": float(h[-1]) if h else None,
                         "lumo": float(l[-1]) if l else None,
                         "gap": float(g.group(1)) if g else None, "nheavy": nh}
    except Exception as e:
        return cid, {"error": str(e)}


def main():
    mets = json.load(open(f"{paths.PIPE}/tecrdb_full_metabolites.json"))
    jobs = [(m["id"], m["smiles"], m.get("charge", 0)) for m in mets]
    out = {}
    with ProcessPoolExecutor(max_workers=min(40, os.cpu_count())) as ex:
        for i, (cid, res) in enumerate(ex.map(run_one, jobs), 1):
            out[cid] = res
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    json.dump(out, open(paths.artifact("qm_features.json"), "w"))
    ok = sum(1 for v in out.values() if v and v.get("mulliken"))
    print(f"done: {ok}/{len(jobs)} with charges -> artifacts/qm_features.json")


if __name__ == "__main__":
    main()
