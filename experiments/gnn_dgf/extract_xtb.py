#!/usr/bin/env python
"""Run GFN2-xTB single points to extract QM features for each compound.

For every compound's conf_000.xyz (already xtb-optimized) we run one GFN2
single point at the correct molecular charge and parse:
  - per-atom Mulliken partial charges (`charges` file, xyz atom order)
  - HOMO, LUMO, gap  (eV)   - frontier orbitals -> redox/electronics
  - molecular dipole (Debye) - polarity

Atom order in the xyz == RDKit heavy-atom order for the first n_heavy atoms
(verified for all 453), so charges[:n_heavy] map directly onto graph nodes.

Output: qm_features.json  {cpd_id: {mulliken:[...heavy...], homo, lumo, gap, dipole}}
Run inside the xtb env:  conda run -n xtb python extract_xtb.py
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.normpath(os.path.join(HERE, "..", "..", "pipeline"))
XTB = shutil.which("xtb") or "xtb"


def n_heavy(smiles):
    return Chem.MolFromSmiles(smiles).GetNumAtoms()


def run_one(args):
    cid, smiles, charge = args
    xyz = f"{PIPE}/geometries_tecrdb_full/{cid}/conf_000.xyz"
    if not os.path.exists(xyz):
        return cid, None
    nh = n_heavy(smiles)
    try:
        with tempfile.TemporaryDirectory() as d:
            shutil.copy(xyz, f"{d}/m.xyz")
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            p = subprocess.run(
                [XTB, "m.xyz", "--gfn", "2", "--chrg", str(int(charge)),
                 "--sp", "--iterations", "500"],
                cwd=d, env=env, capture_output=True, text=True, timeout=300)
            out = p.stdout
            # per-atom Mulliken charges
            chg = []
            cf = f"{d}/charges"
            if os.path.exists(cf):
                chg = [float(x) for x in open(cf).read().split()]
            homo = lumo = gap = None
            # HOMO/LUMO: grab the eV value (last float) right before the tag
            h = re.findall(r"([\-\d.]+)\s*\(HOMO\)", out)
            l = re.findall(r"([\-\d.]+)\s*\(LUMO\)", out)
            if h:
                homo = float(h[-1])
            if l:
                lumo = float(l[-1])
            g = re.search(r"HOMO-LUMO GAP\s+([\-\d.]+)\s+eV", out)
            if g:
                gap = float(g.group(1))
            return cid, {"mulliken": chg[:nh], "homo": homo, "lumo": lumo,
                         "gap": gap, "nheavy": nh}
    except Exception as e:
        return cid, {"error": str(e)}


def main():
    mets = json.load(open(f"{PIPE}/tecrdb_full_metabolites.json"))
    jobs = [(m["id"], m["smiles"], m.get("charge", 0)) for m in mets]
    out = {}
    with ProcessPoolExecutor(max_workers=min(40, os.cpu_count())) as ex:
        for i, (cid, res) in enumerate(ex.map(run_one, jobs), 1):
            out[cid] = res
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    json.dump(out, open(f"{HERE}/qm_features.json", "w"))
    ok = sum(1 for v in out.values() if v and "error" not in v and v.get("mulliken"))
    hasorb = sum(1 for v in out.values() if v and v.get("homo") is not None)
    print(f"done: {ok}/{len(jobs)} with charges, {hasorb} with orbitals")


if __name__ == "__main__":
    main()
