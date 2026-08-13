#!/usr/bin/env python
"""Emit per-reaction measurement conditions for the 367 TECRDB reactions.

Reuses the exact KEGG->ModelSEED compound-set matching from
`pipeline/build_tecrdb_set.py`, so a raw TECRDB.csv row maps to the same
ModelSEED reaction id that produced `n`/`sd_kJ` in tecrdb_full_experiment.json.
For each reaction we take the MEDIAN of pH, ionic strength, temperature and pMg
over its measurements (blanks ignored; None if a field is never reported).

Output: artifacts/rxn_conditions.json  { rxn_id: {p_h, ionic_strength,
temperature, p_mg} }, restricted to the reactions already in the experiment set
so it aligns 1:1 with data.pt's rxn_ids.  No rdkit / no GPU needed.
"""
import _bootstrap  # noqa: F401
import csv
import json
import statistics
import sys
from collections import defaultdict

from gnn import paths

sys.path.insert(0, paths.PIPE)
from build_tecrdb_set import (  # noqa: E402
    PROTON, WATER, kegg_to_modelseed, load_modelseed_reactions, parse_equation,
    signature)

MODELSEED = paths.DB.rsplit("/Biochemistry", 1)[0]
FIELDS = ("p_h", "ionic_strength", "temperature", "p_mg")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    kegg = kegg_to_modelseed(MODELSEED)
    ms_rxns = load_modelseed_reactions(MODELSEED)
    index = defaultdict(list)
    for rid, st in ms_rxns.items():
        index[signature(st, {PROTON, WATER})].append(rid)
        index[signature({c: -v for c, v in st.items()}, {PROTON, WATER})].append(rid)

    keep = set(json.load(open(f"{paths.PIPE}/tecrdb_full_experiment.json")))
    obs = defaultdict(lambda: {k: [] for k in FIELDS})
    for row in csv.DictReader(open(f"{paths.PIPE}/TECRDB.csv")):
        eq, ok = parse_equation(row.get("reaction") or "")
        if not ok or not all(k in kegg for k in eq):
            continue
        hits = index.get(signature({kegg[k]: v for k, v in eq.items()}, {PROTON, WATER}), [])
        if not hits:
            continue
        rid = sorted(hits)[0]           # canonical id, same rule as build_tecrdb_set
        if rid not in keep:
            continue
        for k in FIELDS:
            v = _f(row.get(k))
            if v is not None:
                obs[rid][k].append(v)

    out = {}
    for rid in keep:
        d = obs.get(rid, {})
        out[rid] = {k: (statistics.median(d[k]) if d.get(k) else None) for k in FIELDS}
    json.dump(out, open(paths.artifact("rxn_conditions.json"), "w"), indent=1)

    have = {k: sum(1 for r in out.values() if r[k] is not None) for k in FIELDS}
    print(f"wrote rxn_conditions.json for {len(out)} reactions "
          f"(mapped {len(obs)}); non-null per field: {have}")


if __name__ == "__main__":
    main()
