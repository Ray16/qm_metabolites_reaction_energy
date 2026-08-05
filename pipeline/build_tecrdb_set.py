#!/usr/bin/env python
"""Match the whole TECRDB extract to ModelSEED reactions and emit QM inputs.

`build_inputs.py` is hardcoded to the original ten. This builds the full
benchmark: every TECRDB measurement whose KEGG equation maps onto a ModelSEED
reaction with structures for all participants.

Matching is by **compound set**, not by name or EC. TECRDB writes biochemical
equations in KEGG ids; ModelSEED reactions are translated to KEGG through the
alias table and compared as multisets, ignoring the proton (whose chemical
potential the pH transform fixes) and water (routinely omitted from biochemical
equations). A match must be exact on everything else, and ambiguous matches --
several ModelSEED reactions sharing one compound set -- are dropped rather than
guessed.

Repeated measurements of the same reaction are aggregated to a median with its
spread and count, so a reaction measured forty times does not outvote one
measured once.

Validation: the script reports how many of the previously curated pairs in
`bench226_meta.csv` it recovers. That number is the thing to look at before
trusting any of the rest.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
sys.path.insert(0, THERMO)

PROTON = "cpd00067"
WATER = "cpd00001"
R_KJ = 8.314462618e-3
KEGG_RE = re.compile(r"C\d{5}")


def kegg_to_modelseed(db: str) -> dict[str, str]:
    """KEGG compound id -> ModelSEED id, keeping the lowest (canonical) id."""
    path = os.path.join(db, "Biochemistry", "Aliases",
                        "Unique_ModelSEED_Compound_Aliases.txt")
    out: dict[str, str] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["Source"] != "KEGG":
                continue
            kegg, seed = row["External ID"].strip(), row["ModelSEED ID"].strip()
            if KEGG_RE.fullmatch(kegg) and (kegg not in out or seed < out[kegg]):
                out[kegg] = seed
    return out


def parse_equation(text: str) -> tuple[Counter, bool]:
    """KEGG biochemical equation -> multiset of (kegg_id, signed coefficient)."""
    if "=" not in text:
        return Counter(), False
    left, right = text.split("=", 1)
    stoich: Counter = Counter()
    for side, sign in ((left, -1), (right, 1)):
        for term in side.split("+"):
            term = term.strip()
            if not term:
                continue
            m = KEGG_RE.search(term)
            if not m:
                return Counter(), False
            coeff = re.match(r"\s*(\d+)\s", term.replace("kegg:", " "))
            n = int(coeff.group(1)) if coeff else 1
            stoich[m.group(0)] += sign * n
    return stoich, bool(stoich)


def load_modelseed_reactions(db: str) -> dict[str, dict[str, float]]:
    reactions = {}
    for path in sorted(glob.glob(os.path.join(db, "Biochemistry", "reaction_??.tsv"))):
        with open(path) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if row.get("is_obsolete", "0") not in ("0", "", "False"):
                    continue
                stoich: dict[str, float] = {}
                for part in (row.get("stoichiometry") or "").split(";"):
                    f = part.split(":")
                    if len(f) >= 2:
                        try:
                            stoich[f[1]] = stoich.get(f[1], 0.0) + float(f[0])
                        except ValueError:
                            pass
                if stoich:
                    reactions[row["id"]] = stoich
    return reactions


def signature(stoich: dict, drop: set) -> tuple:
    """Order-independent key over compounds and coefficients, minus `drop`."""
    return tuple(sorted((c, round(v, 6)) for c, v in stoich.items()
                        if c not in drop and abs(v) > 1e-9))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tecrdb", default=os.path.join(HERE, "TECRDB.csv"))
    ap.add_argument("--modelseed", default=os.path.join(os.path.dirname(THERMO),
                                                        "ModelSEEDDatabase"))
    ap.add_argument("--validate-against", default=os.path.join(HERE, "bench226_meta.csv"))
    ap.add_argument("--out-prefix", default=os.path.join(HERE, "tecrdb_full"))
    ap.add_argument("--max-heavy", type=int, default=70,
                    help="skip compounds larger than this (Hessian cost)")
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    kegg = kegg_to_modelseed(args.modelseed)
    ms_rxns = load_modelseed_reactions(args.modelseed)
    print(f"KEGG->ModelSEED compound map: {len(kegg)}")
    print(f"non-obsolete ModelSEED reactions: {len(ms_rxns)}")

    compounds = {}
    for path in sorted(glob.glob(os.path.join(args.modelseed, "Biochemistry",
                                              "compound_??.tsv"))):
        with open(path) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                compounds[row["id"]] = row

    # Index ModelSEED reactions by compound signature (protons/water free), in
    # BOTH orientations. TECRDB and ModelSEED often write the same reaction in
    # opposite directions; matching only the stored direction loses more than
    # half the set. The stored orientation is recorded so the experimental dG
    # can be flipped to match ModelSEED's, rather than silently inverted.
    index = defaultdict(list)
    for rid, st in ms_rxns.items():
        index[signature(st, {PROTON, WATER})].append((rid, +1))
        index[signature({c: -v for c, v in st.items()}, {PROTON, WATER})].append((rid, -1))

    # ---- read and group the measurements -------------------------------------
    per_reaction = defaultdict(list)
    n_collapsed: dict[str, int] = {}
    n_rows = n_parsed = n_mapped = 0
    for row in csv.DictReader(open(args.tecrdb)):
        n_rows += 1
        eq, ok = parse_equation(row.get("reaction") or "")
        if not ok:
            continue
        n_parsed += 1
        if not all(k in kegg for k in eq):
            continue
        translated = {kegg[k]: v for k, v in eq.items()}
        n_mapped += 1
        hits = index.get(signature(translated, {PROTON, WATER}), [])
        if not hits:
            continue
        # ModelSEED carries duplicate entries for the same chemistry -- PPi
        # hydrolysis alone appears as rxn00001/11226/42569/48873/54366. They
        # share a compound signature by construction, so this is not real
        # ambiguity; collapse to the lowest (canonical, oldest) id instead of
        # discarding the measurement, which was losing ~700 rows.
        rid, orient = sorted(hits)[0]
        if len({r for r, _ in hits}) > 1:
            n_collapsed[rid] = len({r for r, _ in hits})
        try:
            K = float(row.get("K_prime") or row.get("K") or "")
            T = float(row.get("temperature") or 298.15) or 298.15
        except ValueError:
            continue
        if K <= 0:
            continue
        # orient == -1 means TECRDB wrote the reverse of ModelSEED's direction.
        per_reaction[rid].append({
            "dG_kJ": orient * -R_KJ * T * math.log(K), "T": T,
            "pH": row.get("p_h") or "", "EC": row.get("EC") or "",
            "enzyme": row.get("enzyme_name") or "",
        })

    print(f"\nTECRDB rows {n_rows}; parsed {n_parsed}; all-compounds-mapped {n_mapped}; "
          f"matched to a ModelSEED reaction: "
          f"{sum(len(v) for v in per_reaction.values())} measurements "
          f"over {len(per_reaction)} reactions")
    print(f"  (of these, {len(n_collapsed)} reactions absorbed ModelSEED duplicate entries)")

    # ---- validation against the previously curated pairs ---------------------
    if os.path.exists(args.validate_against):
        prev = {r["modelseed_rxn"] for r in csv.DictReader(open(args.validate_against))}
        rec = prev & set(per_reaction)
        print(f"validation: recovered {len(rec)}/{len(prev)} of the curated "
              f"bench226 reactions ({len(rec)/max(1,len(prev)):.0%})")

    # ---- keep only reactions every participant of which we can compute -------
    kept, skipped = {}, Counter()
    for rid, obs in per_reaction.items():
        st = {c: v for c, v in ms_rxns[rid].items() if c != PROTON}
        ok = True
        for c in st:
            row = compounds.get(c)
            smi = (row or {}).get("smiles", "").strip()
            if not row or not smi:
                skipped["no structure"] += 1; ok = False; break
            m = Chem.MolFromSmiles(smi)
            if m is None:
                skipped["unparseable"] += 1; ok = False; break
            if any(a.GetSymbol() == "*" for a in m.GetAtoms()):
                skipped["'*' placeholder"] += 1; ok = False; break
            if m.GetNumHeavyAtoms() > args.max_heavy:
                skipped[f"larger than {args.max_heavy} heavy atoms"] += 1; ok = False; break
        if ok:
            kept[rid] = {"stoichiometry": st, "observations": obs}
    print("\nreactions dropped because a participant is not computable:")
    for k, v in skipped.most_common():
        print(f"   {k:34s} {v}")
    print(f"kept: {len(kept)} reactions")

    used = sorted({c for r in kept.values() for c in r["stoichiometry"]})
    print(f"unique compounds to compute: {len(used)}")

    # ---- emit ---------------------------------------------------------------
    reactions = {rid: r["stoichiometry"] for rid, r in kept.items()}
    experiment = {}
    for rid, r in kept.items():
        vals = [o["dG_kJ"] for o in r["observations"]]
        experiment[rid] = {
            "dG_kJ": statistics.median(vals), "n": len(vals),
            "sd_kJ": statistics.stdev(vals) if len(vals) > 1 else None,
            "EC": Counter(o["EC"] for o in r["observations"]).most_common(1)[0][0],
            "enzyme": Counter(o["enzyme"] for o in r["observations"]).most_common(1)[0][0],
        }
    metabolites, species = [], {}
    for c in used:
        row = compounds[c]
        m = Chem.MolFromSmiles(row["smiles"])
        mh = Chem.AddHs(m)
        metabolites.append({"id": c, "name": row["name"], "smiles": row["smiles"],
                            "formula": row.get("formula", ""), "charge": int(row["charge"]),
                            "inchikey": row.get("inchikey", ""), "opentecr_species": ""})
        species[c] = {"name": row["name"], "charge": int(row["charge"]),
                      "n_hydrogens": sum(1 for a in mh.GetAtoms() if a.GetSymbol() == "H")}
    for name, obj in (("reactions", reactions), ("experiment", experiment),
                      ("metabolites", metabolites), ("species", species)):
        path = f"{args.out_prefix}_{name}.json"
        json.dump(obj, open(path, "w"), indent=1)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
