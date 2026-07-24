#!/usr/bin/env python
"""Isodesmic (reference-reaction) prediction of dG'^o.

The absolute formulation fails because each species carries 10-20 kJ/mol of
solvation/RRHO error and four of them accumulate as sqrt(4)*15 ~ 30 kJ/mol.
That error is largely PER-SPECIES, which the data shows directly: rxn00213 and
rxn01675 both contain glucose-1-phosphate and their QM errors agree to 1.6
kJ/mol, while rxn01005 -- same reaction class but no G1P -- differs by 41.

So instead of predicting dG absolutely, predict it relative to a reference
reaction with a measured value:

    dG(target) = dG_exp(reference) + [ dG_QM(target) - dG_QM(reference) ]

The QM error becomes err(target) - err(reference), which cancels to the extent
the two reactions share species. Choosing the reference to maximise shared
species -- especially the badly-modelled ones -- is therefore the whole game.

This is not circular: the reference is a DIFFERENT measured reaction, and QM
supplies the transformation between them. It is the same logic that makes group
contribution work, but with the group additivity replaced by explicit quantum
chemistry on the difference.

Reference selection scores candidates by shared species weighted toward large,
highly charged compounds (where the per-species error is biggest), and requires
the candidate to have both an experimental value and a QM score.

Run:  /homes/rzhu/miniforge3/envs/palm/bin/python isodesmic.py
"""
import csv
import json
import os

import numpy as np
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    scored = {x["r"]: x for x in json.load(open(os.path.join(HERE, "bench226_scored.json")))}
    rx = json.load(open(os.path.join(HERE, "bench802_reactions.json")))
    meta = {r["modelseed_rxn"]: r
            for r in csv.DictReader(open(os.path.join(HERE, "bench802_meta.csv")))}
    mets = {m["id"]: m for m in json.load(open(os.path.join(HERE, "bench226_metabolites.json")))}
    return scored, rx, meta, mets


def species_weight(mets, c):
    """Weight a shared species by how much error it is likely to carry.

    Per-species error grows with size (more conformers, bigger solvation term)
    and with charge (continuum models fail hardest on polyanions), so sharing a
    large polyanion cancels far more error than sharing water.
    """
    m = mets.get(c)
    if m is None:
        return 1.0
    mol = Chem.MolFromSmiles(m["smiles"])
    n = mol.GetNumHeavyAtoms() if mol else 1
    q = abs(int(m["charge"]))
    return (1.0 + n / 10.0) * (1.0 + q)


def match_score(rx, mets, target, cand):
    """Weighted overlap; species must appear with the SAME sign to cancel."""
    A, B = rx[target], rx[cand]
    num = 0.0
    for c, v in A.items():
        if c in B and np.sign(B[c]) == np.sign(v):
            num += species_weight(mets, c) * min(abs(v), abs(B[c]))
    den = sum(species_weight(mets, c) * abs(v) for c, v in A.items())
    return num / den if den else 0.0


def predict(target, scored, rx, meta, mets, exclude=()):
    """Best isodesmic prediction for one target, and the reference used."""
    best = None
    for cand in scored:
        if cand == target or cand in exclude:
            continue
        # a reference must not be the same measurement re-entered
        if meta[cand]["tecrdb_reaction"] == meta[target]["tecrdb_reaction"]:
            continue
        s = match_score(rx, mets, target, cand)
        if s <= 0:
            continue
        if best is None or s > best[0]:
            best = (s, cand)
    if best is None:
        return None
    s, ref = best
    dG = (float(meta[ref]["tecrdb_dG_kJ"])
          + (scored[target]["q"] - scored[ref]["q"]))
    return dict(ref=ref, score=s, pred=dG,
                exp=float(meta[target]["tecrdb_dG_kJ"]))


def main():
    scored, rx, meta, mets = load()
    rows = []
    for t in scored:
        p = predict(t, scored, rx, meta, mets)
        if p is None:
            continue
        p["target"] = t
        p["err_iso"] = p["pred"] - p["exp"]
        p["err_abs"] = scored[t]["err"]
        rows.append(p)

    A = lambda k: np.mean([abs(r[k]) for r in rows])
    print(f"n = {len(rows)} reactions with an available reference\n")
    print(f"  absolute formulation   MAE {A('err_abs'):6.1f}")
    print(f"  isodesmic (best ref)   MAE {A('err_iso'):6.1f}")
    z = np.mean([abs(r["exp"]) for r in rows])
    print(f"  predict-zero           MAE {z:6.1f}")

    print("\nby reference quality (weighted overlap score):")
    print(f"{'score':>10}{'n':>6}{'MAE iso':>10}{'MAE abs':>10}")
    for lo, hi in ((0.0, .25), (.25, .5), (.5, .75), (.75, 1.01)):
        g = [r for r in rows if lo <= r["score"] < hi]
        if len(g) >= 5:
            print(f"{lo:.2f}-{hi:.2f}{len(g):6d}"
                  f"{np.mean([abs(r['err_iso']) for r in g]):10.1f}"
                  f"{np.mean([abs(r['err_abs']) for r in g]):10.1f}")

    json.dump(rows, open(os.path.join(HERE, "isodesmic_results.json"), "w"), indent=1)
    print("\nwrote isodesmic_results.json")


if __name__ == "__main__":
    main()
