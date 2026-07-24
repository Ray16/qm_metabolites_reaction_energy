#!/usr/bin/env python
"""Order metabolites so the benchmark becomes scoreable as early as possible.

Metabolite frequency in TECRDB is hub-structured -- water, ATP, ADP, Pi, NAD(P),
CoA, glutamate appear in a large fraction of reactions. Building alphabetically
means ~a day of compute during which zero reactions are scoreable. A greedy
set-cover ordering instead maximises "reactions fully covered" after every
compound, so there is a usable benchmark of growing size throughout.

Greedy criterion: at each step take the metabolite that unlocks the most
reactions per unit cost, where cost ~ heavy-atom count (xtb time scales steeply
with size). Ties broken by raw reaction frequency.

Also emits the compounds already built, so the big run skips them.

Run:  /homes/rzhu/miniforge3/envs/palm/bin/python order_by_coverage.py
"""
import json
import os
from collections import Counter

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    mets = {m["id"]: m for m in json.load(open(os.path.join(HERE, "bench802_metabolites.json")))}
    rx = json.load(open(os.path.join(HERE, "bench802_reactions.json")))

    built = set()
    for f in ("bench226_xtb.json", "ensemble_deep_xtb.json"):
        p = os.path.join(HERE, f)
        if os.path.isfile(p):
            built |= set(json.load(open(p)))
    built &= set(mets)

    size = {c: Chem.MolFromSmiles(m["smiles"]).GetNumHeavyAtoms()
            for c, m in mets.items()}
    freq = Counter(c for st in rx.values() for c in st)

    have = set(built)
    remaining = set(mets) - have
    order, covered_curve = [], []

    def covered(have_set):
        return sum(1 for st in rx.values() if set(st) <= have_set)

    n0 = covered(have)
    print(f"already built: {len(built)} metabolites -> {n0}/{len(rx)} reactions scoreable")

    while remaining:
        best, best_score = None, None
        for c in remaining:
            gain = covered(have | {c}) - covered(have)
            # gain per unit cost; frequency breaks ties among zero-gain hubs
            score = (gain / max(1, size[c]), gain, freq[c])
            if best_score is None or score > best_score:
                best, best_score = c, score
        have.add(best)
        remaining.discard(best)
        order.append(best)
        covered_curve.append(covered(have))

    json.dump(dict(already_built=sorted(built), order=order,
                   coverage_after_each=covered_curve),
              open(os.path.join(HERE, "bench802_build_order.json"), "w"), indent=1)

    print(f"\nordered {len(order)} remaining metabolites")
    print(f"{'after N new':>12}{'reactions scoreable':>22}{'cum heavy atoms':>18}")
    cum = 0
    for k in (10, 25, 50, 100, 150, 200, len(order)):
        if k > len(order):
            continue
        cum = sum(size[c] for c in order[:k])
        print(f"{k:12d}{covered_curve[k-1]:22d}{cum:18d}")
    print("\nwrote bench802_build_order.json")


if __name__ == "__main__":
    main()
