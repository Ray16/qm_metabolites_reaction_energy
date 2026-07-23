#!/usr/bin/env python
"""Expand undefined stereocentres into explicit stereoisomers before ensembling.

ModelSEED serves several sugars and sugar-phosphates with an UNDEFINED anomeric
carbon (e.g. glucose-6-phosphate, ribose, ribose-5-phosphate). RDKit then embeds
one arbitrary anomer, so the pipeline silently scores a single member of a
mixture. In water these interconvert by mutarotation, so the correct object is
the equilibrium pool, not a chosen anomer.

The fix needs no new machinery: enumerate the stereoisomers, put them all in the
compound's conformer list, and the existing log-sum-exp Boltzmann average
    G = -RT ln sum_i exp(-G_i/RT)
returns exactly the pool free energy, with the anomeric ratio emerging from the
computed energies rather than being assumed.

Only CARBON centres are expanded. Phosphorus in a phosphate ester is flagged by
RDKit as a potential stereocentre because it formally carries four different
substituents, but the P-O(-) oxygens are equivalent by resonance, so those are
not real centres and expanding them would multiply the cost for nothing.

Run:  /homes/rzhu/miniforge3/envs/palm/bin/python expand_stereoisomers.py \
          --mets bench226_metabolites.json --out bench226_metabolites_stereo.json
"""
import argparse
import json
import os

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import (EnumerateStereoisomers,
                                               StereoEnumerationOptions)
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
MAX_ISOMERS = 8          # 2^3; beyond that the compound needs curation, not enumeration


def undefined_carbon_centres(mol):
    return [i for i, t in Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False)
        if t == "?" and mol.GetAtomWithIdx(i).GetSymbol() == "C"]


def expand(smiles):
    """Return list of stereo-resolved SMILES (the input itself if nothing to do)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    und = undefined_carbon_centres(mol)
    if not und:
        return [smiles]
    if 2 ** len(und) > MAX_ISOMERS:
        return [smiles]                      # too ambiguous; leave and flag
    # freeze the already-defined centres, enumerate only the undefined ones
    opts = StereoEnumerationOptions(onlyUnassigned=True, unique=True,
                                    maxIsomers=MAX_ISOMERS)
    out = []
    for iso in EnumerateStereoisomers(mol, options=opts):
        s = Chem.MolToSmiles(iso)
        if s not in out:
            out.append(s)
    return out or [smiles]


def merge_variants(ens_path, out_path):
    """Fold cpdXXXXX#sK conformer lists back under their parent id.

    The scorer then sees one compound whose conformer list spans both anomers,
    and its existing Boltzmann average returns the mutarotation-equilibrium pool
    free energy -- no scorer change required, and the anomeric ratio falls out
    of the computed energies.
    """
    ens = json.load(open(ens_path))
    merged, counts = {}, {}
    for cid, confs in ens.items():
        parent = cid.split("#s")[0]
        merged.setdefault(parent, [])
        counts[parent] = counts.get(parent, 0) + 1
        for cf in confs:
            r = dict(cf)
            r["stereo_variant"] = cid
            r["conf"] = len(merged[parent])
            merged[parent].append(r)
    json.dump(merged, open(out_path, "w"), indent=1)
    pooled = {k: v for k, v in counts.items() if v > 1}
    print(f"merged {len(ens)} entries -> {len(merged)} compounds; "
          f"{len(pooled)} pooled across stereoisomers")
    for k, v in pooled.items():
        print(f"   {k}: {v} variants, {len(merged[k])} conformers total")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", nargs=2, metavar=("ENS_IN", "ENS_OUT"),
                    help="fold #sK stereo variants back under the parent id")
    ap.add_argument("--mets", default="bench226_metabolites.json")
    ap.add_argument("--out", default="bench226_metabolites_stereo.json")
    args = ap.parse_args()

    if args.merge:
        merge_variants(os.path.join(HERE, args.merge[0]),
                       os.path.join(HERE, args.merge[1]))
        return

    mets = json.load(open(os.path.join(HERE, args.mets)))
    out, expanded, skipped = [], 0, []
    for m in mets:
        variants = expand(m["smiles"])
        mol = Chem.MolFromSmiles(m["smiles"])
        n_und = len(undefined_carbon_centres(mol)) if mol else 0
        if len(variants) == 1:
            if n_und:
                skipped.append((m["id"], n_und))
            out.append(m)
            continue
        expanded += 1
        print(f"{m['id']:11} {n_und} undefined C -> {len(variants)} stereoisomers")
        for k, s in enumerate(variants):
            r = dict(m)
            r["id"] = f"{m['id']}#s{k}"       # stereo-variant id
            r["smiles"] = s
            r["parent"] = m["id"]
            out.append(r)

    json.dump(out, open(os.path.join(HERE, args.out), "w"), indent=1)
    print(f"\n{expanded} compounds expanded; {len(mets)} -> {len(out)} entries")
    if skipped:
        print("too ambiguous to enumerate (needs curation):",
              ", ".join(f"{c}({n})" for c, n in skipped))
    print(f"wrote {args.out}")
    print("\nNOTE: the scorer must Boltzmann-average the variants of one parent "
          "together, so the anomeric/epimeric ratio comes out of the energies.")


if __name__ == "__main__":
    main()
