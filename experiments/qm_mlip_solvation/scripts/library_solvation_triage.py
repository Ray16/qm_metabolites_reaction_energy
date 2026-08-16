#!/usr/bin/env python
"""Library-wide triage: which ModelSEED compounds need EXPLICIT solvation?

Continuum solvation fails where solvation is dominated by specific, directional,
short-range H-bonds to a LOCALIZED, HIGH-CHARGE-DENSITY site. So we scan the pH-7
(ChemAxon "Charged") SMILES for anionic/cationic functional groups and bin each
compound by the physics that decides explicit-water need:

  NEED_EXPLICIT : hard, localized, high-density anions the continuum over-solvates
                  -> phosphate (esp. compact poly-P like PPi), carboxylate,
                     sulfonate/sulfate, alkoxide/enolate.
  BORDERLINE    : soft/diffuse or delocalized -> thiolate (soft S-), phenolate
                  (charge into ring), ammonium/guanidinium cations. May be implicit-OK.
  IMPLICIT_OK   : neutral / no localized ionic site.

Also counts compact POLYANIONS (>=2 anionic sites, the PPi-hardest class) and picks a
small representative per group for the calibration ladder (smallest heavy-atom molecule
carrying that group as its dominant feature).

Run (uma env): python scripts/library_solvation_triage.py
"""
import csv
import json
import os
import sys
from collections import defaultdict

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

STRUCT = ("/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/ModelSEEDDatabase/"
          "Biochemistry/Structures/All_ModelSEED_Structures_updated.txt")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")

# SMARTS for pH-7 ionic groups. Order = priority (first match wins for "dominant group").
GROUPS = [
    ("phosphate",   "[PX4](=O)([O-])",                 "NEED_EXPLICIT"),   # includes phospho-esters/anhydrides
    ("sulfonate",   "[SX4](=O)(=O)[O-]",               "NEED_EXPLICIT"),
    ("sulfate",     "[OX2][SX4](=O)(=O)[O-]",          "NEED_EXPLICIT"),
    ("carboxylate", "[CX3](=O)[O-]",                   "NEED_EXPLICIT"),
    ("alkoxide",    "[CX4][O-]",                       "NEED_EXPLICIT"),   # aliphatic O- (rare at pH7, high density)
    ("enolate",     "[CX3]=[CX3][O-]",                 "NEED_EXPLICIT"),
    ("phenolate",   "[c][O-]",                         "BORDERLINE"),      # delocalized into ring
    ("thiolate",    "[#16X1-]",                        "BORDERLINE"),      # soft S-
    ("guanidinium", "[NX3][CX3](=[NX3+])[NX3]",        "BORDERLINE"),
    ("ammonium",    "[NX4+]",                           "BORDERLINE"),
    ("aromatic_N+", "[n+]",                             "BORDERLINE"),
]
ANIONIC_SMARTS = "[O-,S-,N-]"   # count localized anionic sites for polyanion compactness


def load_charged_smiles(path):
    """cpd_id -> (charge, smiles) for the first pH-7 'Charged' SMILE per compound."""
    out = {}
    with open(path) as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) < 8 or row[1] != "SMILE" or row[2] != "Charged":
                continue
            cpd = row[0]
            if cpd in out:
                continue
            try:
                q = int(row[6])
            except ValueError:
                q = None
            out[cpd] = (q, row[7])
    return out


def main():
    log = lambda s: print(s, flush=True)
    smis = load_charged_smiles(STRUCT)
    log(f"loaded {len(smis)} compounds with pH-7 Charged SMILES")
    pats = [(n, Chem.MolFromSmarts(s), tri) for n, s, tri in GROUPS]
    anion_pat = Chem.MolFromSmarts(ANIONIC_SMARTS)

    group_count = defaultdict(int)
    triage_count = defaultdict(int)
    poly_by_group = defaultdict(int)   # compact-polyanion count per dominant group
    reps = {}                       # group -> (heavy_atoms, cpd, smiles)
    polyanion = 0                   # >=2 localized anionic sites (compact-anion risk)
    per_compound = []
    parsed = failed = fragment = 0

    for cpd, (q, smi) in smis.items():
        if "*" in smi:              # R-group / polymer fragment -> unscoreable, skip
            fragment += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            continue
        parsed += 1
        present = [(n, tri) for n, p, tri in pats if p is not None and mol.HasSubstructMatch(p)]
        n_anion = len(mol.GetSubstructMatches(anion_pat))
        is_poly = n_anion >= 2
        if is_poly:
            polyanion += 1
        if not present:
            triage_count["IMPLICIT_OK"] += 1
            per_compound.append((cpd, q, "IMPLICIT_OK", "", n_anion))
            continue
        # dominant group = highest-priority (first in GROUPS order)
        dom_name, dom_tri = present[0]
        for n, tri in present:
            group_count[n] += 1
        if is_poly:
            poly_by_group[dom_name] += 1
        # overall triage: NEED if any NEED group, else BORDERLINE
        triage = "NEED_EXPLICIT" if any(t == "NEED_EXPLICIT" for _, t in present) else "BORDERLINE"
        triage_count[triage] += 1
        per_compound.append((cpd, q, triage, ";".join(n for n, _ in present), n_anion))
        # representative: smallest molecule dominated by this group
        ha = mol.GetNumHeavyAtoms()
        if dom_name not in reps or ha < reps[dom_name][0]:
            reps[dom_name] = (ha, cpd, smi)

    log(f"\nparsed {parsed} scoreable, {fragment} R-group fragments skipped, failed {failed}")
    log(f"\n=== TRIAGE (compound-level, scoreable only) ===")
    tot = parsed
    for t in ["NEED_EXPLICIT", "BORDERLINE", "IMPLICIT_OK"]:
        log(f"  {t:14s}: {triage_count[t]:6d}  ({100*triage_count[t]/tot:.1f}%)")
    log(f"  compact polyanions (>=2 anionic sites): {polyanion} ({100*polyanion/tot:.1f}%)  <- PPi-hardest class")

    log(f"\n=== GROUP COUNTS (a compound can carry several; [poly]=in a compact polyanion) ===")
    for n, _, tri in GROUPS:
        log(f"  {n:14s} [{tri:13s}]: {group_count[n]:6d}   poly-dominant {poly_by_group.get(n,0)}")

    log(f"\n=== CALIBRATION-SET REPRESENTATIVES (smallest per group) ===")
    for n, _, tri in GROUPS:
        if n in reps:
            ha, cpd, smi = reps[n]
            log(f"  {n:14s} [{tri:13s}]: {cpd}  ({ha} heavy)  {smi}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "library_solvation_triage.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["cpd", "charge", "triage", "groups", "n_anionic_sites"])
        w.writerows(per_compound)
    json.dump(dict(n_compounds=parsed, failed=failed,
                   triage={k: triage_count[k] for k in triage_count},
                   groups=dict(group_count), polyanions=polyanion,
                   representatives={n: {"cpd": reps[n][1], "heavy": reps[n][0], "smiles": reps[n][2]}
                                    for n in reps}),
              open(os.path.join(OUT, "library_solvation_triage.json"), "w"), indent=2)
    log(f"\nwrote artifacts/library_solvation_triage.{{tsv,json}}")


if __name__ == "__main__":
    main()
