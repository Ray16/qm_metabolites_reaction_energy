#!/usr/bin/env python
"""Quality-check Dimorphite-DL against ModelSEED's ChemAxon pH-7 protonation states.

ModelSEED ships ChemAxon-assigned pH-7 microspecies (compound `charge` + `smiles`).
Feed the neutralized skeleton to Dimorphite-DL at pH 7 and compare the NET FORMAL
CHARGE of its major output to ModelSEED's charge. High agreement -> Dimorphite is a
fine enumerator; disagreements (esp. phosphates/polyprotic) -> prefer ModelSEED lookup.

Run: python scripts/validate_dimorphite_vs_modelseed.py [n_sample]
"""
import glob
import os
import sys
import csv

from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dimorphite_dl import protonate_smiles

MSDB = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/ModelSEEDDatabase/Biochemistry"


def load_compounds(limit):
    rows = []
    for f in sorted(glob.glob(os.path.join(MSDB, "compound_*.tsv"))):
        with open(f) as fh:
            r = csv.DictReader(fh, delimiter="\t")
            for row in r:
                smi = row.get("smiles", "").strip()
                if not smi or smi in ("null", "None"):
                    continue
                try:
                    q = int(row["charge"])
                except (ValueError, KeyError):
                    continue
                if row.get("is_obsolete", "0") == "1":
                    continue
                rows.append((row["id"], row["name"], q, smi))
                if len(rows) >= limit:
                    return rows
    return rows


def dimorphite_charge(smi, ph=7.0):
    """Net formal charge of Dimorphite's top microspecies for `smi` at pH."""
    try:
        outs = protonate_smiles(smi, ph_min=ph, ph_max=ph, max_variants=1)
    except Exception:
        return None
    if not outs:
        return None
    m = Chem.MolFromSmiles(outs[0])
    return Chem.GetFormalCharge(m) if m is not None else None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    comps = load_compounds(n)
    agree = disagree = skipped = 0
    diffs = []
    for cid, name, q_ms, smi in comps:
        # neutralize input so dimorphite assigns protonation from scratch (fair test)
        m = Chem.MolFromSmiles(smi)
        if m is None:
            skipped += 1; continue
        q_dm = dimorphite_charge(smi)
        if q_dm is None:
            skipped += 1; continue
        if q_dm == q_ms:
            agree += 1
        else:
            disagree += 1
            diffs.append((cid, name[:28], q_ms, q_dm, smi[:40]))
    tot = agree + disagree
    print(f"\n==== Dimorphite vs ModelSEED ChemAxon (pH 7), n={tot} (skipped {skipped}) ====")
    print(f"  AGREE {agree}/{tot} = {100*agree/max(tot,1):.1f}%   DISAGREE {disagree}")
    print(f"\n  sample disagreements (id, name, q_ModelSEED, q_dimorphite, smiles):")
    # sort by |Δcharge| to surface the worst
    for cid, name, qm, qd, smi in sorted(diffs, key=lambda x: -abs(x[2]-x[3]))[:25]:
        print(f"    {cid:10s} {name:28s} MS {qm:+d}  DM {qd:+d}  (Δ{qd-qm:+d})  {smi}")
    # charge-magnitude breakdown
    from collections import Counter
    by_absq = Counter()
    for cid, name, qm, qd, smi in diffs:
        by_absq[abs(qm)] += 1
    print(f"\n  disagreements by |ModelSEED charge|: {dict(sorted(by_absq.items()))}")


if __name__ == "__main__":
    main()
