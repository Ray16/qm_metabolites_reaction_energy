#!/usr/bin/env python
"""Microspecies audit: does our pipeline's pH-7 protonation state (charge) match
Du's experimentally-grounded dominant species at pH 7?

A wrong dominant microspecies makes the whole conformer ensemble / energy meaningless,
so this is checked BEFORE investing in better conformer search. Match by InChIKey
connectivity block (protonation-tolerant), compare our charge vs Du's pH-7 charge.
"""
import csv, json, re, os
from collections import defaultdict
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS"
TC = f"{ROOT}/thermodynamic_calc"
RAW = f"{TC}/data/external/du2018_group_contribution/raw"
COMP = f"{RAW}/TECRDB_compounds_data.csv"
NAMES = f"{RAW}/cid_names.csv"
GAQ = f"{TC}/mlip/G_aq_tecrdb_full.json"
METS = f"{TC}/pipeline/tecrdb_full_metabolites.json"

def truthy(x): return str(x).strip().lower() in ("true", "1", "yes")

def block(smiles):
    if not smiles or not smiles.strip():
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        k = Chem.MolToInchiKey(m)
    except Exception:
        return None
    return k.split("-")[0] if k else None

# --- Du: per compound, the pH7 dominant species charge (+ a connectivity block) ---
rows = list(csv.DictReader(open(COMP)))
n_ph7_flag = sum(1 for r in rows if truthy(r.get("is_pH7_species")))
print(f"Du compounds table: {len(rows)} species rows; is_pH7_species flagged: {n_ph7_flag}")

du_ph7 = {}     # block -> (compound_id, charge_at_pH7)
du_smiles_by_cid = {}
for r in rows:
    cid = r["compound_id"].strip()
    s = (r.get("smiles_form") or "").strip()
    if s and cid not in du_smiles_by_cid:
        du_smiles_by_cid[cid] = s
    if truthy(r.get("is_pH7_species")):
        b = block(s)
        try:
            z = int(r["charge"])
        except (ValueError, KeyError):
            continue
        if b:
            du_ph7[b] = (cid, z)

names = {r["compound id"].strip(): r["compound name"].strip() for r in csv.DictReader(open(NAMES))}

# --- ours: cpd -> (charge, block) ---
gaq = json.load(open(GAQ))
cpd_smiles = {m["id"]: m.get("smiles", "") for m in json.load(open(METS))}

matches, mism = [], []
for cpd, v in gaq.items():
    b = block(cpd_smiles.get(cpd, ""))
    if b is None or b not in du_ph7:
        continue
    du_cid, du_z = du_ph7[b]
    ours_z = v.get("charge")
    rec = (cpd, v.get("name", ""), ours_z, du_z, names.get(du_cid, du_cid))
    (matches if ours_z == du_z else mism).append(rec)

tot = len(matches) + len(mism)
print(f"\nmatched to a Du pH7 species: {tot}")
print(f"  charge AGREES: {len(matches)}")
print(f"  charge DIFFERS: {len(mism)}   <-- wrong dominant microspecies in our pipeline")
if mism:
    print(f"\n{'cpd':10s}{'name':22s}{'ours':>6s}{'Du pH7':>8s}")
    for cpd, nm, oz, dz, dn in sorted(mism, key=lambda x: abs((x[2] or 0)-(x[3] or 0)), reverse=True):
        print(f"  {cpd:10s}{(nm or dn)[:20]:22s}{str(oz):>6s}{str(dz):>8s}")

with open(f"{TC}/data/external/du2018_group_contribution/audit_microspecies.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["cpd","name","our_charge","du_pH7_charge","agree"])
    for cpd, nm, oz, dz, dn in matches + mism:
        w.writerow([cpd, nm or dn, oz, dz, int(oz == dz)])
print("\nwrote audit_microspecies.csv")
