#!/usr/bin/env python
"""Match Du et al. 2018 compound-level formation thermodynamics
(dG_f, dH_f, dS_f) to this project's ab-initio species set, by InChIKey.

Values are loaded from the canonical copy at ModelSEED_FAISS/data/organic_cpd_thermo_data.csv.
SMILES for the Du compounds come from the auxiliary Du tables (kept under
data/external/du2018_group_contribution/raw/). User species come from the
ModelSEED metabolite list and the ab-initio G_aq set.

Run in an env with RDKit, e.g.:  conda run -n boltz-2 python build_validation_set.py
"""
import csv, json, os, re
from collections import defaultdict
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS"
FORMATION_CSV = f"{ROOT}/data/organic_cpd_thermo_data.csv"          # per user: load from here
RAW = f"{ROOT}/thermodynamic_calc/data/external/du2018_group_contribution/raw"
SMILES_CSVS = [f"{RAW}/TECRDB_compounds_data.csv", f"{RAW}/dSf_pKMg_data.csv"]
CID_NAMES = f"{RAW}/cid_names.csv"
MODELSEED = f"{ROOT}/thermodynamic_calc/mlip_modelseed/modelseed_feasible_metabolites.json"
ABINITIO = f"{ROOT}/thermodynamic_calc/mlip/G_aq_tecrdb_full.json"
OUT_CSV = f"{ROOT}/thermodynamic_calc/data/external/du2018_group_contribution/du2018_formation_matched.csv"

CHARGE_SUFFIX = re.compile(r"_(-?\d+)$")

def compound_of(species_id):
    """CHB_17754_0 -> CHB_17754 ; leave already-compound ids unchanged."""
    m = CHARGE_SUFFIX.search(species_id)
    return species_id[: m.start()] if m else species_id

def ikey(smiles):
    """(full_inchikey, connectivity_block) or (None,None)."""
    if not smiles or not smiles.strip():
        return None, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    try:
        k = Chem.MolToInchiKey(mol)
    except Exception:
        return None, None
    if not k:
        return None, None
    return k, k.split("-")[0]

# ---------- 1. Du formation values (compound-level sets) ----------
val_compounds = defaultdict(set)   # data type -> {compound_id}
with open(FORMATION_CSV) as fh:
    for r in csv.DictReader(fh):
        dt = r["data type"]
        cid = r["updated_compound_id"].strip() or compound_of(r["updated_species_id"].strip())
        if cid:
            val_compounds[dt].add(cid)
print("Du formation coverage (unique compounds):")
for dt in ("dG_f", "dH_f", "dS_f", "dG_f_prime", "Cp"):
    print(f"  {dt:11s} {len(val_compounds[dt])}")

# ---------- 2. compound_id -> SMILES ----------
smiles = {}
for path in SMILES_CSVS:
    with open(path) as fh:
        for r in csv.DictReader(fh):
            s = (r.get("smiles_form") or "").strip()
            if not s:
                continue
            cid = (r.get("compound_id") or "").strip()
            if not cid:
                cid = compound_of((r.get("species_id") or "").strip())
            if cid and cid not in smiles:
                smiles[cid] = s
names = {}
with open(CID_NAMES) as fh:
    for r in csv.DictReader(fh):
        names[r["compound id"].strip()] = r["compound name"].strip()

# Du compound -> inchikey blocks (only compounds that carry any formation value)
du_all = set().union(*(val_compounds[d] for d in ("dG_f", "dH_f", "dS_f")))
du_block = {}   # compound_id -> connectivity block
du_full = {}
no_smiles = 0
for cid in du_all:
    full, block = ikey(smiles.get(cid))
    if block:
        du_block[cid] = block
        du_full[cid] = full
    elif cid not in smiles:
        no_smiles += 1
print(f"\nDu compounds w/ any formation value: {len(du_all)}; SMILES-resolved InChIKey: {len(du_block)}; no SMILES: {no_smiles}")

# ---------- 3. user species -> inchikey blocks ----------
ms = json.load(open(MODELSEED))            # list of {id, smiles, ...}
ms_block = defaultdict(set)   # block -> {cpd}
cpd_smiles = {}
for m in ms:
    s = (m.get("smiles") or "").strip()
    cpd = m["id"]
    if s:
        cpd_smiles[cpd] = s
        full, block = ikey(s)
        if block:
            ms_block[block].add(cpd)

abinitio_cpds = set(json.load(open(ABINITIO)).keys())     # 453 species we computed
abinitio_blocks = set()
for cpd in abinitio_cpds:
    full, block = ikey(cpd_smiles.get(cpd, ""))
    if block:
        abinitio_blocks.add(block)
print(f"ModelSEED metabolites: {len(ms)} ({len(ms_block)} distinct InChIKey blocks); "
      f"ab-initio set: {len(abinitio_cpds)} cpds ({len(abinitio_blocks)} blocks)")

# ---------- 4. overlap ----------
def overlap(dt):
    comps = [c for c in val_compounds[dt] if c in du_block]
    in_ms = [c for c in comps if du_block[c] in ms_block]
    in_ai = [c for c in comps if du_block[c] in abinitio_blocks]
    return len(val_compounds[dt]), len(comps), len(in_ms), len(in_ai)

print("\n" + "=" * 74)
print("OVERLAP with user species (InChYKey connectivity-block match)")
print(f"{'quantity':12s}{'Du total':>10s}{'w/SMILES':>10s}{'∈ModelSEED':>12s}{'∈ab-initio(453)':>18s}")
rows_out = []
for dt in ("dG_f", "dH_f", "dS_f"):
    tot, wsm, nms, nai = overlap(dt)
    print(f"{dt:12s}{tot:>10d}{wsm:>10d}{nms:>12d}{nai:>18d}")

# ---------- 5. write matched table ----------
with open(OUT_CSV, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["du_compound_id", "name", "smiles", "inchikey", "inchikey_block",
                "has_dG_f", "has_dH_f", "has_dS_f", "matched_cpd", "in_abinitio_set"])
    for cid in sorted(du_block):
        block = du_block[cid]
        matched = sorted(ms_block.get(block, []))
        w.writerow([cid, names.get(cid, ""), smiles.get(cid, ""), du_full[cid], block,
                    int(cid in val_compounds["dG_f"]), int(cid in val_compounds["dH_f"]),
                    int(cid in val_compounds["dS_f"]),
                    matched[0] if matched else "",
                    int(block in abinitio_blocks)])
print(f"\nwrote {OUT_CSV}")
