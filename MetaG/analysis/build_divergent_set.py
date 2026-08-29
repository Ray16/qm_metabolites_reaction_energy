"""Select ModelSEED reactions where the two incumbent additivity methods (Jankowski GC vs eQuilibrator)
DISAGREE strongly but the molecules are tractable for QM, and build MetaG score_reaction() inputs from
ModelSEED structures. These are exactly the reactions where a first-principles adjudicator has the clearest
value: at least one incumbent is badly wrong and there's no way to tell which."""
import json, glob, re, os
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

DB = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/ModelSEEDDatabase/Biochemistry"
OUT = os.path.join(os.path.dirname(__file__), "divergent_inputs.json")
SENT = 1e7
HPLUS, WATER = "cpd00067", "cpd00001"

# compound -> (smiles, charge)
cpd = {}
for f in glob.glob(f"{DB}/compound_*.json"):
    for c in json.load(open(f)):
        cpd[c["id"]] = (c.get("smiles"), c.get("charge"))


def heavy(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    return m.GetNumHeavyAtoms() if m else 999


picked = {}
for f in sorted(glob.glob(f"{DB}/reaction_*.json")):
    for r in json.load(open(f)):
        if r.get("is_obsolete") or r.get("status") != "OK":
            continue
        t = r.get("thermodynamics") or {}
        gc, eq = t.get("Group contribution"), t.get("eQuilibrator")
        if not (gc and eq) or abs(gc[0]) >= SENT or abs(eq[0]) >= SENT:
            continue
        if abs(gc[0] - eq[0]) < 30:                      # want strong disagreement
            continue
        stoich = r.get("stoichiometry") or []
        species, nH, ok, mx = {}, 0, True, 0
        for s in stoich:
            cid = s["compound"]
            if cid == HPLUS:
                nH += s["coefficient"]                    # H+ -> n_Hplus (product +, reactant -)
                continue
            smi, chg = cpd.get(cid, (None, None))
            m = Chem.MolFromSmiles(smi) if smi else None
            if m is None:
                ok = False; break
            h = m.GetNumHeavyAtoms(); mx = max(mx, h)
            species[s.get("name") or cid] = [s["coefficient"], Chem.GetFormalCharge(m), smi]
        if not ok or not (12 <= mx <= 40) or len(species) < 2:
            continue
        picked[r["id"]] = {"note": f"{r['id']} {r.get('name','')}", "n_Hplus": nH,
                           "species": species, "gc": round(gc[0], 1), "eq": round(eq[0], 1),
                           "delta": round(gc[0] - eq[0], 1), "max_heavy": mx}
        if len(picked) >= 14:
            break
    if len(picked) >= 14:
        break

json.dump(picked, open(OUT, "w"), indent=1)
print(f"selected {len(picked)} divergent, tractable reactions -> {OUT}")
for rid, v in picked.items():
    print(f"  {rid}  GC {v['gc']:+7.1f}  eQ {v['eq']:+7.1f}  Δ {v['delta']:+7.1f}  heavy {v['max_heavy']:2d}  {v['note'][:40]}")
