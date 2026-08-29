"""Stratified probe set: 3 DIVERGENT (|GC-eQ|>30), chemically-DIVERSE reactions per EC top-level
class (EC1-6). EC7 (translocase) skipped -- only 11 tractable, transport chemistry (ΔG≈0). The point
is to extend the GC-vs-eQ adjudication BEYOND the accidental EC1/EC2 bias of the first 14 reactions,
across all real reaction chemistry, so we see whether UMA backs GC or eQ per reaction TYPE.

Diversity within a class: prefer distinct EC subclass (2nd digit) and never two reactions with the
same set of compound names (the sulfotransferase-triple lesson -- near-identical reactions are one
data point, not three). Smaller molecules preferred (faster QM) by sorting on max_heavy.
"""
import json, glob, os
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

DB = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/ModelSEEDDatabase/Biochemistry"
OUT = os.path.join(os.path.dirname(__file__), "category_inputs.json")
HPLUS = "cpd00067"
EC_NAME = {"1": "Oxidoreductase", "2": "Transferase", "3": "Hydrolase",
           "4": "Lyase", "5": "Isomerase", "6": "Ligase"}
N_PER = 3

cpd = {}
for f in glob.glob(f"{DB}/compound_*.json"):
    for c in json.load(open(f)):
        cpd[c["id"]] = c.get("smiles")

# gather candidates per EC class
cands = {c: [] for c in EC_NAME}
for f in sorted(glob.glob(f"{DB}/reaction_*.json")):
    for r in json.load(open(f)):
        if r.get("is_obsolete") or r.get("status") != "OK":
            continue
        t = r.get("thermodynamics") or {}
        gc, eq = t.get("Group contribution"), t.get("eQuilibrator")
        if not (gc and eq) or abs(gc[0]) >= 1e7 or abs(eq[0]) >= 1e7:
            continue
        if abs(gc[0] - eq[0]) < 30:
            continue
        ecs = r.get("ec_numbers") or []
        if not ecs:
            continue
        cls = ecs[0].split(".")[0]
        if cls not in EC_NAME:
            continue
        sub = ".".join(ecs[0].split(".")[:2])            # subclass, e.g. 2.7
        species, nH, ok, mx = {}, 0, True, 0
        for s in r.get("stoichiometry") or []:
            cid = s["compound"]
            if cid == HPLUS:
                nH += s["coefficient"]; continue
            smi = cpd.get(cid); m = Chem.MolFromSmiles(smi) if smi else None
            if m is None:
                ok = False; break
            mx = max(mx, m.GetNumHeavyAtoms())
            species[s.get("name") or cid] = [s["coefficient"], Chem.GetFormalCharge(m), smi]
        if not ok or not (12 <= mx <= 40) or len(species) < 2:
            continue
        cands[cls].append({
            "id": r["id"], "note": f"{r['id']} {r.get('name','')} | EC={ecs[0]}",
            "n_Hplus": nH, "species": species, "gc": round(gc[0], 1), "eq": round(eq[0], 1),
            "delta": round(gc[0] - eq[0], 1), "max_heavy": mx, "ec": ecs[0], "sub": sub,
            "cset": frozenset(species),
        })

picked = {}
for cls in sorted(EC_NAME):
    pool = sorted(cands[cls], key=lambda v: v["max_heavy"])   # smaller (faster) first
    chosen, subs_used, csets_used = [], set(), []
    # pass 1: distinct subclass, distinct compound-set
    for v in pool:
        if len(chosen) >= N_PER:
            break
        if v["sub"] in subs_used:
            continue
        if any(v["cset"] == cs or len(v["cset"] & cs) >= min(len(v["cset"]), len(cs)) for cs in csets_used):
            continue
        chosen.append(v); subs_used.add(v["sub"]); csets_used.append(v["cset"])
    # pass 2: fill remaining slots relaxing the distinct-subclass rule (still distinct compound-set)
    for v in pool:
        if len(chosen) >= N_PER:
            break
        if v in chosen or any(v["cset"] == cs for cs in csets_used):
            continue
        chosen.append(v); csets_used.append(v["cset"])
    for v in chosen:
        picked[v["id"]] = {k: v[k] for k in ("note", "n_Hplus", "species", "gc", "eq", "delta", "max_heavy", "ec")}

json.dump(picked, open(OUT, "w"), indent=1)
print(f"selected {len(picked)} reactions ({N_PER}/class x 6 EC classes) -> {OUT}\n")
for cls in sorted(EC_NAME):
    print(f"EC{cls} {EC_NAME[cls]} (pool: {len(cands[cls])} divergent)")
    for rid, v in picked.items():
        if v["ec"].split(".")[0] == cls:
            print(f"    {rid}  EC={v['ec']:12s} GC {v['gc']:+8.1f}  eQ {v['eq']:+8.1f}  Δ {v['delta']:+8.1f}  heavy {v['max_heavy']:2d}  {v['note'].split('|')[0].split(' ',1)[1][:38]}")
