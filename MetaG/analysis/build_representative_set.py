"""Representative ModelSEED sample for the three-way UMA vs eQ vs GC comparison.

Design: per EC top-level class (1-6), two arms, random within each (seed fixed for reproducibility):
  - CONSENSUS arm  (|GC-eQ| < 8 kJ):  where the incumbents agree, their consensus is the closest thing to
    ground truth we have -> tests whether UMA LANDS THERE (validation).
  - DIVERGENCE arm (|GC-eQ| >= 8 kJ): where they disagree -> which does UMA back (adjudication).
N_PER_ARM per arm per class. Tractable filter: both thermo estimates, 12-40 heavy, >=2 species, parses.
Dedup by compound-name set (no near-identical reactions). Writes representative_inputs.json.
"""
import json, glob, os, random
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

DB = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/ModelSEEDDatabase/Biochemistry"
OUT = os.path.join(os.path.dirname(__file__), "representative_inputs.json")
HPLUS = "cpd00067"
EC_NAME = {"1": "Oxidoreductase", "2": "Transferase", "3": "Hydrolase",
           "4": "Lyase", "5": "Isomerase", "6": "Ligase"}
N_PER_ARM = 10
CONSENSUS = 8.0
random.seed(42)

cpd = {}
for f in glob.glob(f"{DB}/compound_*.json"):
    for c in json.load(open(f)):
        cpd[c["id"]] = c.get("smiles")

pool = {c: {"consensus": [], "diverge": []} for c in EC_NAME}
for f in sorted(glob.glob(f"{DB}/reaction_*.json")):
    for r in json.load(open(f)):
        if r.get("is_obsolete") or r.get("status") != "OK":
            continue
        t = r.get("thermodynamics") or {}
        gc, eq = t.get("Group contribution"), t.get("eQuilibrator")
        if not (gc and eq) or abs(gc[0]) >= 1e7 or abs(eq[0]) >= 1e7:
            continue
        ecs = r.get("ec_numbers") or []
        if not ecs:
            continue
        cls = ecs[0].split(".")[0]
        if cls not in EC_NAME:
            continue
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
        rec = {"id": r["id"], "note": f"{r['id']} {r.get('name','')} | EC={ecs[0]}", "n_Hplus": nH,
               "species": species, "gc": round(gc[0], 1), "eq": round(eq[0], 1),
               "delta": round(gc[0] - eq[0], 1), "max_heavy": mx, "ec": ecs[0],
               "cset": frozenset(species)}
        arm = "consensus" if abs(gc[0] - eq[0]) < CONSENSUS else "diverge"
        pool[cls][arm].append(rec)

picked = {}
for cls in sorted(EC_NAME):
    for arm in ("consensus", "diverge"):
        cand = pool[cls][arm]
        random.shuffle(cand)
        chosen, csets = [], []
        for v in cand:
            if len(chosen) >= N_PER_ARM:
                break
            if any(v["cset"] == cs for cs in csets):
                continue
            chosen.append(v); csets.append(v["cset"])
        for v in chosen:
            picked[v["id"]] = {k: v[k] for k in ("note", "n_Hplus", "species", "gc", "eq", "delta", "max_heavy", "ec")}

json.dump(picked, open(OUT, "w"), indent=1)
print(f"selected {len(picked)} reactions -> {OUT}")
for cls in sorted(EC_NAME):
    ncon = sum(1 for v in picked.values() if v["ec"].split(".")[0] == cls and abs(v["delta"]) < CONSENSUS)
    ndiv = sum(1 for v in picked.values() if v["ec"].split(".")[0] == cls and abs(v["delta"]) >= CONSENSUS)
    print(f"  EC{cls} {EC_NAME[cls]:16s} consensus={ncon:2d} divergence={ndiv:2d}  (pool {len(pool[cls]['consensus'])}/{len(pool[cls]['diverge'])})")
