"""Unseen-chemistry generality set: random ModelSEED reactions NOT in TECRDB, stratified by EC top class
(N_PER_CLASS each, seed 7). Filters: status OK, not obsolete, every compound has a parseable SMILES, no
transition metals, every species <= 40 heavy atoms, >= 2 species, element- and charge-balanced (with n_H+)
per metag.chem. No requirement for incumbent (GC/eQ) estimates, so GC-silent reactions are included.
Writes generality_inputs.json (MetaG input format; gc/eq kept where present, for reference only)."""
import glob, json, os, random, statistics, sys
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from metag.chem import is_balanced
TC = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DB = os.path.join(os.path.dirname(TC), "ModelSEEDDatabase", "Biochemistry")
TEC = set(json.load(open(os.path.join(TC, "experiments/qm_mlip_solvation/scripts/reactions_tecrdb_std.json"))))
N_PER_CLASS = 50
METALS = {21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 42, 74}
random.seed(7)
cpd = {}
for f in glob.glob(f"{DB}/compound_*.json"):
    for c in json.load(open(f)):
        cpd[c["id"]] = c.get("smiles")
pool = {str(k): [] for k in range(1, 7)}
for f in sorted(glob.glob(f"{DB}/reaction_*.json")):
    for r in json.load(open(f)):
        if r.get("is_obsolete") or r.get("status") != "OK" or r["id"] in TEC:
            continue
        ecs = r.get("ec_numbers") or []
        cls = ecs[0].split(".")[0] if ecs else None
        if cls not in pool:
            continue
        species, nH, ok = {}, 0, True
        for s in r.get("stoichiometry") or []:
            if s["compound"] == "cpd00067":
                nH += s["coefficient"]; continue
            smi = cpd.get(s["compound"]); m = Chem.MolFromSmiles(smi) if smi else None
            if m is None or m.GetNumHeavyAtoms() > 40 or any(a.GetAtomicNum() in METALS for a in m.GetAtoms()) \
                    or "*" in smi:
                ok = False; break
            species[(s.get("name") or s["compound"])[:60]] = [s["coefficient"], Chem.GetFormalCharge(m), smi]
        if not ok or len(species) < 2:
            continue
        rec = {"note": f"{r['id']} {r.get('name', '')} | EC={ecs[0]}", "n_Hplus": nH, "species": species, "ec": ecs[0]}
        try:
            if not is_balanced(rec["species"], nH):
                continue
        except Exception:
            continue
        t = r.get("thermodynamics") or {}
        for k, lab in (("Group contribution", "gc_kcal"), ("eQuilibrator", "eq_kcal")):
            v = t.get(k)
            if v and abs(v[0]) < 1e7: rec[lab] = v[0]
        pool[cls].append((r["id"], rec))
picked = {}
for cls, cand in pool.items():
    random.shuffle(cand)
    for rid, rec in cand[:N_PER_CLASS]:
        picked[rid] = rec
json.dump(picked, open(os.path.join(HERE, "generality_inputs.json"), "w"), indent=1)
print({c: len(v) for c, v in pool.items()}, "picked", len(picked))
