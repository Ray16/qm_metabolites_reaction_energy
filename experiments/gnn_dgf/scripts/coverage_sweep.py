#!/usr/bin/env python3
"""Structure-based ModelSEED coverage: GNN vs original dGP vs retrained dGP.

One decomposition pass over every compound with a SMILES. A compound is:
  - GNN-coverable        : RDKit parses the SMILES into a mol (graph exists)
  - originalDGP-coverable: every r1 group in group_names_r1 AND every r2 group in group_names_r2
  - retrainedDGP-coverable: every r1 group in modelseed_group_names_r1 AND r2 in modelseed_group_names_r2
                           (and RDKit-parses, no '*')
Faithful copy of dGPredictor count_substructures().
"""
import csv, glob, json, os
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

BIO = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/ModelSEEDDatabase/Biochemistry"
TOOLS = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/tools"

def load_vocab(p):
    with open(p) as f:
        return set(x for x in f.read().splitlines() if x)

r1_orig = load_vocab(f"{TOOLS}/dGPredictor/data/group_names_r1.txt")
r2_orig = load_vocab(f"{TOOLS}/dGPredictor/data/group_names_r2_py3_modified_manual.txt")
r1_re   = load_vocab(f"{TOOLS}/dGPredictor_freiburger/data/modelseed_group_names_r1.txt")
r2_re   = load_vocab(f"{TOOLS}/dGPredictor_freiburger/data/modelseed_group_names_r2.txt")
print(f"vocab sizes: orig r1={len(r1_orig)} r2={len(r2_orig)} | retrained r1={len(r1_re)} r2={len(r2_re)}")

def count_substructures(radius, m):
    smi_count = {}
    for i in range(m.GetNumAtoms()):
        env = Chem.FindAtomEnvironmentOfRadiusN(m, radius, i)
        atoms = set()
        for bidx in env:
            b = m.GetBondWithIdx(bidx)
            atoms.add(b.GetBeginAtomIdx()); atoms.add(b.GetEndAtomIdx())
        if not atoms:
            atoms = {i}
        smi = Chem.MolFragmentToSmiles(m, atomsToUse=list(atoms), bondsToUse=env, canonical=True)
        smi_count[smi] = smi_count.get(smi, 0) + 1
    return smi_count

def isnull(x): return x is None or x.strip() in ("", "null", "None", "nan")

n_active = n_smiles = 0
n_parse = n_star = 0
cov_gnn = cov_orig = cov_re = 0
# breakdown of dGP failures
orig_fail_r1 = orig_fail_r2 = 0
re_fail_parse = re_fail_r2 = 0
per = {}  # cpd_id -> {smiles, star, gnn, dgp_original, dgp_retrained}

for f in sorted(glob.glob(f"{BIO}/compound_*.tsv")):
    with open(f) as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r.get("is_obsolete","0") == "1":
                continue
            n_active += 1
            smi = r.get("smiles")
            if isnull(smi):
                continue
            n_smiles += 1
            has_star = '*' in smi
            if has_star: n_star += 1
            rec = {"star": has_star, "gnn": 0, "dgp_original": 0, "dgp_retrained": 0}
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                n_parse += 1
                per[r["id"]] = rec
                continue
            # GNN: needs only a parseable graph
            cov_gnn += 1; rec["gnn"] = 1
            s1 = set(count_substructures(1, mol))
            s2 = set(count_substructures(2, mol))
            # original dGP: subset of KEGG-trained vocab, both radii
            o1 = s1 <= r1_orig; o2 = s2 <= r2_orig
            if o1 and o2:
                cov_orig += 1; rec["dgp_original"] = 1
            else:
                if not o1: orig_fail_r1 += 1
                if not o2: orig_fail_r2 += 1
            # retrained dGP: built from ModelSEED; fails only on '*' or novel group
            if has_star:
                re_fail_parse += 1
            else:
                q1 = s1 <= r1_re; q2 = s2 <= r2_re
                if q1 and q2:
                    cov_re += 1; rec["dgp_retrained"] = 1
                else:
                    re_fail_r2 += 1
            per[r["id"]] = rec

json.dump(per, open(os.path.join(os.path.dirname(__file__), "..", "artifacts",
          "coverage_per_compound_struct.json"), "w"))

out = dict(n_active=n_active, n_smiles=n_smiles, n_star=n_star, n_parse_fail=n_parse,
           cov_gnn=cov_gnn, cov_orig=cov_orig, cov_re=cov_re,
           orig_fail_r1=orig_fail_r1, orig_fail_r2=orig_fail_r2,
           re_fail_star=re_fail_parse, re_fail_novelgroup=re_fail_r2)
print(json.dumps(out, indent=2))
def pct(a): return f"{a} ({100*a/n_active:.1f}% of active, {100*a/n_smiles:.1f}% of structured)"
print("\n=== COVERAGE (denominator = 45,662-ish active compounds) ===")
print(f"active compounds          : {n_active}")
print(f"with SMILES (structured)  : {n_smiles}  ({100*n_smiles/n_active:.1f}%)")
print(f"  RDKit parse failures    : {n_parse}")
print(f"  contain '*' (R-group)   : {n_star}")
print(f"GNN coverable             : {pct(cov_gnn)}")
print(f"retrained dGP coverable   : {pct(cov_re)}")
print(f"original  dGP coverable   : {pct(cov_orig)}")
json.dump(out, open("/tmp/claude-21574/-nfs-lambda-stor-01-homes-rzhu-ModelSEED-FAISS/be47355a-a611-4ec8-8e0e-44ebccd5f7fd/scratchpad/coverage_result.json","w"), indent=2)
