"""Build pipeline reaction entries from the TECRDB-367 benchmark (cpd stoich + pH-7
microspecies SMILES/charge). Computes n_Hplus from the H/charge imbalance (reactions are
in the transformed Alberty convention with NO explicit H+). Carries exp dG_kJ + sd_kJ (the
across-conditions spread = experimental uncertainty floor). Emits a full file and a
stratified sample across the structural failure-mode categories.
Run in `uma` env (needs rdkit). Writes reactions_tecrdb_all.json + reactions_tecrdb_sample.json.
"""
import json, os, collections
from rdkit import Chem

HERE = os.path.dirname(__file__)
PIPE = os.path.join(HERE, "..", "..", "..", "pipeline")
rxns = json.load(open(os.path.join(PIPE, "tecrdb_full_reactions.json")))
mets = {m["id"]: m for m in json.load(open(os.path.join(PIPE, "tecrdb_full_metabolites.json")))}
exp = json.load(open(os.path.join(PIPE, "tecrdb_full_experiment.json")))
flags = json.load(open(os.path.join(PIPE, "tecrdb367_failure_flags.json")))

def parsed(cid):
    s = mets.get(cid, {}).get("smiles", "")
    if not s or "*" in s: return None
    m = Chem.MolFromSmiles(s)
    return m

def nH_q(m):
    mh = Chem.AddHs(m)
    return sum(a.GetSymbol() == "H" for a in mh.GetAtoms()), Chem.GetFormalCharge(m)

out = {}
skipped = collections.Counter()
for rid, stoich in rxns.items():
    if rid not in exp: skipped["no-exp"] += 1; continue
    mols = {c: parsed(c) for c in stoich}
    if any(v is None for v in mols.values()): skipped["unparseable"] += 1; continue
    Hr = Hp = qr = qp = 0
    species = {}
    for c, coeff in stoich.items():
        nh, q = nH_q(mols[c])
        smi = Chem.MolToSmiles(mols[c])
        # KEY by FULL name (never truncate: name[:14] collided isomer substrate/product, e.g.
        # GlcNAc-1-P vs GlcNAc-6-P both -> "N-Acetyl-D-glu", collapsing the reaction to one species
        # and producing garbage ΔG). Disambiguate any residual name collision with the unique cid.
        key = mets[c]["name"] or c
        if key in species:
            key = f"{key} [{c}]"
        species[key] = [int(coeff), int(q), smi]
        if coeff < 0: Hr += -coeff * nh; qr += -coeff * q
        else:         Hp += coeff * nh;  qp += coeff * q
    nHplus_H = Hr - Hp            # net H+ released (from H balance)
    nHplus_q = qr - qp            # net H+ released (from charge balance)
    if nHplus_H != nHplus_q:      # not cleanly proton-balanced (O/other imbalance) -> skip
        skipped["unbalanced"] += 1; continue
    e = exp[rid]
    out[rid] = dict(exp=[round(e["dG_kJ"], 2)], exp_sd=round(e.get("sd_kJ", 0) or 0, 2),
                    EC=e.get("EC", ""), n_Hplus=int(nHplus_H), explicit=False,
                    note=f"TECRDB {rid} {e.get('enzyme','')[:40]} | flags={','.join(flags.get(rid,[]))}",
                    species=species)

json.dump(out, open(os.path.join(HERE, "..", "scripts", "reactions_tecrdb_all.json"), "w"), indent=2)
print(f"built {len(out)} runnable TECRDB reactions; skipped {dict(skipped)}")

# stratified sample: pick a few per failure category (dedup, cap total)
by_cat = collections.defaultdict(list)
for rid in out:
    for fl in flags.get(rid, ["CLEAN(tractable)"]):
        by_cat[fl].append(rid)
sample = {}
PER = 6
for cat, rids in sorted(by_cat.items()):
    for rid in rids[:PER]:
        sample[rid] = out[rid]
json.dump(sample, open(os.path.join(HERE, "..", "scripts", "reactions_tecrdb_sample.json"), "w"), indent=2)
print(f"stratified sample: {len(sample)} reactions across {len(by_cat)} categories")
print("categories:", {k: len(v) for k, v in sorted(by_cat.items())})
