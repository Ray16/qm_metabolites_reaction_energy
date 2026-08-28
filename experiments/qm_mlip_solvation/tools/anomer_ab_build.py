"""Build the anomer A/B reaction file: for each test reaction, 3 variants -- orig (stored undefined
anomer), alpha (alpha-pinned), beta (beta-pinned = physical dominant) -- with the flagged sugars swapped
consistently on BOTH sides. Writes artifacts/anomer_ab_reactions.json (RXN_FILE for unified_pipeline)."""
import json

D = json.load(open("scripts/reactions_tecrdb_all.json"))
V = json.load(open("artifacts/anomer_variants.json"))     # {sugar: {stored,alpha,beta,q,ring}}

TEST_RIDS = ["rxn00558", "rxn00223", "rxn00549", "rxn00220", "rxn08430"]

out = {}
for rid in TEST_RIDS:
    rx = D[rid]
    for variant in ("orig", "alpha", "beta"):
        sp = {}
        for nm, (c, q, smi) in rx["species"].items():
            if variant != "orig" and nm in V and V[nm].get(variant):
                smi = V[nm][variant]                       # swap in the pinned-anomer SMILES (charge unchanged)
            sp[nm] = [c, q, smi]
        key = f"{rid}_{variant}"
        out[key] = {k: rx[k] for k in rx if k != "species"}
        out[key]["species"] = sp

json.dump(out, open("artifacts/anomer_ab_reactions.json", "w"), indent=1)
print(f"wrote {len(out)} variant reactions ({len(TEST_RIDS)} rids x 3):")
for k in out:
    print("  ", k)
