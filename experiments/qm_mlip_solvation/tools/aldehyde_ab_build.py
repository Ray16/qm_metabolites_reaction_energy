"""Build the aldehyde-hydration A/B reaction file. For each target reaction, 2 variants:
  orig    : stored free aldehyde (R-CHO)
  hydrate : the strongly-hydrated aldehyde replaced by its gem-diol (R-CH(OH)2), with WATER balanced into
            the stoichiometry (diol = aldehyde + H2O, so add H2O with coeff -c_aldehyde to the reaction).

Unlike the anomer (which cancels + self-heals), the aldehyde<->gem-diol difference does NOT cancel (the
aldehyde is created/destroyed) and does NOT self-heal (ETKDG will not add a water to form the diol). GAP
shows the clean sign-flip signature (reactant rxn00781 err -44, product rxn00786 err +35) = free-aldehyde
energy too high by the missing hydration. If the hydrate collapses those errors, hydration is the fix.
"""
import json
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

D = json.load(open("scripts/reactions_tecrdb_all.json"))

# strongly-hydrated aldehydes to convert (Keq_hyd favours the diol). acetaldehyde (~50%) left OUT for now.
HYDRATE_NAMES = {"Glyceraldehyde3-phosphate", "D-Glyceraldehyde", "Glycolaldehyde", "Glyoxalate",
                 "Formaldehyde", "4-Oxobutanoate", "D-Arabinose5-phosphate", "L-Lactaldehyde",
                 "L-Aspartate4-semialdehyde", "D-Erythrose"}
TEST_RIDS = ["rxn00781", "rxn00786", "rxn03884", "rxn00783", "rxn01200",   # GAP (sign-flip)
             "rxn00276", "rxn15750", "rxn01306"]                            # glyoxylate/glyceraldehyde/glycolaldehyde

_ALDE = Chem.MolFromSmarts("[CX3;H1](=O)[#6]")
_RXN = AllChem.ReactionFromSmarts("[CX3;H1:1](=[OX1:2])>>[C:1]([O:2])[OH]")   # CHO -> CH(OH)2


def to_gemdiol(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None or not m.HasSubstructMatch(_ALDE):
        return None
    prod = _RXN.RunReactants((m,))
    if not prod:
        return None
    p = prod[0][0]
    try:
        Chem.SanitizeMol(p)
    except Exception:
        return None
    return Chem.MolToSmiles(p)


WATER_SMI = "O"
out = {}
for rid in TEST_RIDS:
    rx = D[rid]
    for variant in ("orig", "hydrate"):
        sp = {}
        dwater = 0
        for nm, (c, q, smi) in rx["species"].items():
            if variant == "hydrate" and nm in HYDRATE_NAMES:
                gd = to_gemdiol(smi)
                if gd is not None:
                    smi = gd
                    dwater += -c            # diol = aldehyde + H2O -> add H2O with coeff -c to rebalance
            sp[nm] = [c, q, smi]
        if variant == "hydrate" and dwater != 0:
            # fold the balancing water into an existing H2O species if present, else add one
            wkey = next((n for n, (c, q, s) in sp.items() if s == "O"), None)
            if wkey:
                sp[wkey][0] += dwater
                if sp[wkey][0] == 0:
                    del sp[wkey]
            else:
                sp["H2O_bal"] = [dwater, 0, WATER_SMI]
        key = f"{rid}_{variant}"
        out[key] = {k: rx[k] for k in rx if k != "species"}
        out[key]["species"] = sp

json.dump(out, open("artifacts/aldehyde_ab_reactions.json", "w"), indent=1)
print(f"wrote {len(out)} variant reactions ({len(TEST_RIDS)} rids x 2)")
# sanity: show GAP conversions + water balance
for rid in ["rxn00786", "rxn00781"]:
    print(f"\n{rid}:")
    for v in ("orig", "hydrate"):
        eq = "  ".join(f"{c:+g} {n}[{s}]" for n, (c, q, s) in out[f"{rid}_{v}"]["species"].items())
        print(f"  {v}: {eq}")
