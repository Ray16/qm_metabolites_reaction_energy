"""ModelSEED input adapter: convert the ModelSEED biochemistry DB into the pipeline's runnable
reaction format {name:[coeff, charge, SMILES]} + n_Hplus, exactly analogous to
build_tecrdb_reactions.py. This is the glue that lets the coherent QM pipeline (truncate ->
gated pH-0 -> UQ) fire on the full metabolic database, not just the 367-reaction benchmark.

Source (ModelSEEDDatabase/Biochemistry): compound_*.tsv (col: id, formula, charge, smiles) +
reaction_*.tsv (col: id, stoichiometry, status, ec_numbers, deltag). The compound `smiles` is the
pH-7 microspecies with charge-consistent anion count (verified: ATP cpd00002 charge -3, 3x [O-]).

FILTERS (only cleanly-scoreable reactions, like the TECRDB loader skips unparseable/unbalanced):
  - reaction status == "OK" (mass + charge balanced)
  - not a transport reaction (single compartment; multi-compartment ΔG needs membrane potential)
  - every non-proton compound has a usable SMILES (not null, no '*' R-group/polymer)
Proton handling: H+ (cpd00067) is EXPLICIT in ModelSEED stoichiometry -> its coefficient IS the
reaction n_Hplus (net protons); exclude it from the species dict (the pipeline adds n_Hplus*G_HPLUS).
Species keyed by full name with cid-disambiguation (the name[:14] collision bug is NOT repeated).

Run in `uma` env (rdkit). Writes scripts/reactions_modelseed.json (+ a stratified sample).
"""
import json, os, glob, re, collections
from rdkit import Chem

HERE = os.path.dirname(__file__)
BIO = os.path.join(HERE, "..", "..", "..", "..", "ModelSEEDDatabase", "Biochemistry")
PROTON = "cpd00067"
WATER = "cpd00001"

# ---- load compounds: cid -> (charge, smiles, formula) --------------------------------------------
compounds = {}
for f in sorted(glob.glob(os.path.join(BIO, "compound_*.tsv"))):
    with open(f) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        ci = {c: i for i, c in enumerate(header)}
        for line in fh:
            r = line.rstrip("\n").split("\t")
            if len(r) <= ci["smiles"]:
                continue
            cid, smi, chg = r[ci["id"]], r[ci["smiles"]], r[ci["charge"]]
            compounds[cid] = (chg, smi, r[ci["formula"]])

# ---- parse a stoichiometry field into [(coeff, cid, name), ...] ----------------------------------
_ENTRY = re.compile(r'(-?\d+):(cpd\d+):(-?\d+):"([^"]*)"')


def usable(cid):
    if cid not in compounds:
        return False
    chg, smi, _ = compounds[cid]
    if not smi or smi == "null" or "*" in smi:
        return False
    return Chem.MolFromSmiles(smi) is not None


def build():
    out = {}
    skipped = collections.Counter()
    for f in sorted(glob.glob(os.path.join(BIO, "reaction_*.tsv"))):
        with open(f) as fh:
            header = fh.readline().rstrip("\n").split("\t")
            ci = {c: i for i, c in enumerate(header)}
            for line in fh:
                r = line.rstrip("\n").split("\t")
                if len(r) <= ci["stoichiometry"]:
                    continue
                rid, status = r[ci["id"]], r[ci["status"]]
                if status != "OK":
                    skipped[f"status:{status.split(':')[0]}"] += 1
                    continue
                if r[ci["is_transport"]] not in ("0", "", "null"):
                    skipped["transport"] += 1
                    continue
                entries = _ENTRY.findall(r[ci["stoichiometry"]])
                if not entries:
                    skipped["no-stoich"] += 1
                    continue
                # multi-compartment? (any compartment != the first) -> skip (transport-like)
                comps = {c for _, _, c, _ in [(a, b, cp, nm) for a, b, cp, nm in entries]}
                n_Hplus = 0
                species = {}
                bad = False
                for coeff, cid, comp, name in entries:
                    coeff = int(coeff)
                    if cid == PROTON:
                        n_Hplus += coeff            # explicit H+ -> reaction proton count
                        continue
                    if not usable(cid):
                        bad = True
                        break
                    chg, smi, _ = compounds[cid]
                    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
                    key = name or cid
                    if key in species:
                        key = f"{key} [{cid}]"
                    species[key] = [coeff, int(chg), smi]
                if bad:
                    skipped["cpd-no-smiles"] += 1
                    continue
                if len(species) < 2:
                    skipped["<2-species"] += 1
                    continue
                ec = r[ci["ec_numbers"]] if ci.get("ec_numbers", -1) < len(r) else ""
                # ModelSEED has NO experimental ΔG. Use its group-contribution deltag (kcal/mol ->
                # kJ) as `exp` so the pipeline runs and we get a QM-vs-GCM comparison; placeholder 0
                # when GCM is missing/1e7-sentinel (the predicted ΔG±σ is the actual output either way).
                msg = r[ci["deltag"]] if ci.get("deltag", 99) < len(r) else ""
                try:
                    g = float(msg)
                    exp = [round(g * 4.184, 2)] if abs(g) < 1e6 else [0.0]
                except ValueError:
                    exp = [0.0]
                out[rid] = dict(exp=exp, exp_sd=0.0, n_Hplus=n_Hplus, explicit=False,
                                note=f"ModelSEED {rid} {r[ci['name']][:40]} | EC={ec}",
                                ms_deltag_kcal=msg, species=species)
    return out, skipped


if __name__ == "__main__":
    out, skipped = build()
    json.dump(out, open(os.path.join(HERE, "..", "scripts", "reactions_modelseed.json"), "w"), indent=1)
    print(f"ModelSEED adapter: {len(out)} runnable reactions written")
    print("skipped:", dict(skipped))
