"""Zero-GPU MICROSPECIES AUDIT: is the input molecule itself correct? A wrong tautomer / protonation /
ring-form / undefined stereocentre gives a wrong dG no matter how good the QM is -- this MARKS such
species for correction and prioritises them by the error of the reactions they sit in.

Per unique benchmark species (name, stored charge, SMILES) we flag:
  charge_mismatch     rdkit formal charge(SMILES) != stored charge field         (data-integrity bug)
  carboxyl_neutral    a -COOH is protonated (pKa~4 -> should be -COO- at pH 7)
  amine_neutral       an aliphatic amine is neutral (pKa~10 -> should be -NH3+/=NH2+ at pH 7)
  phosphate_proton    a P-OH is protonated where the pH-7 microspecies is deprotonated
  tautomer_noncanon   rdkit canonical tautomer != stored (e.g. enol stored as keto)
  free_aldehyde       a free R-CHO that is predominantly the gem-diol hydrate in water
  undefined_stereo    unassigned stereocentre(s) -> ETKDG silently randomises the anomer/config

Then JOIN to per_rxn_benchmark: every reaction that contains a flagged species is listed with its
|uma_qm_err|, so the actionable corrections (flagged species in HIGH-error reactions) sort to the top.
Heuristic protonation flags are conservative and marked CONFIDENCE lo/med; review before patching.
Output: artifacts/microspecies_audit.tsv (+ console summary).
"""
import json, os, sys
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import FindMolChiralCenters
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

D = json.load(open("scripts/reactions_tecrdb_all.json"))
B = json.load(open("../gnn_dgf/artifacts/per_rxn_benchmark.json"))

# SMARTS
COOH   = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")            # protonated carboxyl
# aliphatic 1deg/2deg amine, neutral; EXCLUDE amide N, imine/=N, and guanidinium/amidinium-conjugated N
AMINE  = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O);!$(N=*);!$(NC=[NX3+]);!$(NC=[NX2]);!$(N-c)][CX4]")
POH    = Chem.MolFromSmarts("[PX4](=O)[OX2H1]")            # protonated phosphate OH
ALDE   = Chem.MolFromSmarts("[CX3;H1](=O)[#6]")            # free aldehyde R-CHO (not formate)

_TE = rdMolStandardize.TautomerEnumerator()


def audit_species(smi, stored_q):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return [("parse_fail", "hi", "SMILES does not parse")]
    flags = []
    fc = Chem.GetFormalCharge(m)
    if fc != stored_q:
        flags.append(("charge_mismatch", "hi", f"formal charge {fc} != stored {stored_q}"))
    if m.HasSubstructMatch(COOH):
        flags.append(("carboxyl_neutral", "med", "-COOH protonated; pH-7 dominant is -COO-"))
    if m.HasSubstructMatch(AMINE):
        flags.append(("amine_neutral", "med", "aliphatic amine neutral; pH-7 dominant is protonated"))
    if m.HasSubstructMatch(POH):
        flags.append(("phosphate_proton", "lo", "P-OH protonated; check pH-7 microspecies"))
    if m.HasSubstructMatch(ALDE):
        flags.append(("free_aldehyde", "med", "free R-CHO; may be gem-diol hydrate in water"))
    # undefined stereocentres on CARBON only (phosphate-P centres are phantom -> excluded). An anomeric
    # carbon (ring C bonded to two O) that is unassigned is the real ETKDG anomer-mixing defect.
    try:
        cc = FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)
        undef_c, anomeric = [], 0
        for i, lab in cc:
            if lab != "?":
                continue
            a = m.GetAtomWithIdx(i)
            if a.GetSymbol() != "C":                        # skip phantom P/S/N phosphate-type centres
                continue
            undef_c.append(i)
            nO = sum(1 for n in a.GetNeighbors() if n.GetSymbol() == "O")
            if a.IsInRing() and nO >= 1 and any(nb.GetSymbol() == "O" and not nb.IsInRing() for nb in a.GetNeighbors()):
                anomeric += 1                                # ring C with an exocyclic O = anomeric carbon
        if undef_c:
            det = f"{len(undef_c)} undefined C stereocentre(s)" + (f"; {anomeric} anomeric" if anomeric else "")
            flags.append(("undefined_stereo_C", "hi" if anomeric else "med", det))
    except Exception:
        pass
    # tautomer: compare STEREO-STRIPPED skeletons (canonicalizer strips stereo -> compare like-for-like),
    # so we detect real proton/double-bond repositions (keto/enol, amide/imidic), not stereo loss.
    if m.GetNumHeavyAtoms() <= 30:
        try:
            stored_flat = Chem.MolToSmiles(m, isomericSmiles=False)
            canon_flat = Chem.MolToSmiles(_TE.Canonicalize(m), isomericSmiles=False)
            if canon_flat != stored_flat:
                flags.append(("tautomer_skeleton", "lo", f"rdkit-dominant tautomer differs: {canon_flat}"))
        except Exception:
            pass
    return flags


def main():
    # unique species (dedupe by (name, smi, q))
    uniq = {}
    sp2rxn = defaultdict(set)
    for rid, v in D.items():
        for nm, (c, q, smi) in v["species"].items():
            uniq[(nm, smi, q)] = (smi, q)
            sp2rxn[(nm, smi, q)].add(rid)

    def err(rid):
        e = B.get(rid, {}).get("uma_qm_err")
        return abs(e) if e is not None else None

    rows = []
    for key, (smi, q) in uniq.items():
        nm = key[0]
        flags = audit_species(smi, q)
        if not flags:
            continue
        rids = sorted(sp2rxn[key])
        errs = [err(r) for r in rids if err(r) is not None]
        max_err = max(errs) if errs else None
        rows.append(dict(name=nm, smi=smi, q=q, flags=flags, rids=rids,
                         max_err=max_err, n_rxn=len(rids)))

    # sort: species whose flags land in the highest-error reactions first
    rows.sort(key=lambda r: (r["max_err"] is None, -(r["max_err"] or 0)))

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/microspecies_audit.tsv", "w") as fh:
        fh.write("max_err\tn_rxn\tname\tq\tflag\tconfidence\tdetail\tsmiles\trxns\n")
        for r in rows:
            for ftype, conf, detail in r["flags"]:
                fh.write(f"{('' if r['max_err'] is None else round(r['max_err'],1))}\t{r['n_rxn']}\t"
                         f"{r['name']}\t{r['q']:+d}\t{ftype}\t{conf}\t{detail}\t{r['smi']}\t{','.join(r['rids'])}\n")

    # summary
    by_flag = defaultdict(int)
    for r in rows:
        for ftype, _, _ in r["flags"]:
            by_flag[ftype] += 1
    print(f"unique species audited: {len(uniq)}   flagged: {len(rows)}\n")
    print("flag counts (unique species):")
    for f, n in sorted(by_flag.items(), key=lambda x: -x[1]):
        print(f"  {f:20s} {n}")
    # high-error actionable (max_err > 15)
    hot = [r for r in rows if r["max_err"] is not None and r["max_err"] > 15]
    print(f"\nflagged species sitting in a >15 kJ-error reaction: {len(hot)}")
    print(f"{'max_err':>7s}  {'name':32s} {'q':>2s}  flags")
    for r in hot[:35]:
        print(f"{r['max_err']:7.1f}  {r['name'][:32]:32s} {r['q']:+2d}  {[f[0] for f in r['flags']]}")
    print("\nwrote artifacts/microspecies_audit.tsv")


if __name__ == "__main__":
    main()
