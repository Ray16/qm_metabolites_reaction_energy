#!/usr/bin/env python
"""Prepare the QM inputs for the collaborator's dGPredictor-vs-TECRDB top-10 disagreements.

Produces, from `top10_metabolites_stereo_significant.csv` + the ModelSEED reaction
shards, everything `run_uma_composite_single.py` needs -- WITHOUT touching the GPU/xtb job:

  1. metabolites.json  -- 23 metabolites (H+ excluded) in the exact
                                 `qm_thermo.structures.load_metabolites` schema.
  2. reactions.json    -- the 10 reactions as {rxn_id: {cpd_id: coeff}},
                                 read DIRECTLY from the raw DB by id (H+ dropped).
                                 5 of the 10 (the glutathione-redox + glyoxalase)
                                 carry a ModelSEED `status="CI:*"` charge-imbalance
                                 flag, so `load_modelseed_reactions()` filters them
                                 out; we bypass that filter -- the stoichiometry is
                                 exactly what benchmark matched to TECRDB and is charge
                                 balanced with the served-structure charges.
  3. geometries/{cpd}/conf_000.xyz -- ETKDG(v3)+MMFF starting geometry per compound
                                 (pure CPU, seconds each). `xtb --ohess` in the
                                 runner re-optimises, so a single clean conformer is
                                 a sufficient starting point (mirrors the central
                                 composite, which uses one geometry per compound).

Then a no-GPU dry validation: load the metabolite JSON back through
`load_metabolites`, build all 10 `Reaction` objects, and confirm every non-proton
species has a geometry. Run in the `palm` env (rdkit):

    python build_inputs.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
sys.path.insert(0, THERMO)
from qm_thermo import config  # noqa: E402
from qm_thermo.geometry import Geometry, write_xyz  # noqa: E402

PROTON = "cpd00067"
MET_CSV = os.path.join(HERE, "top10_metabolites_stereo_significant.csv")
MET_JSON = os.path.join(HERE, "metabolites.json")
RXN_JSON = os.path.join(HERE, "reactions.json")
# Precomputed per-species data (charge + H-count) so the GPU runner never needs
# rdkit -- the `uma` env has ase but not rdkit.
SPECIES_JSON = os.path.join(HERE, "species.json")
GEOM_DIR = os.path.join(HERE, "geometries")

# The 10 representative ModelSEED reaction ids (ranks 1..10 in the reactions CSV).
RXN_IDS = [
    "rxn00086", "rxn32133", "rxn00070", "rxn34788", "rxn00605",
    "rxn01713", "rxn01834", "rxn00579", "rxn01675", "rxn01005",
]


def build_metabolites() -> list[dict]:
    """Metabolite records in the load_metabolites schema (H+ excluded)."""
    records = []
    with open(MET_CSV) as fh:
        for row in csv.DictReader(fh):
            cid = row["metabolite_id"]
            if cid == PROTON:
                continue  # pH handles the proton; it never needs QM
            smiles = row["smiles"]  # served (pH-7 charged) structure benchmark matched
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise SystemExit(f"[build] RDKit cannot parse {cid} SMILES {smiles!r}")
            charge = Chem.GetFormalCharge(mol)
            records.append({
                "id": cid,
                "name": row["common_name"],
                "smiles": smiles,
                "formula": row["formula"],
                "charge": int(charge),
                "inchikey": row["inchikey"],
                "opentecr_species": "",
            })
    records.sort(key=lambda r: r["id"])
    return records


def build_reactions() -> dict[str, dict]:
    """Read the 10 reactions straight from the raw DB shards, by id (H+ dropped)."""
    db = os.path.join(config.PROJECT_DIR, "ModelSEEDDatabase", "Biochemistry")
    want = set(RXN_IDS)
    raw: dict[str, dict] = {}
    for path in glob.glob(os.path.join(db, "reaction_*.json")):
        for rec in json.load(open(path)):
            if rec["id"] in want:
                raw[rec["id"]] = rec
    out: dict[str, dict] = {}
    for rid in RXN_IDS:
        if rid not in raw:
            raise SystemExit(f"[build] {rid} not found in any reaction shard")
        stoich: dict[str, float] = {}
        for s in raw[rid]["stoichiometry"]:
            cpd = s["compound"]
            if cpd == PROTON:
                continue
            stoich[cpd] = stoich.get(cpd, 0.0) + s["coefficient"]
        stoich = {c: v for c, v in stoich.items() if abs(v) > 1e-9}
        out[rid] = stoich
    return out


def make_geometry(rec: dict) -> None:
    """ETKDGv3 + MMFF (UFF fallback) single-conformer starting geometry."""
    cdir = os.path.join(GEOM_DIR, rec["id"])
    os.makedirs(cdir, exist_ok=True)
    dest = os.path.join(cdir, "conf_000.xyz")

    mol = Chem.AddHs(Chem.MolFromSmiles(rec["smiles"]))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC00
    params.numThreads = 0
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=8, params=params))
    if not cids:
        params.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=8, params=params))
    if not cids:
        raise SystemExit(f"[build] {rec['id']}: RDKit failed to embed any conformer")

    # Prefer MMFF; fall back to UFF for atoms MMFF lacks parameters for.
    energies = []
    if AllChem.MMFFHasAllMoleculeParams(mol):
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000, numThreads=0)
        energies = [(e, cid) for cid, (ok, e) in zip(cids, res)]
        ff = "MMFF"
    else:
        res = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000, numThreads=0)
        energies = [(e, cid) for cid, (ok, e) in zip(cids, res)]
        ff = "UFF"
    energies.sort(key=lambda t: t[0])
    best = energies[0][1]

    conf = mol.GetConformer(best)
    syms = tuple(a.GetSymbol() for a in mol.GetAtoms())
    coords = tuple((p.x, p.y, p.z)
                   for p in (conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())))
    geom = Geometry(syms, coords, charge=int(rec["charge"]), multiplicity=1)
    write_xyz(geom, dest, comment=f"{rec['id']} {rec['name']} {ff} q={rec['charge']}")
    print(f"  [geom] {rec['id']:10s} {ff}  {len(syms)} atoms -> {dest}")


def dry_validate(records: list[dict], reactions: dict[str, dict]) -> None:
    """No-GPU gate: schema loads, reactions build, every species has a geometry."""
    from qm_thermo.structures import load_metabolites
    from qm_thermo.reactions import Reaction, species_info

    mets = {m.cpd_id: m for m in load_metabolites(MET_JSON)}
    print(f"\n[validate] load_metabolites: {len(mets)} records OK "
          f"(charges match recorded).")

    universe = set(mets)
    problems = []
    for rid, stoich in reactions.items():
        rxn = Reaction(rid, stoich)
        missing = {c for c in rxn.compounds() if c not in universe}
        if missing:
            problems.append(f"{rid}: species without a metabolite record: {sorted(missing)}")
    # forward/reverse sign check (ranks 1<->2, 3<->4)
    def canon(st):
        return tuple(sorted((c, round(v, 6)) for c, v in st.items()))
    def neg(st):
        return tuple(sorted((c, round(-v, 6)) for c, v in st.items()))
    pairs = [("rxn00086", "rxn32133"), ("rxn00070", "rxn34788")]
    for a, b in pairs:
        if canon(reactions[a]) != neg(reactions[b]):
            problems.append(f"{a}/{b} are not exact stoichiometric reverses")

    for cid in mets:
        xyz = os.path.join(GEOM_DIR, cid, "conf_000.xyz")
        if not os.path.isfile(xyz):
            problems.append(f"{cid}: missing geometry {xyz}")
        else:
            _ = species_info(mets[cid])  # exercises the H-count / charge path

    if problems:
        print("\n[validate] PROBLEMS:")
        for p in problems:
            print("   -", p)
        raise SystemExit(1)

    # Emit the rdkit-derived per-species data the GPU runner reads (no rdkit needed there).
    species = {cid: {"name": mets[cid].name,
                     "charge": species_info(mets[cid]).charge,
                     "n_hydrogens": species_info(mets[cid]).n_hydrogens}
               for cid in mets}
    json.dump(species, open(SPECIES_JSON, "w"), indent=1)
    print(f"[validate] wrote {SPECIES_JSON} ({len(species)} species: charge + H-count)")
    print(f"[validate] all 10 reactions build; {len(mets)} geometries present; "
          f"fwd/rev pairs consistent.")
    print("[validate] OK -- ready to submit run_uma_composite_single.py on a GPU node.")


def main() -> None:
    os.makedirs(GEOM_DIR, exist_ok=True)
    records = build_metabolites()
    json.dump(records, open(MET_JSON, "w"), indent=1)
    print(f"[build] wrote {MET_JSON} ({len(records)} metabolites, H+ excluded)")

    reactions = build_reactions()
    json.dump(reactions, open(RXN_JSON, "w"), indent=1)
    print(f"[build] wrote {RXN_JSON} ({len(reactions)} reactions)")

    print(f"[build] generating starting geometries in {GEOM_DIR} ...")
    for rec in records:
        make_geometry(rec)

    dry_validate(records, reactions)


if __name__ == "__main__":
    main()
