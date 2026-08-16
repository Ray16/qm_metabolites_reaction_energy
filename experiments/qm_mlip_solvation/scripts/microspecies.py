#!/usr/bin/env python
"""Protonation / microspecies from ModelSEED's ChemAxon pH-7 assignments.

DECISION: use ModelSEED's ChemAxon states, NOT Dimorphite. Validation
(validate_dimorphite_vs_modelseed.py) found only ~59% net-charge agreement, with
genuine Dimorphite failures on sulfate (->0), polyamines (lysine ->0 vs +1), and
polyphosphates/CoA (Δ+2 under-deprotonation) — exactly the metabolite classes we care
about. ModelSEED ships ChemAxon-assigned pH-7 charge + SMILES (+ per-site pKa/pKb) for
every compound; that is the authoritative source and is already computed.

This module looks up a compound's ChemAxon major microspecies (charge + SMILES) by
ModelSEED id / name / InChIKey, and (optionally) enumerates nearby protonation states
from the per-site pKa column for Boltzmann averaging via the transformed free energy.

    G'(species,pH) = -RT ln Σ_i exp( -[ G_i - N_H(i)·g_proton(pH) ] / RT )

which folds microspecies averaging AND proton release/uptake into one number (a
deprotonation is just a lower-N_H microspecies) — no manual n_H+ term.
"""
import csv
import glob
import os

from rdkit import Chem

T = 298.15
RT = 8.314e-3 * T
G_H_GAS, DGSOLV_H = -26.3, -1104.5
MSDB = os.environ.get("MSDB_BIOCHEM",
    "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/ModelSEEDDatabase/Biochemistry")


def g_proton(pH):
    return G_H_GAS + DGSOLV_H - 2.303 * RT * pH        # ~ -1170.7 kJ/mol at pH 7


_CACHE = {}


def load_modelseed(force=False):
    """Index ModelSEED compounds: {id -> dict(charge, smiles, name, inchikey, pka, pkb)}.
    Also secondary indexes by inchikey and lowercased name."""
    if _CACHE and not force:
        return _CACHE
    by_id, by_ikey, by_name = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(MSDB, "compound_*.tsv"))):
        with open(f) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                smi = (row.get("smiles") or "").strip()
                if not smi or smi in ("null", "None") or row.get("is_obsolete") == "1":
                    continue
                try:
                    rec = dict(id=row["id"], name=row["name"], charge=int(row["charge"]),
                               smiles=smi, inchikey=(row.get("inchikey") or "").strip(),
                               pka=(row.get("pka") or "").strip(), pkb=(row.get("pkb") or "").strip())
                except (ValueError, KeyError):
                    continue
                by_id[rec["id"]] = rec
                if rec["inchikey"]:
                    by_ikey.setdefault(rec["inchikey"], rec)
                    by_ikey.setdefault(rec["inchikey"].split("-")[0], rec)  # skeleton match
                by_name.setdefault(rec["name"].lower(), rec)
    _CACHE.update(dict(by_id=by_id, by_ikey=by_ikey, by_name=by_name))
    return _CACHE


def protonation(query):
    """Return ModelSEED ChemAxon (smiles, charge) for a compound id / name / SMILES /
    InChIKey. For a raw SMILES, matches by InChIKey skeleton. None if not found."""
    db = load_modelseed()
    if query in db["by_id"]:
        r = db["by_id"][query]; return r["smiles"], r["charge"]
    if query.lower() in db["by_name"]:
        r = db["by_name"][query.lower()]; return r["smiles"], r["charge"]
    if query in db["by_ikey"]:
        r = db["by_ikey"][query]; return r["smiles"], r["charge"]
    m = Chem.MolFromSmiles(query)                     # try structure match by InChIKey skeleton
    if m is not None:
        try:
            skel = Chem.MolToInchiKey(m).split("-")[0]
            if skel in db["by_ikey"]:
                r = db["by_ikey"][skel]; return r["smiles"], r["charge"]
        except Exception:
            pass
    return None


def n_hydrogens(smi):
    m = Chem.MolFromSmiles(smi)
    return None if m is None else sum(a.GetTotalNumHs() + (a.GetSymbol() == "H") for a in m.GetAtoms())


def transformed_G(micro_G, ph=7.0):
    """Combine microspecies into the species' transformed free energy at pH.
    micro_G = list of (n_H, G_i_kJ). Returns G'(pH) = -RT ln Σ exp(-[G_i - n_H·g_p]/RT)."""
    import math
    gp = g_proton(ph)
    prime = [G - nH * gp for nH, G in micro_G if G is not None]
    if not prime:
        return None
    lo = min(prime)
    return lo - RT * math.log(sum(math.exp(-(p - lo) / RT) for p in prime))


if __name__ == "__main__":
    for q in ["cpd00002", "cpd00012", "ATP", "PPi", "UDP-glucose", "D-Fructose", "cpd00048"]:
        r = protonation(q)
        print(f"  {q:14s} -> {r}")
