"""Read ModelSEED's atom-level ChemAxon pKa/pKb annotations for triage only."""
from __future__ import annotations

import csv
import glob
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSeedPkaSite:
    kind: str  # ``pka`` (acid dissociation) or ``pkb`` (base dissociation)
    fragment: int
    atom: int
    value: float


def parse_sites(text: str, kind: str) -> tuple[ModelSeedPkaSite, ...]:
    """Parse ModelSEED's ``fragment:atom:value`` format without reinterpreting it."""
    if not text or text == "null":
        return ()
    sites = []
    for entry in text.split(";"):
        fields = entry.split(":")
        if len(fields) != 3:
            continue
        try:
            sites.append(ModelSeedPkaSite(kind, int(fields[0]), int(fields[1]), float(fields[2])))
        except ValueError:
            continue
    return tuple(sites)


def load_compounds(database_root: str | Path, compound_ids: set[str]) -> dict[str, dict]:
    """Fetch only requested compounds from a ModelSEED Biochemistry checkout."""
    root = Path(database_root) / "Biochemistry"
    result = {}
    for path in glob.glob(str(root / "compound_*.tsv")):
        with open(path) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["id"] not in compound_ids:
                    continue
                result[row["id"]] = {
                    "name": row["name"], "charge": int(row["charge"]),
                    "smiles": row["smiles"],
                    "sites": parse_sites(row.get("pka", ""), "pka") +
                             parse_sites(row.get("pkb", ""), "pkb"),
                }
    return result


def sites_near_window(sites: tuple[ModelSeedPkaSite, ...], pH_min: float, pH_max: float,
                      margin: float = 1.0) -> tuple[ModelSeedPkaSite, ...]:
    """Return candidate transitions near a measurement range, retaining source labels."""
    return tuple(site for site in sites if pH_min - margin <= site.value <= pH_max + margin)
