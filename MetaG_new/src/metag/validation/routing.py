"""Checks for reaction-context-dependent species routing."""

from collections import defaultdict

from rdkit import Chem


def canonical(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return smiles
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def audit_routing_consistency(original, results):
    """Report original species that receive multiple scored representations."""
    uses = defaultdict(list)
    for reaction_id, result in results.items():
        if reaction_id not in original or not isinstance(result.get("species_scored"), dict):
            continue
        routed = result["species_scored"]
        by_base_name = {name.removesuffix("_t"): value for name, value in routed.items()}
        for name, value in original[reaction_id]["species"].items():
            if len(value) < 3:
                continue
            scored = routed.get(name) or by_base_name.get(name)
            if scored is None or len(scored) < 3:
                continue
            original_key = (canonical(value[2]), int(value[1]))
            route = result.get("routes") or {}
            uses[original_key].append(
                {
                    "rid": reaction_id,
                    "name": name,
                    "scored_smiles": canonical(scored[2]),
                    "scored_charge": int(scored[1]),
                    "truncated": bool(route.get("truncated")),
                    "ph0": bool(route.get("ph0")),
                }
            )

    conflicts = []
    for (smiles, charge), entries in uses.items():
        representations = defaultdict(list)
        for entry in entries:
            key = (entry["scored_smiles"], entry["scored_charge"])
            representations[key].append(entry["rid"])
        if len(representations) <= 1:
            continue
        conflicts.append(
            {
                "original_smiles": smiles,
                "original_charge": charge,
                "n_uses": len(entries),
                "n_representations": len(representations),
                "representations": [
                    {
                        "scored_smiles": key[0],
                        "scored_charge": key[1],
                        "n": len(reaction_ids),
                        "reactions": sorted(set(reaction_ids)),
                    }
                    for key, reaction_ids in representations.items()
                ],
            }
        )
    conflicts.sort(key=lambda row: (-row["n_uses"], -row["n_representations"]))
    return {
        "n_original_species": len(uses),
        "n_context_dependent_species": len(conflicts),
        "n_uses_affected": sum(row["n_uses"] for row in conflicts),
        "conflicts": conflicts,
    }

