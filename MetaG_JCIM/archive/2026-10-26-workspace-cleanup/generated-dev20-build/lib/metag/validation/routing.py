"""Checks for reaction-context-dependent species routing."""

from collections import defaultdict

from rdkit import Chem


def canonical(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid molecular SMILES")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    molecule = Chem.RemoveHs(molecule)
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def audit_routing_consistency(original, results):
    """Report original species that receive multiple scored representations."""
    uses = defaultdict(list)
    unknown = []
    for reaction_id, result in results.items():
        if reaction_id not in original or not isinstance(result.get("species_scored"), dict):
            continue
        routed = result["species_scored"]
        for name, value in original[reaction_id]["species"].items():
            if float(value[0]) == 0:
                continue
            # The planner's declared naming adapter preserves the input label
            # or appends _t. Never guess chemical mappings from other labels.
            candidates = [routed[label] for label in (name, name + "_t") if label in routed]
            candidates = [row for row in candidates if len(row) >= 3 and float(row[0]) == float(value[0])]
            if len(candidates) != 1:
                unknown.append({"reaction": reaction_id, "name": name,
                                "reason": "missing_or_ambiguous_declared_label_mapping"})
                continue
            scored = candidates[0]
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
    missing = sorted(set(original) - {rid for rid, row in results.items()
                                     if isinstance(row.get("species_scored"), dict)})
    return {
        "n_original_species": len(uses),
        "n_context_dependent_species": len(conflicts),
        "n_uses_affected": sum(row["n_uses"] for row in conflicts),
        "conflicts": conflicts,
        "unmapped_uses": unknown,
        "missing_reactions": missing,
        "consistent_known_representations": not conflicts and not unknown and not missing,
        "scope": "Original exact microscopic identities and their declared scored identities. Consistency does not validate protonation, aqueous ensembles or energy cycle closure.",
    }

