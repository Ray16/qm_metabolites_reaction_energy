"""CPU-only reaction/protonation preflight: python -m metag.tools.preflight INPUT -o REPORT."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from rdkit import Chem, rdBase
from metag.reactions import validate_reaction
from metag.routing.pka_transform import pka_model_config
from metag.routing.planner import route_reaction
from metag.validation.protonation import inspect_species
from metag.validation.stereochemistry import unresolved_species
from metag.conditions import CONDITIONS, RT_LN10
from metag.energetics.protonation_models import reference_assignments
from metag.protonation_registry import prepare_reaction_protonation


def audit_reactions(reactions):
    from dataclasses import asdict
    from metag.config import flag
    from metag.routing.policy import RoutingPolicy
    from metag.routing.production_policy import validate_production_policy
    from metag.routing import pka_transform as pk
    import os
    policy = RoutingPolicy.from_runtime(flag, os.environ)
    validate_production_policy(policy)
    config = dict(pka_model_config(), routing=asdict(policy),
                  pka_switches={name: getattr(pk, name)() for name in (
                      "_pka_env_enabled", "_free_ppi_pka_enabled", "_anhydride_pka_enabled",
                      "_carboxyl_pairs_enabled", "_polyacid_pka_enabled", "_arylamine_nonbasic_enabled",
                      "_redox_proton_enabled")})
    if flag('SOURCE_PKA_REGISTRY'):
        from metag.protonation_registry import registry_identity
        config['source_protonation_registry_sha256'] = registry_identity()
    source_root = Path(__file__).parents[1]
    source_hashes = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(source_root.rglob("*.py"))}
    species, records = {}, {}
    for key, reaction in reactions.items():
        try:
            validate_reaction(reaction)
            from metag.chem import isotope_labels
            isotopic = {name: labels for name, (_, _, smi) in reaction["species"].items()
                        if (labels := isotope_labels(smi))}
            if isotopic:
                records[key] = {"status": "blocked", "unsupported_isotopes": isotopic,
                                "error": "isotope-specific mass and solvent/reservoir conventions are not implemented"}
                continue
        except (ValueError, TypeError) as exc:
            records[key] = {"status": "invalid", "error": str(exc)}
            continue
        keys = []
        for name, (_, charge, smi) in reaction["species"].items():
            canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            keys.append(canonical)
            if canonical not in species:
                species[canonical] = dict(inspect_species(canonical), uses=[])
            species[canonical]["uses"].append({"reaction": key, "name": name, "input_charge": charge})
        try:
            routed, routes, _ = route_reaction(reaction, log=lambda _: None)
            routed_species = {name: inspect_species(smi) for name, (_, _, smi) in routed["species"].items()}
            stereo = {"input": unresolved_species(reaction["species"]),
                      "routed": unresolved_species(routed["species"])}
            record = {"routes": routes, "routed_species": routed_species, "stereochemistry": stereo,
                      "species_scored": {name:list(value) for name,value in routed["species"].items()},
                      "n_Hplus_scored": routed.get("n_Hplus", 0),
                      "pka_sites_scored": routed.get("pka_sites")}
            if ((reaction.get('species_protonation_models') or flag('SOURCE_PKA_REGISTRY')) and routes['input_balanced']
                    and not routes['errors'] and not stereo['input'] and not stereo['routed']):
                try:
                    sites, coupled, diagnostics, registry, references = prepare_reaction_protonation(
                        reaction, routed['species'],
                        routed.get('pka_sites', []), routes.get('protonation_sites_by_species'),
                        CONDITIONS, enabled=flag('SOURCE_PKA_REGISTRY'), rt_ln10=RT_LN10,
                        expected_registry_sha256=config.get('source_protonation_registry_sha256'))
                except (ValueError, TypeError, KeyError) as exc:
                    records[key] = dict(record, status='blocked_source_protonation_model', error=str(exc))
                    continue
                routes['source_protonation_registry'] = registry
                if diagnostics is not None:
                    routes['coupled_protonation_models'] = diagnostics
                record.update(pka_sites_scored=sites, coupled_protonation_kj=coupled,
                              protonation_reference_models=references)
            elif not reaction.get('species_protonation_models'):
                record['protonation_reference_models'] = reference_assignments(
                    routed['species'], routes.get('protonation_sites_by_species'))
            if (flag('RING_PROTON_ENSEMBLE') and routes['input_balanced'] and not routes['errors']
                    and not stereo['input'] and not stereo['routed']):
                from metag.state_pools import prepare_ring_pools
                try:
                    plans = prepare_ring_pools(routed, routes, explicit=routed['explicit'], ring_chain=flag('RING_CHAIN'))
                    if plans and flag('ALDEHYDE_HYDRATION') and not flag('CARBONYL_HYDRATION_ALL'):
                        raise ValueError('Ring-proton pools require complete carbonyl hydration or hydration disabled')
                except (ValueError, TypeError, KeyError) as exc:
                    records[key] = dict(record, status='blocked_ring_proton_ensemble', error=str(exc))
                    continue
                routes['ring_proton_state_plan'] = plans
            if routes.get("compound_reference_status") == "failed":
                # No reference route was admitted. The unchanged input is a
                # diagnostic fallback, not a scored representation.
                record.pop("species_scored")
            if not routes["input_balanced"] or routes["errors"]:
                status = "blocked"
            elif stereo["input"] or stereo["routed"]:
                status = "blocked_unresolved_stereochemistry"
            elif any(species[s]["status"] == "unsupported" for s in keys) or any(
                    row["status"] == "unsupported" for row in routed_species.values()):
                status = "review_unsupported_species"
            else:
                status = "review" if any(species[s]["issues"] for s in keys) or any(
                    row["issues"] for row in routed_species.values()) else "no_known_gap_detected"
            records[key] = dict(record, status=status)
        except (ValueError, RuntimeError) as exc:
            records[key] = {"status": "blocked", "error": str(exc)}
    return {"scope": "Structural preflight only; no_known_gap_detected is not validated speciation or accuracy.",
            "config": config, "source_sha256": source_hashes, "rdkit_version": rdBase.rdkitVersion,
            "n_reactions": len(reactions), "n_unique_structures": len(species),
            "status_counts": dict(Counter(r["status"] for r in records.values())),
            "issue_counts": dict(Counter(i for r in species.values() for i in r["issues"])),
            "species": species, "reactions": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.input.read_bytes()
    reactions = json.loads(raw)
    if not isinstance(reactions, dict):
        parser.error("input must be a JSON mapping of reaction identifiers to reactions")
    report = audit_reactions(reactions)
    report.update(input_file=str(args.input.resolve()), input_sha256=hashlib.sha256(raw).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("n_reactions", "n_unique_structures", "status_counts", "issue_counts")}, indent=2))


if __name__ == "__main__":
    main()
