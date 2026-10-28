"""Select audited source ladders by exact scored reference structure.

No reaction identity, name, observed energy or fitted residual selects a model.
Only caller-owned, complete independent-site terms can be replaced. Explicit
caller models/flat terms and disabled neutral-reference policies retain priority.
"""
import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from rdkit import Chem

from metag.conditions import CONDITIONS
from metag.energetics.protonation_models import validate_model, prepare_models, reference_assignments

POLICY_VERSION = '2026-10-21-exact-source-acid-registry-v1'
_PATH = Path(__file__).parent / 'data/source_protonation_models.json'


@lru_cache(maxsize=4)
def _parse(data):
    raw = json.loads(data)
    if (not isinstance(raw, dict) or raw.get('schema') != 'metag-source-protonation-registry-v1'
            or not isinstance(raw.get('models'), list) or not raw['models']):
        raise ValueError('Invalid source-protonation registry')
    models = {}
    for model in raw['models']:
        checked = validate_model(model, model.get('reference_smiles'), 0, CONDITIONS)
        key = checked['reference_smiles']
        if key in models:
            raise ValueError('Duplicate source reference in protonation registry')
        models[key] = checked
    return models


def registry_identity():
    data = _PATH.read_bytes()
    _parse(data)
    return hashlib.sha256(data).hexdigest()


def resolve_protonation_models(reaction, scored_species, ownership, *, enabled, expected_registry_sha256=None):
    """Resolve automatic/caller models; prepare_models still enforces all guards.

    An unavailable ownership route never acquires a hidden reference transform.
    Registered models replace the same owned terms as an explicit declaration.
    Invalid assets are fatal; unsupported route ownership is recorded explicitly.
    """
    models = copy.deepcopy(reaction.get('species_protonation_models', {}))
    report = {'policy': POLICY_VERSION, 'enabled': bool(enabled), 'automatic_species': [],
              'caller_species': sorted(models), 'unapplied_matches': {}}
    if not enabled:
        return models, report
    data = _PATH.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_registry_sha256 is not None and digest != expected_registry_sha256:
        raise ValueError('Source protonation registry changed after job configuration was captured')
    registry = _parse(data)
    report['registry_sha256'] = digest
    for name, (coefficient, charge, smiles) in scored_species.items():
        molecule = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
        if any(atom.GetIsotope() for atom in molecule.GetAtoms()):
            continue
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
        if charge != 0 or canonical not in registry or name in models:
            continue
        original_name = name if name in reaction['species'] else (name[:-2] if name.endswith('_t') else None)
        if original_name not in reaction['species']:
            report['unapplied_matches'][name] = 'original full-molecule identity is unavailable'
            continue
        from metag.routing.pka_transform import _neutralize
        original_reference, _, original_charge = _neutralize(reaction['species'][original_name][2])
        original = Chem.MolFromSmiles(original_reference) if original_reference else None
        if original is not None:
            for atom in original.GetAtoms():
                atom.SetAtomMapNum(0)
        if original is None or original_charge != 0 or Chem.MolToSmiles(Chem.RemoveHs(original)) != canonical:
            report['unapplied_matches'][name] = 'scored core is not the original full source reference'
            continue
        if reaction.get('pka_sites'):
            report['unapplied_matches'][name] = 'explicit caller flat terms retain priority'
            continue
        if not isinstance(ownership, dict) or name not in ownership:
            report['unapplied_matches'][name] = 'no admitted owned neutral-reference transform'
            continue
        model = registry[canonical]
        if len(ownership[name]) != len(model['pKas']):
            raise ValueError('Registered source model has incomplete owned acid-site coverage: ' + name)
        models[name] = copy.deepcopy(model)
        report['automatic_species'].append(name)
    return models, report


def prepare_reaction_protonation(reaction, scored_species, sites, ownership, conditions, *,
                                 enabled, rt_ln10, expected_registry_sha256=None):
    """One CPU implementation shared by admission and scoring."""
    models, registry = resolve_protonation_models(reaction, scored_species, ownership, enabled=enabled,
                                                expected_registry_sha256=expected_registry_sha256)
    if models:
        remaining, coupled, diagnostics = prepare_models(
            models, scored_species, sites, ownership, conditions, rt_ln10=rt_ln10)
        references = diagnostics['species_reference_models']
    else:
        remaining, coupled, diagnostics = sites, 0., None
        references = reference_assignments(scored_species, ownership)
    return remaining, coupled, diagnostics, registry, references
