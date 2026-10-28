"""Caller-owned macroscopic acid ladders with explicit reference/condition guards.

This first supported bridge is deliberately narrow: a neutral, fully protonated
C/H/O carboxylic-acid family without a nonacid carbonyl, enol or other ionizable
group. Intermediate proton-count ensembles (including neutral AMP), bases,
hydration/tautomer mixtures and molecule-to-core transfers require other models.
No measured constants or reaction residuals are bundled or selected here.
"""
from collections import Counter
import math

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from metag.energetics.protonation import normalize_site, macroscopic_transform
from metag.energetics.provenance import content_hash

POLICY_VERSION = '2026-10-17-owned-carboxylic-macrostate-v2'
_ACID = Chem.MolFromSmarts('[CX3](=[OX1])[OX2H1]')


def _canonical(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError('Macroscopic model requires a single valid reference molecule')
    if any(atom.GetAtomMapNum() or atom.GetIsotope() for atom in molecule.GetAtoms()):
        raise ValueError('Mapped/isotope-specific macroscopic reference is unsupported')
    return Chem.MolToSmiles(molecule, isomericSmiles=True), molecule


def validate_model(model, scored_smiles, scored_charge, conditions):
    """Validate the portable source declaration; do not infer missing-state fractions.

    Provenance is caller supplied and auditable, not authenticated by this parser.
    Matching formula/structure is necessary, not evidence of ensemble completeness.
    """
    if not isinstance(model, dict):
        raise ValueError('Species protonation model must be an object')
    expected = {'kind', 'reference_smiles', 'reference_deprotonations', 'reference_scope',
                'reference_formula', 'pKas', 'conditions', 'source'}
    if set(model) != expected:
        raise ValueError('Species protonation model fields must be explicit: ' + ', '.join(sorted(expected)))
    if model['kind'] != 'sequential_macroscopic_acid_pka':
        raise ValueError('Unsupported species protonation model kind')
    if (type(model['reference_deprotonations']) is not int or model['reference_deprotonations'] != 0
            or model['reference_scope'] != 'fully_protonated_carboxylic_acid_family'):
        raise ValueError('An intermediate macrostate cannot be anchored to an arbitrary computed microstate')
    canonical, molecule = _canonical(model['reference_smiles'])
    scored, _ = _canonical(scored_smiles)
    if canonical != scored or scored_charge != 0 or Chem.GetFormalCharge(molecule) != 0:
        raise ValueError('Source reference does not match the scored full neutral species; core transfer is unsupported')
    if model['reference_formula'] != rdMolDescriptors.CalcMolFormula(molecule):
        raise ValueError('Source reference formula mismatch')
    if any(atom.GetAtomicNum() not in (1, 6, 8) or atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
        raise ValueError('Only closed-shell C/H/O carboxylic reference families are currently supported')
    acid_matches = molecule.GetSubstructMatches(_ACID)
    if len({match[0] for match in acid_matches}) != len(acid_matches):
        raise ValueError('Carbonic/other noncarboxylic acid reference needs a separate reservoir model')
    for carbon, oxo, hydroxyl in acid_matches:
        atom = molecule.GetAtomWithIdx(carbon)
        other = [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetIdx() not in (oxo, hydroxyl)]
        if (other and any(neighbor.GetAtomicNum() not in (1, 6) for neighbor in other)) or (not other and atom.GetTotalNumHs()!=1):
            raise ValueError('Source ladder requires carboxylic acid groups, not carbonic/other acid sites')
    acid_oxygen = {index for match in acid_matches for index in match[1:]}
    # Outside carboxyls admit only saturated-carbon alcohol O-H. Other oxygens
    # can require hydration, tautomer, ether protonation or ester/lactone models.
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 8 or atom.GetIdx() in acid_oxygen:
            continue
        neighbors = list(atom.GetNeighbors())
        if (atom.GetTotalNumHs() != 1 or len(neighbors) != 1
                or neighbors[0].GetHybridization() != Chem.HybridizationType.SP3):
            raise ValueError('Reference has a carbonyl/tautomer or other oxygen-state branch needing a coupled model')
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum()==6 and sum(neighbor.GetAtomicNum()==8 and neighbor.GetIdx() not in acid_oxygen
                                       and neighbor.GetTotalNumHs()==1 for neighbor in atom.GetNeighbors())>1:
            raise ValueError('Gem-diol reference needs explicit hydration/state mapping')
    values = model['pKas']
    if not isinstance(values, (list, tuple)) or not values or len(values) != len(acid_matches):
        raise ValueError('Sequential ladder must cover every fully protonated carboxylic acid site')
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError('Sequential pKas must be finite numbers')
    source_conditions = model['conditions']
    if not isinstance(source_conditions, dict) or set(source_conditions) != {
            'T_K', 'ionic_strength_M', 'solvent', 'standard_state', 'magnesium_binding'}:
        raise ValueError('Explicit source temperature, ionic strength, solvent, standard state and Mg convention required')
    for field in ('T_K', 'ionic_strength_M'):
        value = source_conditions[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Source condition must be finite: ' + field)
        if value != conditions[field]:
            raise ValueError('Source condition mismatch: ' + field)
    if (source_conditions['solvent'] != 'water'
            or source_conditions['standard_state'] != 'solute_1M_activity'
            or source_conditions['magnesium_binding'] != 'absent'
            or conditions.get('pMg') is not None or conditions.get('water_activity') != 1):
        raise ValueError('Only water, 1M activity convention and no Mg binding are supported')
    source = model['source']
    required_source = {'locator', 'location', 'artifact_sha256', 'determination', 'parameter_scale',
                       'conversion', 'uncertainty', 'uncertainty_kind', 'covariance', 'development_exposure'}
    if not isinstance(source, dict) or set(source) != required_source:
        raise ValueError('Complete source provenance and uncertainty declarations are required')
    if source['determination'] not in ('measured', 'evaluated') or source['parameter_scale'] != 'molarity_activity':
        raise ValueError('Model requires measured/evaluated parameters on the target activity convention')
    for field in ('locator', 'location', 'conversion', 'uncertainty_kind'):
        if not isinstance(source[field], str) or not source[field].strip():
            raise ValueError('Missing source field: ' + field)
    digest = source['artifact_sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in '0123456789abcdef' for char in digest):
        raise ValueError('Source artifact SHA256 required')
    if type(source['development_exposure']) is not bool:
        raise ValueError('Source development exposure must be declared')
    if source['covariance'] is not None:
        raise ValueError('Parameter covariance propagation is not yet implemented; declare unavailable as null')
    uncertainty = source['uncertainty']
    if uncertainty is not None:
        if (not isinstance(uncertainty, (list, tuple)) or len(uncertainty) != len(values)
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or value < 0 for value in uncertainty)):
            raise ValueError('Source uncertainties must be nonnegative values for every sequential constant')
    return {**model, 'reference_smiles': canonical}


def reference_assignments(scored_species, ownership, models=None):
    """Describe per-molecule transforms independently of labels and extent.

    ``models`` contains validated declarations. Their source identity remains
    part of the assignment; identical parameter values alone do not establish
    that two callers used the same scientific reference model.
    """
    if not isinstance(ownership, dict):
        return {}
    records = {}
    for name, (coefficient, charge, smiles) in scored_species.items():
        if not coefficient:
            continue
        if name in (models or {}):
            assignment = {'kind': 'sequential_macroscopic_acid_pka', 'model': models[name]}
        else:
            terms = []
            for term in ownership.get(name, []):
                side, pka, kind, multiplicity = normalize_site(term)
                if side != ('react' if coefficient < 0 else 'prod') or multiplicity != abs(coefficient):
                    raise ValueError('Protonation ownership and scored stoichiometry disagree')
                terms.append([kind, pka])
            assignment = {'kind': 'independent_sites', 'terms': sorted(terms)}
        molecule = Chem.MolFromSmiles(smiles)
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        molecule = Chem.RemoveHs(molecule)
        records[name] = {
            'reference_smiles': Chem.MolToSmiles(molecule, isomericSmiles=True),
            'reference_charge': charge, 'assignment': assignment,
            'assignment_sha256': content_hash(assignment),
        }
    return records


def prepare_models(models, scored_species, sites, ownership, conditions, *, rt_ln10):
    """Replace owned class-site terms by complete sequential molecular polynomials.

    Returns remaining independent terms, signed coupled contribution and a
    portable diagnostic. A flat user pKa list provides no species ownership.
    """
    if not isinstance(models, dict) or not models:
        raise ValueError('species_protonation_models must be a nonempty mapping')
    if not isinstance(ownership, dict):
        raise ValueError('Per-species protonation ownership is unavailable for this route')
    remaining = Counter(normalize_site(site) for site in sites)
    declarations, rows = {}, {}
    coupled = 0.
    for name, model in models.items():
        if name not in scored_species or name not in ownership:
            raise ValueError('Protonation model has no owned scored species: ' + str(name))
        coefficient, charge, smiles = scored_species[name]
        checked = validate_model(model, smiles, charge, conditions)
        owned = ownership[name]
        if len(owned) != len(checked['pKas']):
            raise ValueError('Class-term ownership does not cover this whole source reference')
        side = 'react' if coefficient < 0 else 'prod'
        for term in owned:
            normalized = normalize_site(term)
            if normalized[0] != side or normalized[2] != 'acid' or normalized[3] != abs(coefficient):
                raise ValueError('Protonation ownership and scored stoichiometry disagree')
            if remaining[normalized] <= 0:
                raise ValueError('Owned site is absent from the reaction transform; double replacement refused')
            remaining[normalized] -= 1
        transformed = macroscopic_transform(checked['pKas'], conditions['pH'], rt_ln10,
                                             reference_deprotonations=0)
        signed = coefficient * transformed['G_transform']
        coupled += signed
        declarations[name] = checked
        rows[name] = {'model':checked, 'removed_independent_terms':owned,
                      'species_transform_kj':transformed['G_transform'], 'reaction_contribution_kj':signed,
                      'proton_count_populations':transformed['populations'],
                      'reaction_dG_d_pH_kj_per_unit': coefficient * transformed['dG_d_pH'],
                      'reaction_dG_d_pKa_kj_per_unit': [coefficient * value for value in transformed['dG_d_pKa']],
                      'uncertainty_status':'not propagated; missing covariance and other model errors remain'}
    references = reference_assignments(scored_species, ownership, declarations)
    seen = {}
    for name, record in references.items():
        identity = (record['reference_smiles'], record['reference_charge'])
        previous = seen.setdefault(identity, (name, record['assignment_sha256']))
        if previous[1] != record['assignment_sha256']:
            raise ValueError('Identical scored species have inconsistent protonation models: '
                             + previous[0] + ', ' + name)
    remainder = [list(term) for term, count in remaining.items() for _ in range(count)]
    return remainder, coupled, {'policy':POLICY_VERSION, 'models_sha256':content_hash(declarations),
             'species_reference_models': references,
             'species':rows, 'coupled_contribution_kj':coupled,
             'reaction_dG_d_pH_kj_per_unit':math.fsum(row['reaction_dG_d_pH_kj_per_unit'] for row in rows.values()),
             'scope':'Source-backed opt-in proton-count polynomial on a declared fully protonated reference family; '
                     'no empirical accuracy, missing-state coverage or parameter uncertainty claim.'}
