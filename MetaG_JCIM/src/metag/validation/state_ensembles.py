"""Controlled component replacement and fixed-charge molecular-state partitions.

Only declared, recorded minima are combined. No states are generated, selected
by reaction residuals, or promoted into production. Selected-minimum controls
are distinguished from reweighting every recorded conformer. Neither establishes
complete aqueous speciation or an experimental prediction interval.
"""
from collections import Counter
from collections.abc import Mapping
import copy
import math

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, inchi

from metag.energetics.provenance import content_hash, ordered_geometry_record
from metag.energetics.strict_smd import HARTREE_KJ
from metag.validation.solvent_ensemble import PRODUCTION_KT_KJ, reweight_smd
from metag.validation.stereochemistry import inspect_stereochemistry

ARMS = ('original', 'electronic_replaced', 'solvent_replaced', 'both_replaced')


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _partition(values):
    if not values or not all(_finite(value) for value in values):
        raise ValueError('A nonempty set of finite state energies is required')
    lowest = min(values)
    terms = [math.exp(-(value - lowest) / PRODUCTION_KT_KJ) for value in values]
    normalization = math.fsum(terms)
    return lowest - PRODUCTION_KT_KJ * math.log(normalization), [term / normalization for term in terms]


def _canonical(smiles):
    molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError('A single declared molecular state is required')
    if any(atom.GetIsotope() or atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
        raise ValueError('Isotope-specific and open-shell state pools are unsupported')
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    molecule = Chem.RemoveHs(molecule)
    return Chem.MolToSmiles(molecule, isomericSmiles=True), molecule


def _source_minima(cache):
    """Verify the entire parent before permitting a selected-minimum control."""
    if not isinstance(cache, Mapping) or cache.get('method') != 'implicit':
        raise ValueError('A primary implicit species cache is required')
    meta = cache.get('meta', {})
    ensemble = meta.get('thermal', {}).get('ensemble', {})
    minima = ensemble.get('validated_minima')
    if (meta.get('solv_relax') or cache.get('settings', {}).get('via')
            or not isinstance(minima, list) or not minima
            or ensemble.get('n_reference_thermal_approximation') != 0):
        raise ValueError('Recorded gas minima with validated per-minimum thermals are required')
    canonical, molecule = _canonical(cache.get('smi'))
    expected_symbols = Counter(atom.GetSymbol() for atom in Chem.AddHs(molecule).GetAtoms())
    charge = Chem.GetFormalCharge(molecule)
    if type(cache.get('q')) is not int or cache['q'] != charge:
        raise ValueError('Cache charge must match the declared molecular state')
    seen = set()
    for row in minima:
        geometry = row.get('geometry', {})
        try:
            expected = ordered_geometry_record(geometry['symbols'], geometry['positions_angstrom'],
                                               geometry['charge'], geometry['spin_multiplicity'])
            same_reference = _canonical(row['reference_smiles'])[0] == canonical
        except (KeyError, TypeError) as exc:
            raise ValueError('Recorded geometry and reference identity are required') from exc
        if (expected != geometry or geometry['sha256'] in seen or not same_reference
                or Counter(geometry['symbols']) != expected_symbols
                or geometry['charge'] != charge or geometry['spin_multiplicity'] != 1):
            raise ValueError('Geometry, formula, charge, spin or reference identity mismatch')
        seen.add(geometry['sha256'])
        components = [row.get(key) for key in ('electronic_kj', 'solvation_kj', 'thermal_kj', 'total_kj')]
        if (not all(_finite(value) for value in components)
                or not math.isclose(math.fsum(components[:3]), components[3], rel_tol=0, abs_tol=1e-7)):
            raise ValueError('Recorded components must be finite and close to the total')
    parent_g, _ = _partition([row['total_kj'] for row in minima])
    if not _finite(cache.get('G')) or not math.isclose(parent_g, cache['G'], rel_tol=0, abs_tol=1e-7):
        raise ValueError('Parent cache energy does not reconstruct from all recorded minima')
    return minima


def solvation_range(cache, *, reference_kj=None):
    """Bound neutral phase transfer under any reweighting of the recorded basis.

    With gas weights p_i>0 and fixed solvent terms s_i, transfer is
    -RT*log(sum(p_i*exp(-s_i/RT))), hence lies in [min(s_i),max(s_i)].
    Electronic/thermal changes may alter p_i but cannot leave that envelope.
    The envelope is conditional on these structures and solvent terms: it is
    not a prediction interval, a missing-state bound, or a guarantee relative
    to a true experimental value. A supplied reference is a central value only.
    Charged transfer comparisons require a separately supported ion convention.
    """
    minima = _source_minima(cache)
    if cache['q'] != 0:
        raise ValueError('Neutral molecular transfer is required; ion conventions are unsupported')
    if reference_kj is not None and not _finite(reference_kj):
        raise ValueError('The optional reference must be a finite central value in kJ/mol')
    gas, _ = _partition([row['electronic_kj'] + row['thermal_kj'] for row in minima])
    aqueous, _ = _partition([row['total_kj'] for row in minima])
    solvent = [row['solvation_kj'] for row in minima]
    lower, upper = min(solvent), max(solvent)
    transfer = aqueous - gas
    if not lower - 1e-7 <= transfer <= upper + 1e-7:
        raise ValueError('Recorded phase partitions violate their fixed-solvent envelope')
    gap = None if reference_kj is None else max(lower-reference_kj, reference_kj-upper, 0.)
    return {'n_recorded_minima':len(minima), 'conditional_transfer_kj':transfer,
            'solvent_range_kj':[lower, upper], 'reference_kj':reference_kj,
            'reference_outside_range':None if gap is None else gap > 0,
            'minimum_central_discrepancy_under_reweighting_kj':gap,
            'uncertainty':None,
            'scope':'All recorded gas minima; fixed structures and solvent terms, arbitrary positive gas-state weights. No reference confidence, unsampled-state or predictive-coverage bound.'}


def component_factorial(cache, records, *, geometry_hashes=None):
    """Replace electronic/solvent terms separately and together, retaining RRHO.

    ``records`` supplies strict matched gas/SMD pairs at each required geometry.
    By default every recorded minimum is required. An explicit geometry list
    permits a bounded control, but its result never replaces the parent ensemble.
    Method-dependent populations are recomputed in all four arms. CDS is already
    included once in the recorded SMD potential and is never added separately.
    """
    minima = _source_minima(cache)
    if geometry_hashes is None:
        selected = minima
        scope = 'all_recorded_minima'
    else:
        if (not isinstance(geometry_hashes, (list, tuple)) or not geometry_hashes
                or len(set(geometry_hashes)) != len(geometry_hashes)):
            raise ValueError('Selection must be a nonempty list of unique geometry hashes')
        selected = [row for row in minima if row['geometry']['sha256'] in geometry_hashes]
        if len(selected) != len(geometry_hashes):
            raise ValueError('Selected geometry is absent from the parent ensemble')
        scope = 'selected_recorded_minima'
    selected_cache = copy.deepcopy(cache)
    selected_cache['meta']['thermal']['ensemble']['validated_minima'] = selected
    selected_cache['G'] = _partition([row['total_kj'] for row in selected])[0]
    solvent = reweight_smd(selected_cache, records)  # validates every replacement and its protocol
    values = {arm: [] for arm in ARMS}
    for row in selected:
        record = records[row['geometry']['sha256']]
        electronic = record['gas_Eh'] * HARTREE_KJ
        thermal = row['thermal_kj']
        values['original'].append(row['total_kj'])
        values['electronic_replaced'].append(electronic + row['solvation_kj'] + thermal)
        values['solvent_replaced'].append(row['electronic_kj'] + record['dgsolv'] + thermal)
        values['both_replaced'].append(electronic + record['dgsolv'] + thermal)
    arms = {}
    keys = [row['geometry']['sha256'] for row in selected]
    for arm, energies in values.items():
        energy, weights = _partition(energies)
        arms[arm] = {'G_kj': energy, 'populations': dict(zip(keys, weights))}
    first = next(iter(records.values()))
    protocol = {key: value for key, value in first['request'].items()
                if key not in ('symbols', 'coordinates_A', 'charge', 'spin_multiplicity')}
    return {'status': 'complete', 'sampling_scope': scope, 'parent_G_kj': cache['G'],
            'parent_n_minima': len(minima), 'n_evaluated_minima': len(selected), 'arms': arms,
            'source_settings_sha256': content_hash(cache.get('settings', {})),
            'replacement_protocol_sha256': content_hash(protocol),
            'replacement_backend_sha256': solvent['solvent_producer_sha256'],
            'uncertainty': None,
            'scope': 'Fixed recorded coordinates and original RRHO; independent component substitutions '
                     'with recomputed weights. Selected controls do not refine the parent ensemble. '
                     'No resampling, complete-state coverage or experimental accuracy established.'}


def compare_balanced_components(caches, records, stoichiometry):
    """Compare four component treatments in an atom/charge-balanced exchange.

    Every recorded minimum is required. A zero net molecular count is required
    so translational standard-state terms cancel; no proton, water or metal
    reservoir is supplied implicitly. Elemental energy zeros can then differ
    between methods without producing a fictitious reaction correction.
    """
    if (not isinstance(caches, Mapping) or not caches or not isinstance(records, Mapping)
            or not isinstance(stoichiometry, Mapping)
            or set(caches) != set(records) or set(caches) != set(stoichiometry)):
        raise ValueError('Every declared exchange species requires a cache, records and coefficient')
    if any(not _finite(value) or value == 0 for value in stoichiometry.values()):
        raise ValueError('Exchange coefficients must be finite, numeric and nonzero')
    balance = Counter()
    charge = 0.
    for name, cache in caches.items():
        _, molecule = _canonical(cache.get('smi'))
        coefficient = stoichiometry[name]
        for atom in Chem.AddHs(molecule).GetAtoms():
            balance[atom.GetSymbol()] += coefficient
        charge += coefficient * Chem.GetFormalCharge(molecule)
    if any(abs(value) > 1e-9 for value in balance.values()) or abs(charge) > 1e-9:
        raise ValueError('Exchange must conserve every element and charge without implicit reservoirs')
    if abs(math.fsum(stoichiometry.values())) > 1e-9:
        raise ValueError('Exchange must have zero net molecular count; no standard-state correction supplied')
    states = {name: component_factorial(cache, records[name]) for name, cache in caches.items()}
    for field in ('source_settings_sha256', 'replacement_protocol_sha256', 'replacement_backend_sha256'):
        if len({state[field] for state in states.values()}) != 1:
            raise ValueError('Exchange species require the same source settings and replacement protocol/backend')
    contrasts = {arm: {'balanced_exchange_kj': math.fsum(
        stoichiometry[name] * state['arms'][arm]['G_kj'] for name, state in states.items())}
        for arm in ARMS}
    original = contrasts['original']['balanced_exchange_kj']
    for contrast in contrasts.values():
        contrast['shift_from_original_kj'] = contrast['balanced_exchange_kj'] - original
    return {'status': 'complete', 'sampling_scope': 'all_recorded_minima', 'states': states,
            'stoichiometry': dict(stoichiometry), 'element_balance': dict(balance),
            'charge_balance': charge, 'contrasts': contrasts, 'uncertainty': None,
            'scope': 'Balanced fixed-coordinate component exchange with every recorded conformer reweighted. '
                     'No implicit reservoirs, selected-minimum replacements, complete aqueous-state coverage, '
                     'experimental accuracy, reaction offset or uncertainty guarantee.'}


def _skeleton(molecule):
    """Element graph with tetrahedral parity; independent of bond order/atom order."""
    graph = Chem.RWMol()
    for original in molecule.GetAtoms():
        atom = Chem.Atom(original.GetAtomicNum())
        atom.SetNoImplicit(True)
        atom.SetChiralTag(original.GetChiralTag())
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            atom.SetNumExplicitHs(original.GetTotalNumHs())
        graph.AddAtom(atom)
    for bond in molecule.GetBonds():
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            raise ValueError('Defined bond stereochemistry is not supported across these state pools')
        graph.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
    return Chem.MolToSmiles(graph.GetMol(), isomericSmiles=True)


def compare_state_ensembles(caches, records, *, reference_state, selections=None):
    """Pool every explicitly declared same-formula/charge state in each method arm.

    This is a conditional fixed-proton-count partition. Different charge states,
    hydration, epimers and overlapping/resonance-equivalent states are refused.
    No assumed state gaps, extra degeneracies, pKas or proton reservoirs enter.
    The caller must predeclare the state inventory; this function cannot prove
    that unlisted chemical states or missing conformer basins are negligible.
    """
    if (not isinstance(caches, Mapping) or not caches or not isinstance(records, Mapping)
            or set(caches) != set(records) or reference_state not in caches
            or (selections is not None and set(selections) != set(caches))):
        raise ValueError('Every declared state requires its cache, replacements and optional selection')
    identity = None
    settings = None
    protocol = None
    backend = None
    seen = set()
    comparisons = {}
    for name, cache in caches.items():
        canonical, molecule = _canonical(cache.get('smi'))
        if inspect_stereochemistry(canonical)['issues']:
            raise ValueError('Unresolved stereochemistry is not a declared state pool')
        state_identity = (rdMolDescriptors.CalcMolFormula(molecule), Chem.GetFormalCharge(molecule),
                          _skeleton(molecule))
        fixed_h = inchi.MolToInchi(molecule, options='/FixedH')
        if not fixed_h or fixed_h in seen:
            raise ValueError('Duplicate or resonance-equivalent states cannot count twice')
        seen.add(fixed_h)
        if identity is not None and state_identity != identity:
            raise ValueError('States must retain formula, charge, heavy skeleton and tetrahedral identity')
        identity = state_identity
        report = component_factorial(cache, records[name], geometry_hashes=None if selections is None else selections[name])
        signatures = (report['source_settings_sha256'], report['replacement_protocol_sha256'],
                      report['replacement_backend_sha256'])
        if settings is not None and signatures != (settings, protocol, backend):
            raise ValueError('State pools require identical source settings and replacement protocols/producers')
        settings, protocol, backend = signatures
        comparisons[name] = report
    pools = {}
    names = list(comparisons)
    for arm in ARMS:
        energies = [comparisons[name]['arms'][arm]['G_kj'] for name in names]
        energy, weights = _partition(energies)
        reference = comparisons[reference_state]['arms'][arm]['G_kj']
        pools[arm] = {'G_kj': energy, 'relative_to_reference_kj': energy - reference,
                      'populations': dict(zip(names, weights)),
                      'state_gaps_from_reference_kj': dict(zip(names, [g - reference for g in energies]))}
    return {'status': 'complete', 'reference_state': reference_state, 'states': comparisons, 'pools': pools,
            'sampling_scope': 'all_recorded_minima' if selections is None else 'selected_recorded_minima',
            'uncertainty': None,
            'scope': 'Conditional declared fixed-charge state set; no unlisted-state bound, experimental '
                     'speciation validation, production correction or accuracy claim.'}
