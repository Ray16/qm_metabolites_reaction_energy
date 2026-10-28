"""Reaction-level assembly of supported ring-proton state ensembles.

Raw species physics stays in the energy backend. The callback must supply the
same standard-state, hydration and thermal treatment for every declared state.
"""
import math

from metag.routing.ring_protons import enumerate_ring_protons, POLICY_VERSION
from metag.energetics.aqueous_states import mixture_sampling_uncertainty

KT_KJ = 2.4789  # identical to the production species partition convention


def prepare_ring_pools(reaction, routes, *, explicit=False, ring_chain=False):
    """Validate supported state sets and protonation-factor compatibility on CPU."""
    from metag.routing import pka_transform as pk
    plans = {}
    for name, (_, charge, smiles) in reaction['species'].items():
        states = enumerate_ring_protons(smiles, charge)
        if not states.applicable:
            continue
        if not states.complete:
            raise ValueError(name + ': ' + str(states.reason))
        if ring_chain:
            raise ValueError('Joint ring-chain/ring-proton enumeration is unsupported')
        requested = name in explicit if isinstance(explicit, (list, set, tuple)) else bool(explicit)
        if requested:
            raise ValueError('Ring-proton pools require implicit state energies: ' + name)
        if reaction.get('pka_sites') and (
                not isinstance(routes.get('protonation_sites_by_species'), dict)
                or name not in routes['protonation_sites_by_species']):
            raise ValueError('Ring-proton pool has no owned protonation terms: ' + name)
        signatures = []
        for state in states.states:
            neutral, acids, netq = pk._neutralize(state)
            neutral_v2, acids_v2, bases, netq_v2 = pk._neutralize_v2(state)
            if neutral is None or neutral_v2 is None:
                raise ValueError('Ring-proton state has no compatible existing protonation reference: ' + name)
            signatures.append((tuple(sorted(acids)), netq, tuple(sorted(acids_v2)),
                               tuple(sorted(bases)), netq_v2))
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError('Ring-proton states have different protonation factors; a joint model is required: ' + name)
        plans[name] = {'policy': POLICY_VERSION, 'reference': states.reference,
                       'states': list(states.states), 'state_set_sha256': states.identity,
                       'charge': charge, 'protonation_factor_signature': signatures[0]}
    return plans


def apply_ring_pools(plans, species, energies, sigmas, *, evaluate_state):
    """Evaluate all states, refusing incomplete pools; return energies/diagnostics.

    Repeated occurrences of a molecular state share one estimate. Reference and
    alternate drawings of the same complete state set therefore receive exactly
    the same pooled energy and sampling spread. No extra degeneracy is added.
    """
    available = {}
    for name, plan in plans.items():
        key = (plan['charge'], plan['reference'])
        entry = (energies[name], sigmas[name])
        if key in available and available[key] != entry:
            raise ValueError('Repeated ring-proton reference has inconsistent energy/sampling estimates')
        available[key] = entry
    pools, diagnostics = {}, {}
    output_g, output_s = dict(energies), dict(sigmas)
    for name, plan in plans.items():
        family = plan['state_set_sha256']
        if family not in pools:
            rows = []
            for state in plan['states']:
                key = (plan['charge'], state)
                source = 'scored_reference'
                if key not in available:
                    value, sigma, detail = evaluate_state(name, plan['charge'], state)
                    source = 'evaluated_state'
                    available[key] = (value, sigma)
                else:
                    value, sigma = available[key]
                    detail = {}
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                       for v in (value, sigma)) or sigma < 0:
                    raise ValueError('Incomplete ring-proton ensemble: ' + name + ' / ' + state)
                rows.append({'smiles': state, 'G_kj': value, 'sampling_sigma_kj': sigma,
                             'source': source, 'diagnostics': detail})
            low = min(row['G_kj'] for row in rows)
            normalizer = math.fsum(math.exp(-(row['G_kj'] - low) / KT_KJ) for row in rows)
            pooled = low - KT_KJ * math.log(normalizer)
            sigma, uncertainty = mixture_sampling_uncertainty(
                [row['G_kj'] for row in rows], [row['sampling_sigma_kj'] for row in rows], KT_KJ)
            for row, population in zip(rows, uncertainty['conditional_populations']):
                row['conditional_population'] = population
            pools[family] = (pooled, sigma, rows, uncertainty)
        pooled, sigma, rows, uncertainty = pools[family]
        output_g[name], output_s[name] = pooled, sigma
        diagnostics[name] = {'policy': POLICY_VERSION, 'state_set_sha256': family,
                             'reference_smiles': plan['reference'], 'reference_G_kj': energies[name],
                             'pooled_G_kj': pooled, 'mixture_shift_kj': pooled - energies[name],
                             'states': rows, 'sampling_uncertainty': uncertainty,
                             'scope': 'Complete supported ring-N proton subspace at fixed formula/charge; '
                                      'same declared independent-site factors. Other tautomers, charge states '
                                      'and unsampled conformer basins remain unbounded; no predictive coverage claim.'}
    return output_g, output_s, diagnostics


def prepare_sugar_pools(reaction, routes, *, explicit=False, caller_pka_sites=False,
                        ring_chain=False, ring_protons=False):
    """Validate full routed sugar bases and species-owned pKa bookkeeping on CPU."""
    from collections import Counter
    from rdkit import Chem
    from metag.routing.sugar_states import enumerate_sugar_states, canon, POLICY
    from metag.routing import pka_transform as pk
    from metag.conditions import PH, RT_LN10
    from metag.energetics.protonation import normalize_site, site_free_energy
    declarations = routes.get('declared_sugar_pools', {})
    if not declarations:
        return {}
    if caller_pka_sites:
        raise ValueError('Flat caller pKa terms have no owned joint sugar reference')
    if ring_chain or ring_protons:
        raise ValueError('Joint sugar/ring-chain/ring-proton state model is unsupported')
    if routes.get('truncated'):
        raise ValueError('A truncated drawing cannot represent a declared full sugar pool')
    coupled = (routes.get('coupled_protonation_models') or {}).get('species', {})
    plans = {}
    for name, declaration in declarations.items():
        if name not in reaction['species']:
            raise ValueError('Routing removed a declared sugar pool: ' + name)
        if name in coupled:
            raise ValueError('Joint macroscopic protonation/sugar reference not declared: ' + name)
        requested = name in explicit if isinstance(explicit, (list, set, tuple)) else bool(explicit)
        if requested:
            raise ValueError('Sugar pools require the same implicit-state treatment: ' + name)
        coefficient, charge, smiles = reaction['species'][name]
        enumeration = enumerate_sugar_states(smiles, charge)
        if not enumeration.applicable or not enumeration.complete:
            raise ValueError('Routed sugar basis incomplete: ' + name + ' / ' + str(enumeration.reason))
        states = [row['smiles'] for row in enumeration.states]
        reference = canon(Chem.MolFromSmiles(smiles))
        use_factors = bool(routes.get('ph0') or routes.get('zwitterion_ph0')
                           or routes.get('routing_mode') == 'compound_reference')
        owned = Counter()
        if use_factors:
            ownership = routes.get('protonation_sites_by_species')
            if not isinstance(ownership, dict) or name not in ownership:
                raise ValueError('Sugar reference has no owned protonation terms: ' + name)
            side = 'react' if coefficient < 0 else 'prod'
            for site in ownership[name]:
                actual_side, pka, kind, multiplicity = normalize_site(site)
                if actual_side != side:
                    raise ValueError('Sugar protonation ownership has the wrong reaction side: ' + name)
                owned[(kind, float(pka))] += multiplicity / abs(coefficient)
        # Follow the actual owned route: an acid-only reference can retain a
        # protonated amine and need not have zero net charge. Never introduce an
        # unowned base transform merely to force every reference to be neutral.
        use_bases = any(kind == 'base' for kind, pka in owned)
        def reference_terms(state):
            if not use_factors:
                return state, [], [], Chem.GetFormalCharge(Chem.MolFromSmiles(state))
            if use_bases:
                return pk._neutralize_v2(state)
            mapped, acids, netq = pk._neutralize(state)
            return mapped, acids, [], netq
        expected = set()
        for row in declaration['source_states']:
            mapped, acids, bases, netq = reference_terms(row['smiles'])
            if mapped is None or netq != charge:
                raise ValueError('Unsupported charge-reference change for sugar: ' + name)
            expected.add(canon(Chem.MolFromSmiles(mapped)))
        if expected != set(states):
            raise ValueError('Routing changed the declared sugar skeleton/state basis: ' + name)
        factors, signatures = {}, {}
        for state in states:
            mapped, acids, bases, netq = reference_terms(state)
            if mapped is None or netq != charge or canon(Chem.MolFromSmiles(mapped)) != state:
                raise ValueError('Sugar state does not match its owned reference model: ' + name)
            signatures[state] = Counter([('acid', float(p)) for p in acids]
                                        + [('base', float(p)) for p in bases])
            factors[state] = math.fsum(site_free_energy(p, kind, PH, RT_LN10)
                                      for kind, values in [('acid', acids), ('base', bases)]
                                      for p in values)
        expected_signature = signatures[reference]
        if (set(owned) != set(expected_signature)
                or any(not math.isclose(owned[k], expected_signature[k], rel_tol=0., abs_tol=1e-12)
                       for k in owned)):
            raise ValueError('Sugar reference pKa terms differ from its owned model: ' + name)
        plans[name] = {'policy':POLICY, 'charge':charge, 'reference':reference,
                       'states':states, 'state_factors_kj':factors,
                       'reference_model':'acid_and_base' if use_bases else 'acid_only' if use_factors else 'fixed_charge',
                       'source_state_set_sha256':declaration['source_state_set_sha256'],
                       'routed_state_set_sha256':enumeration.state_set_sha256,
                       'source':declaration['source']}
    return plans


def apply_sugar_pools(plans, species, energies, sigmas, *, evaluate_state):
    """Assemble complete declared sugar families with owned protonation factors.

    Plans are constructed from source-defined structures and the configured
    species protonation model. ``state_factors_kj`` contain that model's whole
    independent-site contribution for each state. Reference factors are removed
    here because reaction assembly already adds them once. This is bookkeeping,
    not a fitted energy offset. No missing state is renormalized away.
    """
    from metag.energetics.provenance import content_hash
    available = {}
    for name, plan in plans.items():
        if name not in species or name not in energies or name not in sigmas:
            raise ValueError('Declared sugar reference is missing: ' + name)
        key = (plan['charge'], plan['reference'])
        value = (energies[name], sigmas[name])
        if key in available and available[key] != value:
            raise ValueError('Repeated sugar reference has inconsistent estimates')
        available[key] = value
    pooled, output_g, output_s, diagnostics = {}, dict(energies), dict(sigmas), {}
    for name, plan in plans.items():
        states = list(plan['states'])
        factors = plan['state_factors_kj']
        if (not states or len(set(states)) != len(states)
                or set(states) != set(factors) or plan['reference'] not in states):
            raise ValueError('A complete unique sugar state basis and owned factors are required')
        if any(isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(f)
               for f in factors.values()):
            raise ValueError('Finite owned sugar protonation factors required')
        family = content_hash({'charge':plan['charge'],
                               'states':sorted(states), 'state_factors_kj':factors})
        if family not in pooled:
            rows = []
            for smiles in states:
                key = (plan['charge'], smiles)
                if key in available:
                    value, sigma = available[key]
                    detail, origin = {}, 'scored_or_shared_state'
                else:
                    value, sigma, detail = evaluate_state(name, plan['charge'], smiles)
                    available[key] = value, sigma
                    origin = 'evaluated_state'
                if (any(isinstance(x, bool) or not isinstance(x, (int, float))
                        or not math.isfinite(x) for x in (value, sigma)) or sigma < 0):
                    raise ValueError('Incomplete sugar ensemble: ' + name + ' / ' + smiles)
                rows.append({'smiles':smiles, 'G_before_common_transform_kj':value,
                             'owned_protonation_factor_kj':factors[smiles],
                             'transformed_G_kj':value + factors[smiles],
                             'sampling_sigma_kj':sigma, 'source':origin, 'diagnostics':detail})
            low = min(row['transformed_G_kj'] for row in rows)
            normalization = math.fsum(math.exp(-(row['transformed_G_kj'] - low)/KT_KJ)
                                      for row in rows)
            total = low - KT_KJ*math.log(normalization)
            sigma, uncertainty = mixture_sampling_uncertainty(
                [row['transformed_G_kj'] for row in rows],
                [row['sampling_sigma_kj'] for row in rows], KT_KJ)
            for row, population in zip(rows, uncertainty['conditional_populations']):
                row['conditional_population'] = population
            pooled[family] = total, sigma, rows, uncertainty
        total, sigma, rows, uncertainty = pooled[family]
        reference_factor = factors[plan['reference']]
        value = total - reference_factor
        if value > energies[name] + 1e-7:
            raise ValueError('Complete sugar pool cannot exceed its included reference state')
        output_g[name], output_s[name] = value, sigma
        diagnostics[name] = {'policy':plan['policy'], 'state_set_sha256':family,
                             'source_state_set_sha256':plan['source_state_set_sha256'],
                             'reference_smiles':plan['reference'],
                             'reference_G_before_transform_kj':energies[name],
                             'reference_owned_factor_kj':reference_factor,
                             'pooled_transformed_G_kj':total,
                             'pooled_G_before_reference_transform_kj':value,
                             'mixture_shift_kj':value - energies[name], 'states':rows,
                             'sampling_uncertainty':uncertainty,
                             'scope':'Complete declared ring/open state basis under configured hydration '
                                     'and owned independent-site model. Sampling diagnostics only; '
                                     'no universal state completeness, model-error bound or predictive interval.'}
    return output_g, output_s, diagnostics
