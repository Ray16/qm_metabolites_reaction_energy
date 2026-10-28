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
