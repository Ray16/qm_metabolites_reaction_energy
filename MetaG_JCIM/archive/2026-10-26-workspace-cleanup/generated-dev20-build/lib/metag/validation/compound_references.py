"""Audit owned compound-reference transforms without fitting/projection."""
from collections import defaultdict
import json
import math

from metag.chem import is_balanced
from metag.energetics.protonation import normalize_site
from metag.validation.routing import canonical


def audit_compound_references(original, records):
    uses = defaultdict(list)
    unknown, errors = [], []
    for rid,reaction in original.items():
        row = records.get(rid,{})
        route = row.get('routes') or {}
        owned = route.get('compound_references')
        if route.get('compound_reference_status') != 'complete' or not isinstance(owned,dict):
            unknown.append(rid)
            continue
        scored = row.get('species_scored')
        if not isinstance(scored,dict) or set(owned) != set(reaction['species']) or set(scored) != set(owned):
            errors.append({'reaction':rid,'reason':'reference_ownership_or_scored_labels_mismatch'})
            continue
        expected_terms = []
        expected_hplus = reaction.get('n_Hplus',0)
        for name,(coefficient,charge,smiles) in reaction['species'].items():
            model = owned[name]
            try:
                if model['input_charge'] != charge or canonical(model['input_smiles']) != canonical(smiles):
                    raise ValueError('reference_input_mismatch')
                if (canonical(scored[name][2]) != canonical(model['reference_smiles'])
                        or scored[name][1] != model['reference_charge'] or scored[name][0] != coefficient):
                    raise ValueError('reference_scored_identity_mismatch')
                uptake = model['proton_uptake']
                if isinstance(uptake,bool) or not isinstance(uptake,int):
                    raise ValueError('reference_proton_count_invalid')
                exchange = {'original':[-1,charge,smiles],
                            'reference':[1,model['reference_charge'],model['reference_smiles']]}
                if not is_balanced(exchange,-uptake):
                    raise ValueError('reference_proton_exchange_does_not_balance')
                expected_hplus -= coefficient*uptake
                for kind,key in [('acid','acid_pkas'),('base','base_pkas')]:
                    for pka in model[key]:
                        side='react' if coefficient<0 else 'prod'
                        if coefficient:
                            term=[side,pka,kind,abs(coefficient)]
                            normalize_site(term)
                            expected_terms.append(term)
                signature = json.dumps({key:model[key] for key in (
                    'policy','reference_smiles','reference_charge','proton_uptake',
                    'acid_pkas','base_pkas','pka_provider')},sort_keys=True,allow_nan=False)
                if coefficient:
                    uses[(canonical(smiles),charge)].append({'reaction':rid,'signature':signature})
            except (KeyError,TypeError,ValueError) as exc:
                errors.append({'reaction':rid,'species':name,'reason':str(exc)})
        actual_terms=row.get('pka_sites_scored') or []
        try:
            if reaction.get('species_protonation_models'):
                from metag.conditions import CONDITIONS, RT_LN10
                from metag.energetics.protonation_models import prepare_models
                expected_terms, _, _ = prepare_models(
                    reaction['species_protonation_models'], scored, expected_terms,
                    route.get('protonation_sites_by_species'), CONDITIONS, rt_ln10=RT_LN10)
            same_terms=sorted(normalize_site(site) for site in actual_terms)==sorted(normalize_site(site) for site in expected_terms)
        except (ValueError,TypeError,KeyError):
            same_terms=False
        if not same_terms:
            errors.append({'reaction':rid,'reason':'owned_reference_terms_differ_from_scored_terms'})
        reported_hplus=row.get('n_Hplus_scored')
        if (isinstance(reported_hplus,bool) or not isinstance(reported_hplus,(int,float))
                or not math.isfinite(reported_hplus)
                or not math.isclose(expected_hplus,reported_hplus,rel_tol=0,abs_tol=1e-9)):
            errors.append({'reaction':rid,'reason':'owned_reference_proton_bookkeeping_mismatch'})
    conflicts=[]
    for (smiles,charge),entries in uses.items():
        groups=defaultdict(list)
        for entry in entries:
            groups[entry['signature']].append(entry['reaction'])
        if len(groups)>1:
            conflicts.append({'input_smiles':smiles,'input_charge':charge,
                              'references':[{'model':json.loads(signature),'reactions':sorted(set(rids))}
                                            for signature,rids in groups.items()]})
    return {'n_compounds_with_owned_references':len(uses),
            'n_context_dependent_reference_models':len(conflicts),'conflicts':conflicts,
            'unavailable_reactions':sorted(unknown),'errors':errors,
            'passed':bool(original) and not unknown and not errors and not conflicts,
            'scope':'Owned identities, proton exchange and scored independent-site terms; consistent algebra does not validate microconstants, class transfer, populations or absolute energies.'}
