"""Audit measured reference conditions without inventing missing activities.

Total magnesium concentration is not free Mg2+ activity. Missing pMg cannot be
interpreted as absent magnesium when the experimental recipe includes it.
"""
import math
import re
from decimal import Decimal, InvalidOperation

TABLE_KEY = ('part', 'page', 'col l/r', 'table from top')
_MG = re.compile(r'\b(?i:magnesium)\b|\bMg(?=\b|[A-Z0-9])')
_NEGATIVE_MG = re.compile(r'(?:no|without|absence of|free of)\s+(?:added\s+)?(?:magnesium|Mg\b)|(?:magnesium|Mg)[ -]free', re.I)


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def table_key(row):
    """Join spreadsheet coordinates consistently across integer/decimal exports."""
    result = []
    for field in TABLE_KEY:
        text = str(row.get(field, '')).strip()
        try:
            value = Decimal(text)
            text = str(value.normalize()) if value.is_finite() else text
        except InvalidOperation:
            pass
        result.append(text)
    return tuple(result)


def magnesium_evidence(row, metadata=None):
    """Return affirmative recipe mentions, not a guessed free concentration.

    Comments can discuss magnesium hypothetically, so only recipe fields trigger
    the exclusion. Negated recipe mentions remain available in the raw audit.
    """
    metadata = metadata or {}
    found = []
    for source, data, fields in [('row', row, ('Cofactor', 'buffer')),
                                 ('metadata', metadata, ('Cofactor', 'buffer'))]:
        for field in fields:
            value = str(data.get(field, ''))
            for clause in re.split(r'[;\n]', value):
                if _MG.search(clause) and not _NEGATIVE_MG.search(clause):
                    found.append({'source': source, 'field': field, 'text': clause.strip()})
    return found


def magnesium_exclusion(row, metadata=None):
    if number(row.get('p_mg')) is None and magnesium_evidence(row, metadata):
        return 'magnesium present but free magnesium activity/pMg unknown'
    return None


def audit_conditions(row, metadata=None, comments=None, target_temperature=298.15):
    metadata, comments = metadata or {}, comments or {}
    issues = []
    evidence = magnesium_evidence(row, metadata)
    pmg = number(row.get('p_mg'))
    if pmg is None:
        issues.append('magnesium_present_free_activity_unknown' if evidence else 'magnesium_condition_unspecified')
    ambiguity = magnesium_total_activity_ambiguity(row, metadata)
    if ambiguity:
        issues.append('pMg_matches_total_magnesium_recipe')
    temperature = number(row.get('temperature'))
    if temperature is None:
        issues.append('temperature_missing')
    elif abs(temperature - target_temperature) > 1e-6:
        issues.append('temperature_differs_from_prediction')
    if number(row.get('ionic_strength')) is None:
        issues.append('ionic_strength_missing')
    if number(row.get('p_h')) is None:
        issues.append('ph_missing')
    text = ' '.join(str(d.get(k, '')) for d, fields in [(row, ('additional data',)),
                      (metadata, ('method', 'curator comment')), (comments, ('comment',))] for k in fields)
    if re.search(r'kinetic|Haldane', text, re.I):
        issues.append('kinetic_derived_equilibrium_or_method_requires_review')
    for field in ('todo_check_in_primary_literature', 'misannotated_value_type', 'wrong_value',
                  'missing_value_to_be_added', 'value_given_with_approximate_sign'):
        value = str(row.get(field, '')).strip()
        if value and value.lower() not in ('0', '0.0', 'false', 'no'):
            issues.append('curation_' + field)
    if not metadata:
        issues.append('table_metadata_missing')
    return {'table_key': list(table_key(row)), 'issues': issues,
            'exclusion': ('pMg matches total magnesium recipe; source free activity required' if ambiguity
                          else magnesium_exclusion(row, metadata)),
            'magnesium_evidence': evidence, 'magnesium_total_activity_ambiguity': ambiguity,
            'temperature_K': temperature, 'target_temperature_K': target_temperature,
            'pMg': pmg, 'ionic_strength_M': number(row.get('ionic_strength')),
            'raw_row': dict(row), 'table_metadata': dict(metadata), 'table_comments': dict(comments)}


def magnesium_total_activity_ambiguity(row, metadata=None):
    """Flag a pMg field numerically copied from an added Mg-salt concentration.

    This is an ambiguity check, not an inferred free activity or a correction.
    Only explicit c(Mg-salt)=... M / mol dm-3 recipes are parsed. A source-backed
    `p_mg_basis=free_activity` declaration with `p_mg_source` can resolve it.
    Molal recipes and prose without units require separate source curation.
    """
    pmg = number(row.get('p_mg'))
    if pmg is None:
        return []
    if (row.get('p_mg_basis') == 'free_activity' and
            isinstance(row.get('p_mg_source'), str) and row['p_mg_source'].strip()):
        return []
    pattern = re.compile(r'\bc\s*\(\s*Mg(?:Cl2|SO4)?\s*\)\s*=\s*'
                         r'([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*'
                         r'(mM|M|mol\s*(?:dm\^?-?3|/\s*[lL]))\b')
    found = []
    for origin, data in [('row', row), ('metadata', metadata or {})]:
        for field in ('additional data', 'Cofactor', 'buffer'):
            text = str(data.get(field, ''))
            for match in pattern.finditer(text):
                concentration = float(match.group(1)) * (0.001 if match.group(2) == 'mM' else 1)
                if concentration > 0 and abs(pmg + math.log10(concentration)) <= 0.051:
                    found.append({'source': origin, 'field': field, 'text': match.group(0),
                                  'added_salt_M': concentration, 'reported_pMg': pmg,
                                  'reason': 'pMg matches negative log of added total Mg salt; free activity not established'})
    return found
