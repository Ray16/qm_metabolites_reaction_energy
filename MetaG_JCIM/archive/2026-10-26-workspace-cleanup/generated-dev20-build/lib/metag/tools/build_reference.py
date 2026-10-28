"""Rebuild reference labels with explicit exclusions and per-measurement provenance.

Unknown Mg conditions never imply no magnesium. This tool retains reaction
structures even when no valid reference remains, with exp=[] and a reason.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
from collections import Counter, defaultdict

from metag.validation import reference_transform as B
from metag.validation.reference_conditions import audit_conditions, table_key, number, magnesium_evidence, magnesium_total_activity_ambiguity

ALIAS = {
    "2 glutathione(red)(ox) + NADP(aq) = glutathione(ox)(aq) + NADPH(aq)":
        "2 kegg:C00051 + kegg:C00006 = kegg:C00127 + kegg:C00005",
    "inosine(aq) + orthophosphate(aq) = hypoxanthine(aq) + alpha-D-ribose 1-phosphate(aq)":
        "kegg:C00294 + kegg:C00009 = kegg:C00262 + kegg:C00620",
    "CoA(aq) + acetate(aq) + ATP(aq) = acetyl-CoA(aq) + ADP(aq) + orthophosphate(aq)":
        "kegg:C00010 + kegg:C00033 + kegg:C00002 = kegg:C00024 + kegg:C00008 + kegg:C00009",
}


def _norm(name):
    return re.sub(r"\s+\(", "(", " ".join((name or "").split()))


def _key(row):
    return table_key(row)


def opentecr_rows(source):
    """openTECR rows with a KEGG equation attached: [(row, kegg_equation, how)]."""
    old = list(csv.DictReader(open(source / "cc_training_TECRDB.csv")))
    rows = list(csv.DictReader(open(source / "sheet_actual_data.csv")))
    metadata = {_key(m): m for m in csv.DictReader(open(source / "sheet_table_metadata.csv"))}
    comments_path = source / "sheet_table_comments.csv"
    comments = {_key(m): m for m in csv.DictReader(open(comments_path))}
    table_rxn = {key: _norm(m["reaction"]) for key, m in metadata.items() if _norm(m["reaction"])}
    for row_number, row in enumerate(rows, 2):
        row["_source_row"] = row_number
        row["_reference_metadata"] = metadata.get(_key(row), {})
        row["_reference_comments"] = comments.get(_key(row), {})
    name2kegg = defaultdict(set, {_norm(k): {v} for k, v in ALIAS.items()})
    kegg2ec = defaultdict(set)                                        # EC guard for name-mapped rows
    for o in old:
        kegg2ec[o["reaction"]].update(e.strip() for e in o["EC"].split("&"))
    for r in rows:
        if "#entry" in r["id"]:
            o = old[int(r["id"].split("#entry")[1]) - 1]
            name2kegg[_norm(r["reaction"])].add(o["reaction"])
            name2kegg[_norm(o["description"])].add(o["reaction"])
    out, how = [], Counter()
    for r in rows:
        name = _norm(r["reaction"]) or table_rxn.get(_key(r), "")
        if "#entry" in r["id"]:
            out.append((r, old[int(r["id"].split("#entry")[1]) - 1]["reaction"], "entry id")); how["entry id"] += 1
        elif len(name2kegg.get(name, ())) == 1:
            keq = next(iter(name2kegg[name]))
            # a blank reaction cell inherits the table's reaction; openTECR has tables whose reaction text does not
            # match their enzyme (64BOJ/GAU: EC 2.1.3.5 oxamate carbamoyltransferase labelled as the ornithine
            # reaction), so a row with an EC must share an EC component with what TECRDB records for that reaction
            ec = {e.strip() for e in (r["EC"] or "").split("&") if e.strip()}
            if ec and not ec & kegg2ec[keq]:
                how["unmapped (EC disagrees with reaction)"] += 1; continue
            out.append((r, keq, "name")); how["name"] += 1
        else:
            how["unmapped (ambiguous)" if name in name2kegg else "unmapped (name not in TECRDB)"] += 1
    return out, how


def flag(value):
    return str(value or '').strip().lower() not in {'', '0', '0.0', 'false', 'no'}


def measurement_exclusion(row, metadata=None):
    """Generic eligibility independent of reaction identity or prediction error."""
    for key in ('solvent_reaction', 'wrong_value', 'chemical_reference_reaction', 'virtual_entry',
                'misannotated_value_type', 'todo_check_in_primary_literature', 'missing_value_to_be_added'):
        if flag(row.get(key)):
            return key
    if flag(row.get('duplicate_table')) and not flag(row.get('authoritative_version_of_a_duplicate_table')):
        return 'duplicate_table'
    if row.get('type of Kprime', '').strip() in {'K', 'Kc'}:
        return 'species_level_constant'
    k = number(row.get('K_prime'))
    if k is None or k <= 0:
        return 'missing_or_invalid_Kprime'
    if number(row.get('p_h')) is None:
        return 'missing_pH'
    t = number(row.get('temperature'))
    if t is None or t <= 0:
        return 'missing_or_invalid_temperature'
    if number(row.get('p_mg')) is None:
        # Recipe silence, '-' and total Mg concentrations are not free activities.
        return ('Mg_present_free_activity_unknown' if magnesium_evidence(row, metadata)
                else 'Mg_condition_unspecified')
    if magnesium_total_activity_ambiguity(row, metadata):
        return 'pMg_matches_total_magnesium_recipe'
    ionic = number(row.get('ionic_strength'))
    if ionic is not None and ionic < 0:
        return 'invalid_ionic_strength'
    return None


def rebuild(inputs, source, modelseed, *, impute_i=0.25, transformer=None,
            mapping_policy="exclude_ambiguous", temperature_K=None):
    if mapping_policy not in ("exclude_ambiguous", "historical_first"):
        raise ValueError("unknown mapping policy")
    if temperature_K is not None and (not math.isfinite(temperature_K) or temperature_K <= 0):
        raise ValueError("temperature_K must be finite and positive")
    source, modelseed = Path(source), Path(modelseed)
    aliases = (B.kegg_to_modelseed(modelseed) if mapping_policy == "exclude_ambiguous"
               else B.kegg_to_modelseed(modelseed, ambiguity="historical_first"))
    ms = B.load_modelseed_reactions(modelseed)
    if not ms:
        raise ValueError('No ModelSEED reactions found')
    index = defaultdict(list)
    for rid, st in ms.items():
        index[B.signature(st, {B.PROTON, B.WATER})].append((rid, 1))
        index[B.signature({c: -v for c, v in st.items()}, {B.PROTON, B.WATER})].append((rid, -1))
    rows, mapping = opentecr_rows(source)
    tr = transformer if transformer is not None else B.Transformer(ms)
    measurements, used = [], defaultdict(list)
    counts, unmatched = Counter(), Counter()
    for row, equation, how in rows:
        st, ok = B.parse_equation(equation)
        if not ok or any(c not in aliases for c in st):
            unmatched['invalid_equation' if not ok else 'missing_or_ambiguous_compound_alias'] += 1
            continue
        mapped = defaultdict(float)
        for c, coefficient in st.items():
            mapped[aliases[c]] += coefficient
        hits = index.get(B.signature(mapped, {B.PROTON, B.WATER}), [])
        if not hits:
            unmatched['no_reaction_match'] += 1
            continue
        panel_hits = sorted({hit for hit in hits if hit[0] in inputs})
        if not panel_hits:
            continue
        # A representative ID locates an excluded record; it does NOT resolve
        # an ambiguity. Every candidate remains available for source adjudication.
        rid, orient = sorted(hits)[0] if mapping_policy == "historical_first" else panel_hits[0]
        if rid not in inputs:
            continue  # reproduce historical panel selection only in replay mode
        raw = {k: v for k, v in row.items() if not k.startswith('_')}
        record = dict(audit_conditions(raw, row['_reference_metadata'], row['_reference_comments']),
                      reaction=rid, orientation=orient, mapping=how, mapping_candidates=sorted(hits),
                      source_row=row['_source_row'], equation=equation)
        why = measurement_exclusion(raw, row['_reference_metadata'])
        if mapping_policy == "exclude_ambiguous" and len(set(hits)) != 1:
            why = 'ambiguous_reaction_mapping'
        if why is None and temperature_K is not None and not math.isclose(
                number(row.get('temperature')), temperature_K, rel_tol=0, abs_tol=1e-6):
            why = 'temperature_mismatch'
        if why is None:
            try:
                if tr.reaction(rid) is None:
                    why = 'missing_balanced_transformable_reaction'
                else:
                    t, ph, pmg = (number(row[k]) for k in ('temperature', 'p_h', 'p_mg'))
                    ionic = number(row.get('ionic_strength'))
                    if ionic is None and impute_i is None:
                        why = 'ionic_strength_unspecified'
                    else:
                        actual_i = impute_i if ionic is None else ionic
                        observed = orient * -B.R_KJ * t * math.log(number(row['K_prime']))
                        correction = tr.correction(rid, ph, actual_i, pmg, t)
                        reference = observed + correction
                        if not math.isfinite(reference):
                            raise ValueError('nonfinite condition transform')
                        record.update(observed_dG=observed, condition_correction=correction,
                                      reference_dG=reference, transform_I_M=actual_i,
                                      ionic_strength_imputed=ionic is None)
            except (ValueError, RuntimeError, KeyError, Warning) as exc:
                why = 'condition_transform_failed'
                record['transform_error'] = str(exc)
        record['selection'] = 'excluded' if why else 'used'
        record['exclusion_reason'] = why
        measurements.append(record)
        if why:
            counts[why] += 1
        else:
            used[rid].append(record)
    sources = [*sorted(source.glob('*.csv')), modelseed/'Biochemistry/Aliases/Unique_ModelSEED_Compound_Aliases.txt',
               *sorted((modelseed/'Biochemistry').glob('reaction_??.tsv'))]
    hashes = {('snapshot/' + p.name if p.parent == source else 'modelseed/' + str(p.relative_to(modelseed))):
              hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    transform_provenance = tr.provenance() if hasattr(tr, 'provenance') else {'method': 'injected_test_transformer'}
    code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (Path(__file__), Path(B.__file__),
                             Path(__file__).parents[1] / "validation/reference_conditions.py")}
    policy = {'version': '2026-10-10-source-condition-ambiguity', 'target': B.STD, 'impute_I_M': impute_i,
              'mapping_policy': mapping_policy, 'require_temperature_K': temperature_K,
              'temperature': 'per measurement; not normalized to 298.15 K',
              'missing_pMg': 'exclude; never assume Mg-free',
              'total_Mg_pMg_ambiguity': 'exclude numerical copies of added salt concentration unless source-backed free activity is declared', 'unresolved_curation': 'exclude'}
    reference_id = hashlib.sha256(json.dumps({'sources': hashes, 'policy': policy, 'transform': transform_provenance, 'code': code_hashes}, sort_keys=True).encode()).hexdigest()
    output, changes = {}, {}
    for rid, rx in inputs.items():
        selected = used.get(rid, [])
        clean = {k: v for k, v in rx.items() if k != 'exp' and not k.startswith('exp_')}
        values = [r['reference_dG'] for r in selected]
        temperatures = sorted({r['temperature_K'] for r in selected})
        label = [round(statistics.median(values), 2)] if values else []
        reasons = Counter(r['exclusion_reason'] for r in measurements
                          if r['selection'] == 'excluded' and (r['reaction'] == rid or
                          (r['exclusion_reason'] == 'ambiguous_reaction_mapping' and
                           any(candidate[0] == rid for candidate in r['mapping_candidates']))))
        clean.update(exp_exclusions=dict(reasons), exp=label, exp_n=len(values), exp_reference_id=reference_id,
                     exp_status='available_with_declared_conditions' if values else 'unresolved_reference',
                     exp_conditions='pH 7, I 0, pMg 14; temperature as measured (not a 298.15 K normalization)',
                     exp_temperatures_K=temperatures,
                     exp_I_imputed=any(r['ionic_strength_imputed'] for r in selected),
                     exp_sd=round(statistics.stdev(values), 2) if len(values)>1 else (0.0 if values else None),
                     exp_source_rows=[r['source_row'] for r in selected])
        output[rid] = clean
        if label != rx.get('exp', []):
            changes[rid] = {'before': rx.get('exp', []), 'after': label, 'n_used': len(values)}
    report = {'policy': policy, 'reference_id': reference_id, 'source_sha256': hashes,
              'mapping': dict(mapping), 'transform_provenance': transform_provenance, 'code_sha256': code_hashes,
              'measurements': measurements, 'changed_labels': changes,
              'summary': {'reactions':len(output), 'labelled':sum(bool(v['exp']) for v in output.values()),
                          'unresolved':sum(not v['exp'] for v in output.values()),
                          'measurements_used':sum(len(v) for v in used.values()), 'excluded':dict(counts),
                          'unmatched_source_rows':dict(unmatched)}}
    return output, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--modelseed', type=Path, required=True)
    parser.add_argument('--reactions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--impute-I', type=float, default=0.25)
    parser.add_argument('--require-reported-I', action='store_true')
    parser.add_argument('--temperature-K', type=float, help='retain only measurements at this temperature; no temperature extrapolation')
    parser.add_argument('--mapping-policy', choices=['exclude_ambiguous', 'historical_first'],
                        default='exclude_ambiguous', help='historical_first is for legacy reproduction only')
    args = parser.parse_args(argv)
    if args.impute_I < 0 or not math.isfinite(args.impute_I):
        parser.error('--impute-I must be finite and nonnegative')
    inputs = json.loads(args.reactions.read_text())
    output, report = rebuild(inputs, args.source, args.modelseed,
                             impute_i=None if args.require_reported_I else args.impute_I,
                             mapping_policy=args.mapping_policy, temperature_K=args.temperature_K)
    report['input_sha256'] = hashlib.sha256(args.reactions.read_bytes()).hexdigest()
    for path, obj in [(args.output, output), (args.report, report)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+'.tmp')
        temporary.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
