"""CPU preflight for reaction-panel representation consistency, before QM.

METAG_ROUTING_MODE=full_molecule python -m metag.tools.audit_routes INPUT -o REPORT
"""
import argparse
import hashlib
import json
from pathlib import Path

from metag.tools.preflight import audit_reactions
from metag.validation.routing import audit_routing_consistency


def audit_route_panel(reactions):
    report = audit_reactions(reactions)
    consistency = audit_routing_consistency(reactions, report['reactions'])
    report['representation_consistency'] = consistency
    from metag.validation.compound_references import audit_compound_references
    references = audit_compound_references(reactions, report['reactions'])
    report['compound_reference_consistency'] = references
    from metag.validation.protonation import audit_model_consistency
    protonation = audit_model_consistency(report['reactions'])
    report['protonation_model_consistency'] = protonation
    require_owned = report['config']['routing']['mode'] == 'compound_reference'
    blocked = {rid:row['status'] for rid,row in report['reactions'].items()
               if row['status'] not in ('no_known_gap_detected', 'review')}
    report['panel_gate'] = {
        'passed': bool(reactions) and consistency['consistent_known_representations'] and not blocked
                  and protonation['passed']
                  and (references['passed'] if require_owned else True),
        'blocked_reactions': blocked,
        'scope': 'Structural and known-representation gate only; not energy cycle closure, validated chemical pools, experimental accuracy or calibrated uncertainty.'}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--require-consistent', action='store_true',
                        help='Write the report, then exit 1 if structural/representation gate fails')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output exists; choose a new evidence path')
    raw = args.input.read_bytes()
    reactions = json.loads(raw)
    if not isinstance(reactions, dict):
        parser.error('Input must be a reaction mapping')
    report = audit_route_panel(reactions)
    report.update(input_file=str(args.input.resolve()), input_sha256=hashlib.sha256(raw).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'n_reactions':report['n_reactions'], 'status_counts':report['status_counts'],
                      'n_context_dependent_species':report['representation_consistency']['n_context_dependent_species'],
                      'panel_gate_passed':report['panel_gate']['passed'],
                      'n_blocked_reactions':len(report['panel_gate']['blocked_reactions'])}, indent=2))
    if args.require_consistent and not report['panel_gate']['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
