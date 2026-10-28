"""CPU comparison of declared molecular-state ensembles using frozen QM records.

python -m metag.tools.compare_states PLAN.json -o NEW_REPORT.json

The plan declares states and either reference_state (fixed-charge state pool)
or stoichiometry (atom/charge-balanced exchange with zero net molecular count).
Each state contains a cache
{path, sha256}, records keyed by geometry hash with {path, sha256}, and an
optional geometry_hashes list for a selected-minimum control. Paths resolve
relative to the plan. Records are strict SMD records, complete wrappers, or
estimated worker outputs containing result.record. No calculations are launched.

A neutral transfer diagnostic instead declares mode="solvation_range" and states
whose entries contain cache and optional reference_kj (a central value only).
It uses every validated recorded minimum without requiring new QM records.
"""
import argparse
import hashlib
import json
from pathlib import Path

from metag.validation.state_ensembles import (
    compare_state_ensembles, compare_balanced_components, solvation_range,
)


def compare_from_plan(plan_path, destination):
    plan_path, destination = Path(plan_path), Path(destination)
    raw_plan = plan_path.read_bytes()
    plan = json.loads(raw_plan)
    if not isinstance(plan, dict) or set(plan) not in (
            {'reference_state', 'states'}, {'stoichiometry', 'states'}, {'mode', 'states'}):
        raise ValueError('Plan requires states and reference_state, stoichiometry or a supported mode')
    transfer_mode = 'mode' in plan
    if transfer_mode and plan['mode'] != 'solvation_range':
        raise ValueError('Unsupported comparison mode')
    states = plan['states']
    if not isinstance(states, dict) or not states:
        raise ValueError('A nonempty declared state inventory is required')
    hashes = {}

    def load(entry):
        if not isinstance(entry, dict) or set(entry) != {'path', 'sha256'}:
            raise ValueError('Each source requires path and sha256')
        path = (plan_path.parent / entry['path']).resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry['sha256']:
            raise ValueError('Source file changed: ' + str(path))
        hashes[str(path)] = digest
        return json.loads(data)

    caches, records, selections = {}, {}, {}
    for name, state in states.items():
        if transfer_mode:
            if not isinstance(state, dict) or set(state) not in (
                    {'cache'}, {'cache', 'reference_kj'}):
                raise ValueError('Transfer state requires cache and optional reference_kj')
            caches[name] = solvation_range(load(state['cache']), reference_kj=state.get('reference_kj'))
            continue
        if not isinstance(state, dict) or set(state) not in (
                {'cache', 'records'}, {'cache', 'records', 'geometry_hashes'}):
            raise ValueError('State requires cache, records and optional geometry_hashes')
        caches[name] = load(state['cache'])
        if not isinstance(state['records'], dict):
            raise ValueError('State records must map every required geometry to a source file')
        records[name] = {}
        for key, entry in state['records'].items():
            wrapper = load(entry)
            if wrapper.get('status') == 'estimated':
                record = (wrapper.get('result') or {}).get('record')
            elif wrapper.get('status') == 'complete':
                record = wrapper.get('record', wrapper)
            else:
                raise ValueError('A required quantum result failed or is incomplete')
            if not isinstance(record, dict):
                raise ValueError('A strict matched gas/SMD record is required')
            records[name][key] = record
        if 'geometry_hashes' in state:
            selections[name] = state['geometry_hashes']
    if selections and set(selections) != set(states):
        raise ValueError('Selections must be explicit for every state, or absent for all states')
    if transfer_mode:
        report = {'status': 'complete', 'states': caches,
                  'sampling_scope': 'All validated recorded neutral gas minima; fixed solvent terms',
                  'uncertainty': None}
    elif 'stoichiometry' in plan:
        if selections:
            raise ValueError('Balanced component exchanges require every recorded minimum; selections are unsupported')
        report = compare_balanced_components(caches, records, plan['stoichiometry'])
    else:
        report = compare_state_ensembles(caches, records, reference_state=plan['reference_state'],
                                         selections=selections or None)
    report.update(plan_sha256=hashlib.sha256(raw_plan).hexdigest(), source_files_sha256=hashes)
    # Serialize before creating a file, and never overwrite an earlier analysis.
    data = json.dumps(report, indent=2, allow_nan=False) + '\n'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        stream.write(data)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('-o', '--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare_from_plan(args.plan, args.output)
    print(json.dumps({'status': report['status'], 'sampling_scope': report['sampling_scope'],
                      'n_states': len(report['states'])}, indent=2))


if __name__ == '__main__':
    main()
