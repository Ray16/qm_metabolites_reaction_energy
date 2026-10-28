"""Read-only openTECR condition audit; never transforms or rewrites labels."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from metag.validation.reference_conditions import audit_conditions, table_key


def audit_snapshot(source):
    source = Path(source)
    paths = {name: source / ('sheet_' + name + '.csv') for name in
             ('actual_data', 'table_metadata', 'table_comments')}
    def read(path):
        with path.open(newline='') as handle:
            return list(csv.DictReader(handle))
    metadata = {table_key(row): row for row in read(paths['table_metadata'])}
    comments = {table_key(row): row for row in read(paths['table_comments'])}
    records = [audit_conditions(row, metadata.get(table_key(row)), comments.get(table_key(row)))
               for row in read(paths['actual_data'])]
    counts = Counter(issue for record in records for issue in record['issues'])
    return {'scope': 'All source measurements, not only benchmark-matched/usable rows. No labels changed.',
            'source_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
            'summary': {'measurements': len(records), 'issues': dict(counts),
                        'excluded_unknown_free_magnesium': sum(r['exclusion'] is not None for r in records)},
            'measurements': records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_snapshot(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
