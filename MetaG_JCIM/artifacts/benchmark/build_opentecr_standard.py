"""Compatibility entry point for the portable, provenance-preserving reference builder.

Use --source, --modelseed, --reactions, --output and --report; no external legacy
builder or user-specific paths are required. See docs/scientific-policy-audit.md.
"""
from metag.tools.build_reference import main, measurement_exclusion


def usable(row, metadata=None):
    return measurement_exclusion(row, row.get('_reference_metadata', {}) if metadata is None else metadata)


if __name__ == '__main__':
    main()
