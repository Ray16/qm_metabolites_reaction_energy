"""Read one calibration snapshot and distinguish identity from a missing match."""
import hashlib
import json
from pathlib import Path


def read_snapshot(path):
    """Return parsed data and a hash of the same bytes, even during atomic updates."""
    data = Path(path).read_bytes()
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError('Calibration artifact must be a JSON object')
    return value, hashlib.sha256(data).hexdigest()


def configuration_issues(config):
    """An absent source/condition identity cannot establish calibration applicability."""
    if not isinstance(config, dict):
        return ['calibration requires an estimator configuration object']
    issues = []
    for name in ('model', 'physics'):
        if not isinstance(config.get(name), str) or not config[name].strip():
            issues.append('configuration lacks ' + name)
    digest = config.get('estimator_source_sha256')
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in '0123456789abcdef' for char in digest)):
        issues.append('configuration lacks an exact estimator source SHA256')
    conditions = config.get('conditions')
    if isinstance(conditions, str):
        try:
            conditions = json.loads(conditions)
        except ValueError:
            conditions = None
    if not isinstance(conditions, dict) or not conditions:
        issues.append('configuration lacks target conditions')
    return issues
