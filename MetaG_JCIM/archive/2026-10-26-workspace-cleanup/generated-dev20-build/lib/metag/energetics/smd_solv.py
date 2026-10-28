"""Strict component SMD API; legacy reaction correction remains quarantined.

Uses converged matched-coordinate calculations and full producer cache identity.
This quantity is a potential difference, not a validated reaction free energy.
"""
from metag.energetics.strict_smd import (
    HARTREE_KJ, REDOX_PY, _key, _compute_smd, smd_dgsolv, main,
)

if __name__ == "__main__":
    main()
