"""Explicit search-effort presets; neither accuracy nor coverage is guaranteed."""
import os
import sys

THOROUGH_SETTINGS = {"CONV_TOL": "0.5", "CONV_HITS": "3", "CONV_MAX": "12",
                     "SAMPLE_SCALE": "2", "KEEP_SCALE": "2"}
PROFILE_VERSION = "2026-10-11-search-effort-v1"


def sampling_environment(profile, existing):
    """Return a new environment; preserve user settings for the current preset."""
    if profile not in ("current", "thorough"):
        raise ValueError("sampling profile must be current or thorough")
    environment = dict(existing)
    if profile == "thorough":
        environment.update(THOROUGH_SETTINGS)
    return environment


def apply_sampling_profile(profile):
    """Apply before pipeline import, whose sampling constants are read once.

    The thorough preset changes only search effort. All QM, chemical identity,
    force, Hessian and state-coverage checks remain in force. Independent seeds
    still need their own repeatability audit; a plateau is not completeness.
    """
    environment = sampling_environment(profile, os.environ)
    if profile == "thorough":
        if "metag.pipeline" in sys.modules:
            raise RuntimeError("Apply the sampling profile before importing metag.pipeline; its sampling constants are fixed at import")
        for name in THOROUGH_SETTINGS:
            os.environ[name] = environment[name]
    return {"name": profile, "version": PROFILE_VERSION,
            "settings": {name: environment.get(name) for name in THOROUGH_SETTINGS},
            "scope": "Search effort only; no validated accuracy improvement, complete-state or uncertainty-coverage claim"}
