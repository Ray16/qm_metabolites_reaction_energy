"""Opt-in real-backend reaction checks on V100 or a newer CUDA GPU.

Run in the pinned QM environment through a GPU reservation, with
METAG_RUN_GPU_TESTS=1. This file also runs directly without installing pytest.
The unresolved isocitrate fixture must be withheld by the current stereo policy;
the three declared reactions exercise routing, sampling, solvation and RRHO.
Experimental labels do not enter these hardware checks.
"""
import os
import json
import hashlib
import math
from pathlib import Path
import time

try:
    import pytest
except ModuleNotFoundError:
    if __name__ != "__main__":
        raise
    pytest = None


if pytest is not None:
    pytestmark = [pytest.mark.integration, pytest.mark.gpu]
_REACTIONS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "smoke_reactions.json")


def _gpu_available():
    if os.environ.get("METAG_RUN_GPU_TESTS", "").lower() not in ("1", "true", "yes"):
        return False
    # An explicitly requested hardware test must expose broken dependencies or
    # CUDA setup instead of quietly counting an unavailable backend as a skip.
    import torch
    import fairchem.core  # noqa: F401
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("reserve exactly one usable CUDA GPU for this check")
    return True


def test_score_several_reactions():
    if not _gpu_available():
        if pytest is not None:
            pytest.skip("set METAG_RUN_GPU_TESTS=1 with a reserved GPU and UMA runtime")
        raise RuntimeError("set METAG_RUN_GPU_TESTS=1 inside a GPU reservation")
    import torch
    from metag.energetics.uma import load_uma
    from metag.pipeline import score_reaction, _MODEL
    pu = load_uma(_MODEL)
    with open(_REACTIONS) as stream:
        reactions = json.load(stream)
    outcomes = {}
    for rid, rx in reactions.items():
        rx = {k: v for k, v in rx.items() if k not in {"exp", "note"}}
        began = time.monotonic()
        print("SMOKE_REACTION_START", rid, flush=True)
        r = score_reaction(pu, rx, log=lambda line: print(line, flush=True), key=rid)
        if rid == "rxn00973":
            assert r["dG"] is None
            assert "unresolved stereochemical identity" in r["suspect"]
        else:
            assert r["suspect"] is None, f"{rid}: suspect {r['suspect']}"
            assert r["dG"] is not None and math.isfinite(r["dG"]), (rid, r["dG"])
            assert r["dG_raw"] is not None and math.isfinite(r["dG_raw"])
            assert r["config"]["hessian"] == "autograd"
            assert r["config"]["producer"]["model"] == pu.metag_provenance
            assert r["U_samp"] is not None and math.isfinite(r["U_samp"]) and r["U_samp"] >= 0
            # The current estimator has no validated predictive intervals.
            assert not r["ci_info"]["coverage_calibrated"]
            assert r["sigma_pred"] is None and list(r["ci95"]) == [None, None]
        outcomes[rid] = {"elapsed_seconds": time.monotonic() - began, "outcome": r}
        print("SMOKE_REACTION_VERIFIED", rid, "dG", r["dG"], flush=True)
    assert set(outcomes) == set(reactions)
    report = {
        "host": os.uname().nodename,
        "pid": os.getpid(),
        "start_ticks": Path("/proc/self/stat").read_text().split(") ")[1].split()[19],
        "GPU": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "CPUs": sorted(os.sched_getaffinity(0)),
        "checkpoint": pu.metag_provenance,
        "inputs_sha256": hashlib.sha256(Path(_REACTIONS).read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_estimated": sum(row["outcome"]["dG"] is not None for row in outcomes.values()),
        "n_expected_stereo_refusals": 1,
        "outcomes": outcomes,
        "scope": "Real-backend execution and result-contract checks; no experimental accuracy claim.",
    }
    if os.environ.get("METAG_SMOKE_REPORT"):
        with open(os.environ["METAG_SMOKE_REPORT"], "x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print("OK: three reaction estimates and one expected stereo refusal", flush=True)


if __name__ == "__main__":
    test_score_several_reactions()
