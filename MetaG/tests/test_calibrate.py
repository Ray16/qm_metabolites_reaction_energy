"""Calibration logic (metag.tools.calibrate.calibrate_classes) on synthetic residuals: the nested-CV
interval must reach ~95% held-out coverage, small-n classes must not be overconfident, and a heavy-tailed
class must get a q95abs floor above m*sigma."""
import numpy as np
from metag.tools.calibrate import calibrate_classes


def _make_rows():
    rng = np.random.default_rng(0)
    rows = []
    # a well-sampled biased class, a small-n class, and a heavy-tailed class
    for i in range(80):
        rows.append((f"big{i}", "biased", float(rng.normal(12, 8))))       # bias +12
    for i in range(3):
        rows.append((f"sm{i}", "small", float(rng.normal(0, 6))))           # n=3
    for i in range(30):
        e = rng.normal(0, 6) if rng.random() > 0.15 else rng.normal(0, 40)  # heavy tail
        rows.append((f"ht{i}", "heavy", float(e)))
    return rows


def test_calibration_structure_and_coverage():
    calib = calibrate_classes(_make_rows())
    assert set(calib["classes"]) == {"biased", "small", "heavy"}
    assert 1.8 <= calib["interval_sigma_mult"] <= 3.0
    assert calib["cv_coverage_interval95"] >= 0.90        # nested held-out, should be near 0.95


def test_small_n_not_overconfident():
    calib = calibrate_classes(_make_rows())
    # small-n class q95abs pooled toward global -> not a spuriously tiny width
    assert calib["classes"]["small"]["q95abs"] > 10.0


def test_heavy_tail_floor_exceeds_msigma():
    calib = calibrate_classes(_make_rows())
    c = calib["classes"]["heavy"]
    m = calib["interval_sigma_mult"]
    assert c["q95abs"] > m * c["sigma"]                   # floor active for the heavy-tailed class


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  {name} OK")
    print("PASS: calibration logic")
