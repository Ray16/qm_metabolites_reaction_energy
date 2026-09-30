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


def test_nested_cv_refits_anchor_offsets(monkeypatch):
    """The CV must refit TECRDB-referenced anchor offsets per fold. A pool whose members disagree gets
    held-out residuals that reflect the fold-refit offset, not the in-sample one (in-sample = 0 bias)."""
    import metag.tools.calibrate as C
    monkeypatch.setattr(C, "_anchor_refs", lambda: {"toy": ("tecrdb", {f"a{i}" for i in range(10)}),
                                                    "ext": ("external", {"e0"})})
    rng = np.random.default_rng(1)
    raw_err = rng.normal(20.0, 6.0, 10)                   # anchored class: raw error ~ +20 ± 6
    off = float(raw_err.mean())                           # deployed offset = in-sample mean
    recs = [{"rid": f"a{i}", "class": "toy", "dG_raw": raw_err[i], "dG": raw_err[i] - off, "exp": 0.0,
             "anchor": {"subclass": "toy", "offset": off}} for i in range(10)]
    recs += [{"rid": f"o{i}", "class": "other", "dG": float(rng.normal(0, 8)), "exp": 0.0} for i in range(60)]
    cal = C.calibrate(recs)
    # held-out MAE of the anchored rows under fold offsets must exceed the in-sample MAE (leakage removed)
    folds = {r["rid"]: None for r in recs}
    assert cal["cv_heldout_MAE"] is not None
    ins = np.mean(np.abs(raw_err - off))
    rows = C._normalize(recs)
    import hashlib
    fold = [int(hashlib.md5(r["rid"].encode()).hexdigest(), 16) % 5 for r in rows]
    refs = C._anchor_refs()
    ho = []
    for f in range(5):
        tr = [i for i in range(len(rows)) if fold[i] != f]
        fo = C._fold_offsets(rows, tr, refs)
        ho += [abs(C._residual(rows[i], fo, refs)) for i in range(len(rows)) if fold[i] == f and rows[i]["sc"]]
    assert np.mean(ho) > ins                              # LOO-style inflation of the anchored error
    # an externally referenced anchor is never refit
    assert "ext" not in C._fold_offsets(rows, list(range(len(rows))), refs)


def test_refit_anchor_offsets_matches_pool_mean(monkeypatch):
    import metag.tools.calibrate as C
    monkeypatch.setattr(C, "_anchor_refs", lambda: {"toy": ("tecrdb", {"a0", "a1"})})
    recs = [{"rid": "a0", "class": "t", "dG": 0, "dG_raw": 12.0, "exp": 0.0, "anchor": {"subclass": "toy", "offset": 10}},
            {"rid": "a1", "class": "t", "dG": 0, "dG_raw": 8.0, "exp": 0.0, "anchor": {"subclass": "toy", "offset": 10}},
            {"rid": "x", "class": "t", "dG": 0, "dG_raw": 90.0, "exp": 0.0, "anchor": {"subclass": "toy", "offset": 10}}]
    assert C.refit_anchor_offsets(recs)["toy"]["offset"] == 10.0   # non-pool member x is ignored


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  {name} OK")
    print("PASS: calibration logic")
