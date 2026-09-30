"""CPU tests of metag.pipeline.score_reaction bookkeeping with the QM layer mocked (no GPU):
  * 1 atm -> 1 M standard state: every species (solutes AND liquid water) gets +RT ln 24.46, so a
    reaction's ΔG shifts by exactly STD_STATE_KJ·Δn and Δn = 0 reactions are untouched;
  * the truncated-core failure path re-scores the ORIGINAL dict with full molecules (was KeyError);
  * ci95 uses the same σ_total as sigma_pred (U_samp included);
  * route provenance is returned.
Requires the `uma` env only for imports (ase); no model is loaded."""
import math
import os
import pytest

P = pytest.importorskip("metag.pipeline")

G_FAKE = {"CCO": -100.0, "CC=O": -50.0, "[HH]": -5.0, "CC(=O)O": -300.0, "CCCC": -400.0, "CC": -190.0,
          "CO": -80.0, "C": -30.0}
G_WATER = -200.0


@pytest.fixture(autouse=True)
def _mock_qm(monkeypatch):
    monkeypatch.setattr(P, "implicit_G", lambda pu, q, smi, *a, **k: (G_FAKE[smi], 0.5))
    monkeypatch.setattr(P, "water_ref_G", lambda pu, log=None: G_WATER)
    for f in ("COFACTOR_RING", "AUTO_TRUNCATE", "PH0_AUTO", "ANCHOR_CORRECT", "ALDEHYDE_HYDRATION",
              "WATER_REF_HYDROLYASE", "TRUNC_VALIDATE", "ZWITTERION_PH0", "NEUTRAL_QM"):
        monkeypatch.setenv(f, "0")
    monkeypatch.setenv("STD_STATE_1M", "1")       # exercise the std-state bookkeeping (default is off)


def _rx(species, n_h=0):
    return {"note": "test", "n_Hplus": n_h, "species": species}


def _score(rx):
    return P.score_reaction(None, rx, log=lambda *_: None)


def test_std_state_constant():
    assert abs(P.STD_STATE_KJ - 7.93) < 0.01


def test_std_state_shifts_by_delta_n(monkeypatch):
    # butane -> ethane + ethane-ish (Δn = +1): dehydrogenation stand-in, 1 reactant -> 2 products
    lyase = _rx({"A": [-1, 0, "CCCC"], "B": [1, 0, "CC"], "C": [1, 0, "CC"]})
    iso = _rx({"A": [-1, 0, "CCO"], "B": [1, 0, "CC=O"], "H2": [1, 0, "[HH]"], "W": [-1, 0, "O"]})  # Δn = 0
    on_l, on_i = _score(lyase)["dG_raw"], _score(iso)["dG_raw"]
    monkeypatch.setenv("STD_STATE_1M", "0")
    off_l, off_i = _score(lyase)["dG_raw"], _score(iso)["dG_raw"]
    assert abs((on_l - off_l) - P.STD_STATE_KJ) < 0.06      # Δn = +1
    assert abs(on_i - off_i) < 1e-9                          # Δn = 0 (water counted) -> unchanged
    assert abs(off_l - (-190 - 190 + 400)) < 0.06


def test_truncation_failure_retries_full_molecules(monkeypatch):
    import metag.routing.truncate as T
    monkeypatch.setenv("AUTO_TRUNCATE", "1"); monkeypatch.setenv("ROUTE_FULL", "0")
    monkeypatch.setattr(T, "build_truncated_reaction", lambda sp, radius=2: ({"X_t": [-1, 0, "BAD"],
                                                                              "Y_t": [1, 0, "BAD"]}, 0))
    fail_on_bad = lambda pu, q, smi, *a, **k: (None, None) if smi == "BAD" else (G_FAKE[smi], 0.5)
    monkeypatch.setattr(P, "implicit_G", fail_on_bad)
    r = _score(_rx({"A": [-1, 0, "CCO"], "B": [1, 0, "CO"], "C": [1, 0, "C"]}))
    assert r is not None                                     # was: KeyError('rxn') via run_reaction
    assert r["routes"]["truncated"] is False
    assert abs(r["dG_raw"] - (-80 - 30 + 100 + P.STD_STATE_KJ)) < 0.06


def test_ci95_consistent_with_sigma_pred(monkeypatch):
    rx = _rx({"A": [-1, 0, "CCO"], "B": [1, 0, "CC=O"], "H2": [1, 0, "[HH]"]})
    small = _score(rx)
    monkeypatch.setattr(P, "implicit_G", lambda pu, q, smi, *a, **k: (G_FAKE[smi], 30.0))
    big = _score(rx)
    assert big["sigma_pred"] > small["sigma_pred"] + 20
    hw_small = big_hw = None
    hw_small = small["ci95"][1] - small["dG"]; big_hw = big["ci95"][1] - big["dG"]
    assert big_hw > hw_small + 20                            # interval widens with U_samp too
    assert big_hw >= big["ci_info"]["sigma_mult"] * big["sigma_pred"] - 0.2


def test_routes_and_scope_reported():
    r = _score(_rx({"A": [-1, 0, "CCO"], "B": [1, 0, "CC=O"], "H2": [1, 0, "[HH]"]}))
    assert set(r["routes"]) >= {"cofactor_ring", "truncated", "ph0", "errors"}
    assert "externally_calibrated" in r["ci_info"] and "calibration_scope" in r["ci_info"]
    assert r["std_state_kJ"] == round(P.STD_STATE_KJ, 2)     # fixture enables it


def test_trunc_validate_rejects_radius_sensitive(monkeypatch):
    import metag.routing.truncate as T
    monkeypatch.setenv("AUTO_TRUNCATE", "1"); monkeypatch.setenv("ROUTE_FULL", "0")
    monkeypatch.setenv("TRUNC_VALIDATE", "1")
    cores = {2: ({"A_t": [-1, 0, "CO"], "B_t": [1, 0, "C"], "C_t": [1, 0, "CC"]}, 0),   # dG ~ -140
             3: ({"A_t": [-1, 0, "CCO"], "B_t": [1, 0, "CC=O"], "C_t": [1, 0, "[HH]"]}, 0)}  # dG ~ +45
    monkeypatch.setattr(T, "build_truncated_reaction", lambda sp, radius=2: cores[radius])
    full = {"A": [-1, 0, "CCCC"], "B": [1, 0, "CC"], "C": [1, 0, "CC"]}
    r = _score(_rx(full))
    assert r["trunc_validation"]["verdict"].startswith("rejected")
    assert r["routes"]["truncated"] is False                 # fell back to full molecules
    cores[3] = cores[2]                                      # radius-invariant -> accepted
    r2 = _score(_rx(full))
    assert r2["trunc_validation"]["verdict"] == "accepted" and r2["routes"]["truncated"]
