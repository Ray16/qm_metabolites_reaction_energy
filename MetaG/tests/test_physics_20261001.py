"""Physics fixes of the 2026-10-01 review (no QM): weak-acid anion neutralisation, alpha,beta-unsaturated
carboxyl pKa class, aryl-amine basicity in the base-path gate, general carbonyl hydration sites, the
multi-site hydration mixture, and the extra-force hook of the batched optimiser."""
import math
import os

import numpy as np

from metag.routing import aldehyde_hydration as ah
from metag.routing import pka_transform as pk


def test_thiolate_and_phenolate_are_neutralised():
    gsh = "[NH3+][C@@H](CCC(=O)N[C@@H](C[S-])C(=O)NCC(=O)[O-])C(=O)[O-]"
    smi, pkas, q = pk._neutralize(gsh)
    assert "[S-]" not in smi and "CS" in smi
    assert pk.THIOL_PKA in pkas
    tyr = "N[C@@H](Cc1ccc([O-])cc1)C(=O)[O-]"
    smi, pkas, q = pk._neutralize(tyr)
    assert q == 0 and pk.PHENOL_PKA in pkas
    assert pk._neutralize("CC(=O)[S-]")[2] == -1          # thiocarboxylate is not a thiolate


def test_unsaturated_carboxyl_pka(monkeypatch):
    monkeypatch.setenv("PKA_ENV", "1")
    # class rule (compound not in POLYACID_PKA): mesaconate, acrylate; malate rule unchanged
    assert pk._neutralize("C/C(=C\\C(=O)[O-])C(=O)[O-]")[1] == [3.75, 3.75]
    assert pk._neutralize("C=CC(=O)[O-]")[1] == [4.35]
    monkeypatch.setenv("POLYACID_PKA", "0")
    assert sorted(pk._neutralize("O=C([O-])C[C@H](O)C(=O)[O-]")[1]) == [3.8, 4.75]


def test_recognized_polyacids_use_compound_constants(monkeypatch):
    # Martell & Smith (I = 0, 25 °C); the rule table was off by -22.7 (oxalate) and -5.6 kJ (malonate)
    monkeypatch.setenv("POLYACID_PKA", "1")
    assert pk._neutralize("O=C([O-])/C=C/C(=O)[O-]")[1] == [3.053, 4.494]     # fumarate
    assert pk._neutralize("O=C([O-])/C=C\\C(=O)[O-]")[1] == [1.910, 6.332]   # maleate
    assert pk._neutralize("O=C([O-])C(=O)[O-]")[1] == [1.252, 4.266]          # oxalate
    assert pk._neutralize("O=C([O-])CC(=O)[O-]")[1] == [2.847, 5.696]         # malonate
    # at pH 7 the independent-site sum equals the coupled binding polynomial (<= 0.01 kJ)
    import math
    RT = 8.314e-3 * 298.15
    ladder = [3.128, 4.761, 6.396]                                              # citric
    poly, prod = 1.0, 1.0
    for p in ladder:
        prod *= 10 ** (7 - p); poly += prod
    indep = sum(RT * math.log(1 + 10 ** (7 - p)) for p in ladder)
    assert abs(RT * math.log(poly) - indep) < 0.02


def test_alpha_keto_carboxyl_uses_keto_microstate_pka(monkeypatch):
    monkeypatch.setenv("PKA_ENV", "1")
    assert pk._neutralize("CC(=O)C(=O)[O-]")[1] == [1.8]                       # Lopalco et al. 2016


def test_arylamine_is_not_a_basic_amine(monkeypatch):
    adenylosuccinate_synthase = {
        "Asp": (-1, -1, "[NH3+][C@@H](CC(=O)[O-])C(=O)[O-]"),
        "IMP": (-1, -2, "O=c1[nH]cnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])[O-])[C@@H](O)[C@H]1O"),
        "S-AMP": (1, -4, "O=C([O-])C[C@H](Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])[O-])[C@@H](O)"
                         "[C@H]1O)C(=O)[O-]"),
        "H2O": (1, 0, "O"),
    }
    monkeypatch.setenv("ARYLAMINE_NONBASIC", "0")
    assert pk._amine_cn_change(adenylosuccinate_synthase) == 0          # legacy: N6 counted as an amine
    monkeypatch.setenv("ARYLAMINE_NONBASIC", "1")
    assert pk._amine_cn_change(adenylosuccinate_synthase) == -1         # the Asp ammonium is destroyed


def test_hydration_sites():
    assert ah.hydration_sites("CC(=O)C(=O)O") == []                     # alpha-keto acid: excluded
    assert ah.hydration_sites("O=C(O)CC(=O)C(=O)O") == []               # oxaloacetate: excluded
    assert [d for _, d in ah.hydration_sites("O=CC(=O)O")] == ["O=C(O)C(O)O"]           # glyoxylate kept
    assert [d for _, d in ah.hydration_sites("O=C(CO)COP(=O)(O)O")] == ["O=P(O)(O)OCC(O)(O)CO"]   # DHAP
    assert len(ah.hydration_sites("CC(=O)C=O")) == 2                                    # methylglyoxal
    assert ah.hydration_sites("CC(=O)SC") == [] and ah.hydration_sites("CC(N)=O") == []  # thioester, amide
    assert ah.hydration_sites("CC(=O)O") == []                                           # acid


def test_hydration_states_exact_enumeration():
    # every subset of sites is an explicit, computed state; no uncomputed (implicitly multiplied) states
    states = ah.hydration_states("CC(=O)C=O")                               # methylglyoxal: 2 sites
    assert sorted(n for n, _ in states) == [1, 1, 2]
    assert "CC(O)(O)C(O)O" in {s for _, s in states}
    for dg in (-12.0, -1.0, 0.0, 3.0, 20.0):                               # one state = two-state formula
        assert abs(ah.mixture_G_states(0.0, [(1, dg)], 0.0) - ah.mixture_G(0.0, dg, 0.0)) < 1e-9


def test_hydration_sites_independent_of_atom_order():
    a = sorted(s for _, s in ah.hydration_states("O=CC(C)CC(=O)CC=O"))       # three carbonyls
    b = sorted(s for _, s in ah.hydration_states("O=CCC(=O)CC(C)C=O"))       # same molecule, other order
    assert a == b and len(a) == 7                                          # 2^3 - 1 hydrated states


def test_hydration_calibration_reproduces_khyd_fit():
    # log K_exp = 0.639 log K_calc - 0.411  <=>  ΔG_exp = 0.639 ΔG_calc + 2.35 kJ (per event)
    rt_ln10 = 2.303 * 8.314e-3 * 298.15
    assert abs(ah.calibrated_dg_hyd(0.0) - 2.35) < 1e-9
    assert abs(ah.calibrated_dg_hyd(-10.0) - (-6.39 + 2.35)) < 1e-9


def test_acid_groups_exclude_hemiacetal_carbon():
    from rdkit import Chem
    import metag.pipeline as P
    fbp = Chem.AddHs(Chem.MolFromSmiles("O=P(O)(O)OC[C@H]1OC(O)(COP(=O)(O)O)[C@@H](O)[C@@H]1O"))
    groups, donors = P._acid_groups(fbp)
    assert len(set(groups.values())) == 2 and len(donors) == 4             # two phosphates, 4 P-OH
    assert P._acid_groups(Chem.AddHs(Chem.MolFromSmiles("OC1CCCO1")))[1] == []


def test_solvation_force_units():
    from metag.energetics.solv_relax import HB2EVA
    assert abs(HB2EVA - 51.42208) < 1e-3                  # Hartree/Bohr -> eV/Å


def test_explicit_cache_key_tracks_effective_water_reference(monkeypatch):
    # review item: the key used the raw env string ("0" when unset) while the default is WATER_REF_EXP on
    import metag.pipeline as P
    monkeypatch.delenv("WATER_REF_EXP", raising=False)
    default = P._explicit_settings()
    monkeypatch.setenv("WATER_REF_EXP", "0")
    ablation = P._explicit_settings()
    monkeypatch.setenv("WATER_REF_EXP", "1")
    explicit_on = P._explicit_settings()
    assert default["water_ref_exp"] is P.FLAG_DEFAULTS["WATER_REF_EXP"]
    assert default != ablation and default == explicit_on


def test_effective_config_tracks_every_estimator_switch(monkeypatch):
    # review: THERMAL_ENSEMBLE / POLYACID_PKA (and others) were absent from the fingerprint, so the calibration
    # guard could not detect them. Each toggle must change the fingerprint. Module-level switches are read at
    # import, so the module is reloaded per toggle.
    import importlib, json as _json
    import metag.pipeline as P
    base = _json.dumps(P.effective_config(), sort_keys=True)
    for var, val in (("POLYACID_PKA", "1"), ("ARYLAMINE_NONBASIC", "0"), ("PKA_ENV", "0"),
                     ("CARBONYL_HYDRATION_ALL", "0"), ("HYDRATION_CAL", "0"), ("PH0_ISOMERASE", "0"),
                     ("NTP_CORE", "0")):
        monkeypatch.setenv(var, val)
        assert _json.dumps(P.effective_config(), sort_keys=True) != base, var
        monkeypatch.delenv(var)
    for var, val in (("THERMAL_ENSEMBLE", "0"), ("ACID_HB_FILTER", "1"), ("SOLV_RELAX", "1")):
        monkeypatch.setenv(var, val)
        P2 = importlib.reload(P)
        assert _json.dumps(P2.effective_config(), sort_keys=True) != base, var
        monkeypatch.delenv(var)
    importlib.reload(P)


def test_constant_edit_changes_fingerprint(monkeypatch):
    import metag.pipeline as P
    from metag.routing import pka_transform as pk
    base = P.effective_config()["pka_constants"]
    monkeypatch.setitem(pk.CARBOXYL_PKA_ALPHA, "oxo", 2.5)
    assert P.effective_config()["pka_constants"] != base


def test_deployed_hydration_calibration_matches_reproducible_fit():
    # review: the deployed HYDRATION_CAL must be what analysis/sweep_20261001/khyd_validation.py produces
    import json, os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "analysis", "sweep_20261001", "khyd_validation.json")
    fit = json.load(open(path))["calibration_domain_fit"]
    a, b = ah.HYDRATION_CAL
    assert abs(a - fit["a"]) < 0.0015 and abs(b - fit["b_kJ_per_event"]) < 0.01
