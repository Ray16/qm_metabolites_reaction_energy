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
    assert pk._neutralize("O=C([O-])/C=C/C(=O)[O-]")[1] == [3.75, 3.75]      # fumarate
    assert pk._neutralize("C=CC(=O)[O-]")[1] == [4.35]                         # acrylate
    assert sorted(pk._neutralize("O=C([O-])C[C@H](O)C(=O)[O-]")[1]) == [3.8, 4.75]   # malate unchanged
    # the fumarate transform now matches its measured macroscopic pKa's (3.03, 4.44) to < 0.2 kJ
    RT_LN10 = 2.303 * 8.314e-3 * 298.15
    t = lambda ps: sum(RT_LN10 * math.log10(1 + 10 ** (7 - p)) for p in ps)
    assert abs(t([3.75, 3.75]) - t([3.03, 4.44])) < 0.2


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


def test_multi_site_mixture_reduces_to_single_site():
    for dg in (-12.0, -1.0, 0.0, 3.0, 20.0):
        assert abs(ah.mixture_G_sites(0.0, [dg], 0.0) - ah.mixture_G(0.0, dg, 0.0)) < 1e-9
    # two independent sites lower G by the sum of the single-site terms
    a = ah.mixture_G_sites(0.0, [-3.0], 0.0); b = ah.mixture_G_sites(0.0, [2.0], 0.0)
    assert abs(ah.mixture_G_sites(0.0, [-3.0, 2.0], 0.0) - (a + b)) < 1e-9


def test_hydration_calibration_reproduces_khyd_fit():
    # log K_exp = 0.67 log K_calc - 0.34  <=>  ΔG_exp = 0.67 ΔG_calc + 0.34 RT ln10
    rt_ln10 = 2.303 * 8.314e-3 * 298.15
    assert abs(ah.calibrated_dg_hyd(0.0) - 0.34 * rt_ln10) < 0.02
    assert abs(ah.calibrated_dg_hyd(-10.0) - (-6.7 + 0.34 * rt_ln10)) < 0.02


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
