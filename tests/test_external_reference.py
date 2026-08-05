"""Tests for the parameter-free external-reference correction layer.

These lock in the two properties that make the layer defensible:
  1. it never touches a reaction it is not configured for (baseline preserved);
  2. an isodesmic correction is exactly the reference reaction's residual,
     so a target that equals the reference is mapped onto the experiment.
"""
import unittest

from qm_thermo import config
from qm_thermo.reactions import SpeciesInfo
from qm_thermo import external_reference as xr


class ExternalReferenceTests(unittest.TestCase):
    def setUp(self):
        # Minimal synthetic system: two anions A(-2), B(-1) and products.
        self.C = config.DEFAULT_CONDITIONS
        self.S = {
            "A": SpeciesInfo("A", n_hydrogens=2, charge=-2),
            "B": SpeciesInfo("B", n_hydrogens=1, charge=-1),
            "P": SpeciesInfo("P", n_hydrogens=0, charge=-2),
            "Q": SpeciesInfo("Q", n_hydrogens=3, charge=-1),
        }
        self.G = {"A": -1000.0, "B": -500.0, "P": -1400.0, "Q": -140.0}
        # target and reference share A on the same side -> charge-balanced residual
        self.reactions = {"target": {"A": -1.0, "P": 1.0}}

    def test_isodesmic_shift_equals_reference_residual(self):
        spec = {
            "reaction_class": "t", "applies_to": ["target"],
            "reference_name": "ref", "reference_stoichiometry": {"A": -1.0, "P": 1.0},
            "experimental_dG_kJ": 5.0, "citation": "unit-test",
        }
        base = {"target": xr._dG(self.reactions["target"], self.G, self.S, self.C)}
        out = xr.apply_isodesmic(spec, self.reactions, self.G, self.S, self.C, baseline=base)
        # target IS the reference here -> corrected value must equal the experiment.
        self.assertAlmostEqual(out["target"].corrected_kJ, 5.0, places=6)
        self.assertEqual(out["target"].provenance["residual_reaction_net_charge"], 0)

    def test_redox_equalization_uses_counterpart_not_target(self):
        spec = {
            "reaction_class": "redox", "target": "nad", "reference_counterpart": "nadp",
            "direction_sign": 1, "couple_target": "NAD+/NADH",
            "couple_reference": "NADP+/NADPH", "E0_target_V": -0.320,
            "E0_reference_V": -0.324, "citation": "unit-test",
        }
        baseline = {"nad": -35.0, "nadp": 4.7}
        out = xr.apply_redox_equalization(spec, {"nad": {}}, baseline=baseline)
        # corrected = counterpart QM (4.7) + tiny external E0' offset, NOT the target's -35.
        offset = -2 * config.DEFAULT_CONDITIONS.F_kJ_per_V * (-0.320 - (-0.324))
        self.assertAlmostEqual(out["nad"].corrected_kJ, 4.7 + offset, places=6)
        self.assertLess(abs(offset), 1.0)

    def test_uncorrected_reactions_keep_baseline(self):
        references = {"isodesmic": [{
            "reaction_class": "t", "applies_to": ["target"], "reference_name": "ref",
            "reference_stoichiometry": {"A": -1.0, "P": 1.0},
            "experimental_dG_kJ": 5.0, "citation": "unit-test"}]}
        reactions = dict(self.reactions)
        reactions["other"] = {"B": -1.0, "Q": 1.0}
        base = {r: xr._dG(reactions[r], self.G, self.S, self.C) for r in reactions}
        values, prov = xr.apply_all(references, reactions, self.G, self.S, self.C,
                                    {}, baseline=base)
        self.assertAlmostEqual(values["other"], base["other"], places=9)
        self.assertNotIn("other", prov)


if __name__ == "__main__":
    unittest.main()
