import math
import unittest

from qm_thermo.composite import ConformerTerms, boltzmann_ensemble, extract_ensemble_energy
from qm_thermo.speciation import (
    ProtonationFamily, monoprotic_base_fraction, monoprotic_family_correction_kJ,
)
from qm_thermo.reaction_correction import CalibrationRow, leave_signature_out


class CompositeTests(unittest.TestCase):
    def test_energy_terms_are_summed_before_averaging(self):
        ensemble = boltzmann_ensemble(
            [ConformerTerms(10.0, -4.0, 1.0), ConformerTerms(15.0, -4.0, 1.0)],
            temperature_K=298.15,
        )
        self.assertAlmostEqual(ensemble.gibbs_kJ, 7.0 - 8.314462618e-3 * 298.15 *
                               math.log(1.0 + math.exp(-5.0 / (8.314462618e-3 * 298.15))))
        self.assertAlmostEqual(sum(ensemble.weights), 1.0)

    def test_mace_and_uma_breakdown_keys_are_supported(self):
        record = {"conformers": [
            {"E_elec_kJ": 1.0, "dGsolv_kJ": 2.0, "G_RRHO_kJ": 3.0},
            {"E_UMA_kJ": 7.0, "dGsolv_kJ": 2.0, "G_RRHO_kJ": 3.0},
        ]}
        self.assertLess(extract_ensemble_energy(record, temperature_K=298.15).gibbs_kJ, 6.0)

    def test_monoprotic_family_has_equal_populations_at_pka(self):
        self.assertAlmostEqual(monoprotic_base_fraction(7.0, 7.0), 0.5)
        rt = 8.314462618e-3 * 298.15
        self.assertAlmostEqual(monoprotic_family_correction_kJ(7.0, 7.0), -rt * math.log(2))

    def test_polyprotic_partition_is_anchored_to_reference_state(self):
        family = ProtonationFamily("A", (4.0, 8.0), 1, "test", "test://citation")
        self.assertAlmostEqual(sum(family.fractions(7.0)), 1.0)
        # At a pH where the reference (one deprotonation) is dominant, the
        # partition correction is small and finite rather than an N_H shift.
        self.assertLess(abs(family.correction_from_reference_kJ(7.0)), 0.5)

    def test_class_calibration_holds_equivalent_signature_out(self):
        rows = [
            CalibrationRow("fwd", "same", "phosphate_transfer", 0.0, 10.0),
            CalibrationRow("rev", "same", "phosphate_transfer", 0.0, -10.0),
            CalibrationRow("other", "other", "phosphate_transfer", 0.0, 100.0),
        ]
        scored = leave_signature_out(rows, min_signatures=2, shrinkage=0.0)
        # fwd/rev are excluded together, so their own residual cannot leak in.
        self.assertEqual(scored[0]["training_signatures_in_class"], 1)
        self.assertFalse(scored[0]["calibrated"])


if __name__ == "__main__":
    unittest.main()
