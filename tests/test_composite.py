import math
import unittest

from qm_thermo.composite import ConformerTerms, boltzmann_ensemble, extract_ensemble_energy
from qm_thermo.speciation import monoprotic_base_fraction, monoprotic_family_correction_kJ


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


if __name__ == "__main__":
    unittest.main()
