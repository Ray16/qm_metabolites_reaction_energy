import math
import unittest

from experiments.explicit_water.grand_canonical_clusters import (
    R_KJ, TEMPERATURE, grand_free_energy, pka_statistics, water_ladder_limit, write_xyz,
)


class GrandCanonicalClusterTests(unittest.TestCase):
    def test_one_cluster_is_its_water_adjusted_free_energy(self):
        record = {"counts": {"2": {"minima": [{"G_kJ": 18.0}]}}}
        energy, occupancy = grand_free_energy(record, 5.0)
        self.assertAlmostEqual(energy, 8.0)
        self.assertEqual(occupancy, {"2": 1.0})

    def test_degenerate_minima_produce_the_correct_entropy(self):
        record = {"counts": {"0": {"minima": [{"G_kJ": 10.0}, {"G_kJ": 10.0}]}}}
        energy, _ = grand_free_energy(record, 0.0)
        self.assertAlmostEqual(energy, 10.0 - R_KJ * TEMPERATURE * math.log(2.0))

    def test_cluster_standard_state_is_added_once_per_cluster(self):
        record = {"counts": {"1": {"minima": [{"G_kJ": 10.0}]}}}
        energy, _ = grand_free_energy(record, 5.0, cluster_standard_state_kj=2.0)
        self.assertAlmostEqual(energy, 7.0)

    def test_ladder_scales_with_formal_anionic_charge(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            xyz = os.path.join(directory, "anion.xyz")
            write_xyz(["P", "O", "O", "O"],
                      __import__("numpy").array([[0., 0., 0.], [1.5, 0., 0.],
                                                   [-0.75, 1.3, 0.], [-0.75, -1.3, 0.]]), xyz)
            self.assertEqual(water_ladder_limit(xyz, -3, 2, None), 6)

    def test_pka_statistics_preserves_group_and_charge_strata(self):
        rtln10 = R_KJ * TEMPERATURE * math.log(10.0)
        pairs = [
            {"key": "control", "acid": "BH", "base": "B", "q_acid": 1, "q_base": 0,
             "kind": "cationic", "group": "ammonium", "pKa_exp": 7.0},
            {"key": "phos", "acid": "AH", "base": "A", "q_acid": -1, "q_base": -2,
             "kind": "anionic", "group": "phosphate", "pKa_exp": 6.0},
        ]
        energies = {"BH": 0.0, "B": 7.0 * rtln10 - (-1122.8), "AH": 0.0,
                    "A": 6.0 * rtln10 - (-1122.8)}
        stats = pka_statistics(pairs, energies)
        self.assertEqual(stats["by_group"]["phosphate"]["n"], 1)
        self.assertEqual(stats["by_resulting_anion_charge"]["2"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
