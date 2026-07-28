import math
import unittest

from experiments.explicit_water.grand_canonical_clusters import (
    R_KJ, TEMPERATURE, grand_free_energy, water_ladder_limit, write_xyz,
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


if __name__ == "__main__":
    unittest.main()
