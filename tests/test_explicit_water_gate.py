import unittest

from experiments.explicit_water.score_macepolar_pka import pka_statistics


class ExplicitWaterGateTests(unittest.TestCase):
    def test_cation_referencing_removes_common_offset(self):
        pairs = [
            {"key": "cat", "acid": "BH", "base": "B", "q_acid": 1, "q_base": 0,
             "kind": "cationic", "group": "amine", "pKa_exp": 7.0},
            {"key": "anion", "acid": "AH", "base": "A", "q_acid": 0, "q_base": -1,
             "kind": "anionic", "group": "carboxyl", "pKa_exp": 5.0},
        ]
        # Both pairs share a 10-kJ/mol offset, which the cation control removes.
        rtln10 = 8.314462618e-3 * 298.15 * __import__("math").log(10.0)
        mu_h = -1122.8
        energies = {"BH": 0.0, "B": 7.0 * rtln10 - mu_h + 10.0,
                    "AH": 0.0, "A": 5.0 * rtln10 - mu_h + 10.0}
        stats = pka_statistics(pairs, energies)
        self.assertAlmostEqual(stats["anion_mae_kJ"], 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
