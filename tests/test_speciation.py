"""Tests for the protonation-ensemble correction.

The properties worth locking in are the bound and the decay, because they are
what makes this term safe to apply everywhere: it cannot silently become large,
so switching it on can only be a small correctness adjustment and never a
disguised fit.
"""
import math
import unittest

from qm_thermo.speciation import (
    R_KJ,
    independent_site_correction_kJ,
    monoprotic_family_correction_kJ,
)

RT = R_KJ * 298.15


class IndependentSiteCorrectionTests(unittest.TestCase):
    def test_no_sites_is_no_correction(self):
        self.assertEqual(independent_site_correction_kJ([], 7.0), 0.0)

    def test_maximum_is_rt_ln2_at_pk_equals_ph(self):
        # A site titrating exactly at the working pH is 50:50, the most mixed a
        # single site can be, so this is the largest correction it can produce.
        value = independent_site_correction_kJ([7.0], 7.0)
        self.assertAlmostEqual(value, -RT * math.log(2), places=9)
        self.assertAlmostEqual(value, -1.718, places=2)

    def test_correction_is_never_positive_and_never_exceeds_the_bound(self):
        for pK in (-4.0, 0.0, 3.3, 6.9, 7.0, 7.1, 12.0, 16.4):
            value = independent_site_correction_kJ([pK], 7.0)
            self.assertLessEqual(value, 0.0)
            self.assertGreaterEqual(value, -RT * math.log(2) - 1e-12)

    def test_decays_with_distance_from_ph(self):
        near = independent_site_correction_kJ([7.0], 7.0)
        one = independent_site_correction_kJ([8.0], 7.0)
        two = independent_site_correction_kJ([9.0], 7.0)
        self.assertLess(near, one)
        self.assertLess(one, two)
        self.assertAlmostEqual(one, -0.236, places=3)
        self.assertAlmostEqual(two, -0.025, places=3)

    def test_symmetric_above_and_below_ph(self):
        # An acid one unit below pH and a base one unit above are equally mixed.
        self.assertAlmostEqual(independent_site_correction_kJ([6.0], 7.0),
                               independent_site_correction_kJ([8.0], 7.0), places=12)

    def test_sites_are_additive(self):
        combined = independent_site_correction_kJ([7.0, 8.0], 7.0)
        separate = (independent_site_correction_kJ([7.0], 7.0)
                    + independent_site_correction_kJ([8.0], 7.0))
        self.assertAlmostEqual(combined, separate, places=12)

    def test_agrees_with_the_monoprotic_family_form_at_the_dominant_state(self):
        # For a single acidic site above pH the dominant state is protonated, and
        # the curated monoprotic helper must give the same mixing entropy.
        pKa, pH = 8.0, 7.0
        self.assertAlmostEqual(independent_site_correction_kJ([pKa], pH),
                               monoprotic_family_correction_kJ(pH, pKa), places=12)


if __name__ == "__main__":
    unittest.main()
