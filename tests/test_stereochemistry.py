"""Tests for the stereochemical integrity layer.

These lock in the three judgements the module exists to make:
  1. resonance-equivalent phosphate/sulfate centres are NOT missing stereochemistry
     (and the exclusion stays narrow enough to keep a real P centre);
  2. enantiomeric ambiguity is harmless, diastereomeric ambiguity is not;
  3. a reaction with one structure on both sides is degenerate, not a prediction.
"""
import unittest

from qm_thermo import stereochemistry as stereo

ATP = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])O)[C@H](O)[C@@H]1O"
G6P = "O=P([O-])([O-])OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"
PPI = "O=P([O-])([O-])OP(=O)([O-])O"
PROPANEDIOL = "CC(O)CO"
L_ALANINE = "C[C@H](N)C(=O)[O-]"
# Oxaloacetate; ModelSEED stores this same SMILES for "enol-Oxaloacetate".
KETO_OAA = "O=C([O-])CC(=O)C(=O)[O-]"
# A phosphorothioate: phosphorus with four *different* substituents is genuinely
# stereogenic and must survive the artifact filter.
THIOPHOSPHATE = "CO[P@](=O)([S-])OCC"
# Delocalised guanidinium C=N: perceived as an unspecified double bond, but the
# E/Z forms are one substance (identical InChIKey).
PHOSPHOLOMBRICINE = "NC(NCCOP(=O)([O-])OC[C@H]([NH3+])C(=O)[O-])=[NH+]P(=O)([O-])[O-]"


class PhantomCentreTests(unittest.TestCase):
    def test_phosphate_phosphorus_is_not_missing_stereochemistry(self):
        assessment = stereo.assess(ATP)
        self.assertEqual(len(assessment.artifacts), 3)
        self.assertEqual(assessment.undefined, ())
        self.assertEqual(assessment.ambiguity, stereo.AMBIGUITY_NONE)

    def test_pyrophosphate_has_no_real_stereo_element(self):
        assessment = stereo.assess(PPI)
        self.assertEqual(assessment.real, ())
        self.assertEqual(assessment.ambiguity, stereo.AMBIGUITY_NONE)

    def test_exclusion_does_not_swallow_a_genuine_phosphorus_centre(self):
        # Four oxygens is the whole rule; a P-S bond must keep the centre real.
        assessment = stereo.assess(THIOPHOSPHATE)
        self.assertEqual(assessment.artifacts, ())
        self.assertTrue(any(element.role == stereo.CARBON for element in assessment.real))

    def test_enumeration_collapses_phantom_centres(self):
        # Naive enumeration of ATP's three phantom centres would give 2**3.
        self.assertEqual(len(stereo.enumerate_resolved(ATP)), 1)

    def test_inchi_demotes_a_perceived_but_unreal_ambiguity(self):
        # The guanidinium C=N is perceived as unspecified and is not covered by
        # the oxo rule, but all four enumerated forms are one substance.
        assessment = stereo.assess(PHOSPHOLOMBRICINE)
        self.assertTrue(assessment.undefined)          # perception still flags it
        self.assertEqual(assessment.distinct_states, 1)
        self.assertEqual(assessment.ambiguity, stereo.AMBIGUITY_NONE)

    def test_verification_can_be_skipped(self):
        unverified = stereo.assess(PHOSPHOLOMBRICINE, verify=False)
        self.assertIsNone(unverified.distinct_states)
        self.assertEqual(unverified.ambiguity, stereo.AMBIGUITY_DIASTEREOMERIC)


class AmbiguityConsequenceTests(unittest.TestCase):
    def test_anomeric_carbon_is_diastereomeric(self):
        assessment = stereo.assess(G6P)
        self.assertEqual(assessment.ambiguity, stereo.AMBIGUITY_DIASTEREOMERIC)
        self.assertEqual(len(assessment.anomeric_undefined), 1)
        self.assertTrue(assessment.thermodynamically_ambiguous)
        self.assertEqual(len(stereo.enumerate_resolved(G6P)), 2)

    def test_lone_undefined_centre_is_only_enantiomeric(self):
        assessment = stereo.assess(PROPANEDIOL)
        self.assertEqual(assessment.ambiguity, stereo.AMBIGUITY_ENANTIOMERIC)
        # Mirror images share a free energy in an achiral solvent, so this must
        # not be reported as something that can move a number.
        self.assertFalse(assessment.thermodynamically_ambiguous)

    def test_fully_specified_structure_is_unambiguous(self):
        self.assertEqual(stereo.assess(L_ALANINE).ambiguity, stereo.AMBIGUITY_NONE)

    def test_unparseable_structure_is_reported_not_raised(self):
        assessment = stereo.assess("not-a-smiles")
        self.assertTrue(assessment.parse_error)
        self.assertEqual(assessment.ambiguity, stereo.AMBIGUITY_NONE)


class DegeneracyTests(unittest.TestCase):
    def test_identical_structures_both_sides_is_degenerate(self):
        structures = {"cpd00032": KETO_OAA, "cpd02469": KETO_OAA}
        reactions = {"rxn00266": {"cpd00032": -1.0, "cpd02469": 1.0}}
        self.assertIn("rxn00266", stereo.degenerate_reactions(reactions, structures))

    def test_genuinely_different_structures_are_not_degenerate(self):
        structures = {"a": G6P, "b": PROPANEDIOL}
        reactions = {"rxn": {"a": -1.0, "b": 1.0}}
        self.assertEqual(stereo.degenerate_reactions(reactions, structures), {})

    def test_reaction_with_an_unknown_structure_is_skipped_not_guessed(self):
        structures = {"a": G6P, "b": ""}
        reactions = {"rxn": {"a": -1.0, "b": 1.0}}
        self.assertEqual(stereo.degenerate_reactions(reactions, structures), {})

    def test_collisions_group_identical_structures_only(self):
        structures = {"cpd00032": KETO_OAA, "cpd02469": KETO_OAA, "other": PROPANEDIOL}
        collisions = stereo.find_collisions(structures, {"cpd00032": "Oxaloacetate"})
        self.assertEqual(len(collisions), 1)
        self.assertEqual(collisions[0].compound_ids, ("cpd00032", "cpd02469"))


class StructureGuardTests(unittest.TestCase):
    """The loader must refuse inputs that would mix diastereomers in one ensemble."""

    def _meta(self, smiles, charge):
        return {"id": "cpdTEST", "smiles": smiles, "charge": charge}

    def test_diastereomeric_input_is_rejected(self):
        from qm_thermo import structures
        with self.assertRaises(structures.StructureError) as caught:
            structures._validate_stereochemistry(self._meta(G6P, -2))
        self.assertIn("resolve_stereochemistry", str(caught.exception))

    def test_enantiomeric_input_is_allowed(self):
        from qm_thermo import structures
        structures._validate_stereochemistry(self._meta(PROPANEDIOL, 0))

    def test_phantom_centres_do_not_trip_the_guard(self):
        from qm_thermo import structures
        structures._validate_stereochemistry(self._meta(ATP, -3))


class IsomerFamilyTests(unittest.TestCase):
    def test_population_weighted_correction_matches_rt_ln_f(self):
        import math

        from qm_thermo.speciation import IsomerFamily, R_KJ

        family = IsomerFamily(
            compound_id="cpd00079", state_labels=("alpha", "beta"),
            populations=(0.36, 0.64), source="unit-test", citation="unit-test")
        self.assertTrue(family.resolved)
        expected = R_KJ * 298.15 * math.log(0.64)
        self.assertAlmostEqual(family.ensemble_correction_kJ("beta"), expected, places=9)
        # Anchoring on the minor state must give the larger (more negative) shift.
        self.assertLess(family.ensemble_correction_kJ("alpha"),
                        family.ensemble_correction_kJ("beta"))

    def test_populations_without_provenance_are_rejected(self):
        from qm_thermo.speciation import IsomerFamily

        with self.assertRaises(ValueError):
            IsomerFamily(compound_id="x", state_labels=("a", "b"), populations=(0.5, 0.5))

    def test_populations_must_be_normalised(self):
        from qm_thermo.speciation import IsomerFamily

        with self.assertRaises(ValueError):
            IsomerFamily(compound_id="x", state_labels=("a", "b"), populations=(0.5, 0.9),
                         source="s", citation="c")

    def test_unresolved_family_refuses_to_invent_a_weight(self):
        from qm_thermo.speciation import IsomerFamily

        family = IsomerFamily(compound_id="x", state_labels=("a", "b"))
        self.assertFalse(family.resolved)
        with self.assertRaises(ValueError):
            family.ensemble_correction_kJ("a")
        self.assertAlmostEqual(family.state_spread_kJ({"a": -10.0, "b": -4.0}), 6.0)


class ReactionMappingTests(unittest.TestCase):
    def test_affected_reactions_report_the_offending_compounds(self):
        reactions = {"r1": {"cpd00079": -1.0, "x": 1.0}, "r2": {"y": -1.0, "z": 1.0}}
        affected = stereo.affected_reactions(reactions, ["cpd00079"])
        self.assertEqual(affected, {"r1": ("cpd00079",)})


if __name__ == "__main__":
    unittest.main()
