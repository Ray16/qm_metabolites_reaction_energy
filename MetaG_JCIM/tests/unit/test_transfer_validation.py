import pytest
import os
from pathlib import Path
import subprocess
import sys

from metag.validation.transfer import audit_holdout, compound_identity, reaction_signature, training_span_exposure
from metag.tools.audit_transfer import build_audit


def target(row, kind, pairs, families=()):
    return {"row": row, "type": kind, "signature": pairs,
            "compounds": [s for s, _ in pairs], "families": list(families)}


def test_reversed_fractional_equations_and_spectators_match():
    a = [("CCO", -0.5), ("CC=O", 0.5), ("O", 2), ("O", -2)]
    b = [("CC=O", -3), ("CCO", 3)]
    assert reaction_signature(a) == reaction_signature(b)
    assert reaction_signature(a) != reaction_signature(b + [("O", 1)])


def test_stereo_policy_exposes_encoder_equivalent_reactions():
    a = [("C/C=C/C", -1), ("CCCC", 1)]
    b = [("C/C=C\\C", -1), ("CCCC", 1)]
    assert reaction_signature(a) != reaction_signature(b)
    assert reaction_signature(a, stereo_sensitive=False) == reaction_signature(b, stereo_sensitive=False)
    training = [target(12, "reaction dg", a)]
    audit = audit_holdout(training, [{"id": "test", "signature": b}], [12])
    assert not audit["passed"]
    assert audit["violations"][0]["rows"] == [12]


def test_compound_holdout_catches_formation_targets_and_explicit_hydrogens():
    training = [target(9, "formation dg", [("[H]OC([H])([H])C([H])([H])[H]", -1)]),
                target(2, "reaction dg", [("CCO", -1), ("CC=O", 1)])]
    audit = audit_holdout(training, [], [9], heldout_compounds=["CCO"])
    assert not audit["passed"]
    assert audit["violations"][0]["target_type"] == "formation dg"
    assert audit_holdout(training, [], [], heldout_compounds=["CCO"])["passed"]


def test_family_holdout_reports_retained_formation_exposure():
    training = [target(42, "formation dg", [("CCO", -1)]),
                target(3, "formation dg", [("CC=O", -1)]),
                target(90, "reaction dg", [("CCO", -1), ("CC=O", 1)], ["1.1.1"])]
    test = {"id": "r", "normalized_compounds": ["CCO", "CC=O"], "published_training_rows": [90]}
    audit = audit_holdout(training, [test], [42, 3], heldout_families=["1.1.1"])
    assert audit["passed"]
    assert audit["n_training_formations"] == 2
    assert audit["compound_exposure"][0]["all_compounds_seen"]
    span = training_span_exposure(training, [42, 3], [("CCO", -1), ("CC=O", 1)])
    assert span["in_training_stoichiometric_span"]
    assert not training_span_exposure(training, [42], [("CCO", -1), ("CC=O", 1)])["in_training_stoichiometric_span"]


def test_hess_law_exposure_without_formation_labels():
    training = [target(7, "reaction dg", [("C", -1), ("CC", 1)]),
                target(8, "reaction dg", [("CC", -1), ("CCC", 1)])]
    assert training_span_exposure(training, [7, 8], [("C", -1), ("CCC", 1)])["in_training_stoichiometric_span"]


def test_generic_training_fragments_are_recorded_not_silently_dropped():
    training = [target(4, "reaction dg", [("*S", -1), ("*SC=O", 1)])]
    audit = audit_holdout(training, [], [4])
    assert audit["generic_structure_training_rows"] == [4]
    assert audit["n_training_reactions"] == 1


def test_atom_maps_do_not_hide_equivalence_and_isotopes_remain_distinct():
    a = [("[CH3:1][CH2:2][OH:3]", -1), ("[CH3:1][CH:2]=[O:3]", 1)]
    b = [("CCO", -1), ("CC=O", 1)]
    assert reaction_signature(a) == reaction_signature(b)
    assert compound_identity("[13CH3]CO") != compound_identity("CCO")


def test_pure_stereo_target_reports_encoder_collision_without_false_reaction_match():
    pairs = [("C/C=C/C", -1), ("C/C=C\\C", 1)]
    unrelated = [("C[C@H](O)[C@H](O)C", -1), ("C[C@H](O)[C@@H](O)C", 1)]
    training = [target(8, "reaction dg", unrelated)]
    report = training_span_exposure(training, [8], pairs)
    assert report["status"] == "encoder_collapsed_zero"
    audit = audit_holdout(training, [{"id": "stereo", "signature": pairs}], [8])
    assert audit["passed"]
    assert audit["target_representation_collisions"][0]["target"] == "stereo"
    assert not training_span_exposure(training, [8], pairs, stereo_sensitive=True)["in_training_stoichiometric_span"]


def test_physically_cancelled_target_is_rejected():
    with pytest.raises(ValueError, match="physical net stoichiometry"):
        training_span_exposure([], [], [("CCO", -1), ("CCO", 1)])


@pytest.mark.parametrize("rows", [[999], [1, 1]])
def test_unknown_and_duplicate_training_rows_rejected(rows):
    with pytest.raises(ValueError):
        audit_holdout([target(1, "formation dg", [("O", 1)])], [], rows)


def test_realistic_manifest_audit_uses_row_ids_not_list_positions():
    manifest = {"training": [target(90, "reaction dg", [("CCO", -1), ("CC=O", 1)], ["1.1.1"])],
                "evaluation": {"r": {"published_training_rows": [90], "normalized_compounds": ["CCO", "CC=O"]}}}
    splits = {"reaction_family_folds": [{"id": "f", "family": "1.1.1", "test_reactions": ["r"], "train_rows": []}]}
    result = build_audit(manifest, splits, check_span=True)
    assert result["summary"]["n_passed"] == 1
    assert not result["folds"][0]["training_span"]["r"]["in_training_stoichiometric_span"]


@pytest.mark.parametrize("module", ["metag.tools.audit_transfer", "metag.tools.evaluate"])
def test_validation_command_help_without_site_packages(module):
    source = Path(__file__).resolve().parents[2] / "src"
    result = subprocess.run([sys.executable, "-S", "-m", module, "--help"],
                            env={**os.environ, "PYTHONPATH": str(source)},
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
