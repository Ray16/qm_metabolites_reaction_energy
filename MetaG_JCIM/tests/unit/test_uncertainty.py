"""Calibrated uncertainty (metag.uncertainty): mechanism classification, sigma_pred = sqrt(U_samp^2 +
sigma_class^2), and the symmetric CV-validated prediction interval with the class bias as separate
point-estimate metadata. Confirms the shipped artifact loads and produces sane, non-overconfident values."""
from metag import uncertainty as u


def test_calibration_loaded():
    assert len(u.CLASS_STATS) >= 10          # real calibrated artifact, not the fallback stub
    assert 1.8 <= u._INTERVAL_MULT <= 3.0


def test_mech_class_examples():
    assert u.mech_class("creatine kinase | EC=2.7.3.2", ["O=P(O)(O)O"]) == "phosphagen(P-N/Mg)"
    assert u.mech_class("triose-phosphate isomerase", ["CC(=O)C(=O)[O-]"]) == "isomerase/mutase"
    assert u.mech_class("some novel enzyme", ["CCO"]) == "other/clean"


def test_structural_class_overrides_note():
    # STRUCTURAL detection wins over the note: two adenylyl-transfers (same chemistry) must get the SAME
    # class regardless of note ("adenylyltransferase" mis-maps to kinase; a cryptic note -> other/clean).
    ATP = "Nc1ncnc2c1ncn2C3OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C3O"
    AMP = "OCC3OC(n1cnc2c(N)ncnc12)C(O)C3O"
    PPi = "OP(=O)(O)OP(=O)(O)O"
    adyl = {"ATP": [-1, -3, ATP], "acetate": [-1, -1, "CC(=O)O"],
            "PPi": [1, -3, PPi], "acAMP": [1, -1, "CC(=O)OP(=O)(O)" + AMP[1:]]}
    smis = [v[2] for v in adyl.values()]
    # note says "adenylyltransferase" (note-only -> kinase) and a cryptic note; structural must override both
    assert u.mech_class("ATP:acetate adenylyltransferase", smis) == "kinase/phosphotransfer"   # note-only
    assert u.mech_class("ATP:acetate adenylyltransferase", smis, species=adyl) == "adenylylate(external-ref)"
    assert u.mech_class("ENTF-RXN.c", smis, species=adyl) == "adenylylate(external-ref)"                    # note-independent
    # -> same sigma for the same chemistry, whatever the note
    s1, _ = u.reaction_sigma("ATP:acetate adenylyltransferase", smis, 2.0, species=adyl)
    s2, _ = u.reaction_sigma("ENTF-RXN.c", smis, 2.0, species=adyl)
    assert s1 == s2


def test_reaction_sigma_dominated_by_class():
    # the predictive sigma must be the class-level error (~10-25 kJ), NOT the ~1-3 kJ sampling spread
    sig, br = u.reaction_sigma("malate dehydrogenase (NAD)", ["CC(=O)C(=O)[O-]"], U_samp=2.0)
    assert sig > 8.0
    assert br["sigma_class"] > br["U_samp"]


def test_prediction_interval_symmetric_and_covers():
    lo, hi, center, info = u.prediction_interval("fumarate hydratase", ["OC(=O)CC(O)C(=O)O"], 5.0, level=95)
    assert lo < 5.0 < hi                                  # brackets the prediction
    assert center == 5.0                                  # symmetric: centred on dG, not de-biased
    assert abs((hi - center) - (center - lo)) < 1e-6      # symmetric half-widths
    assert "point_bias" in info                           # bias reported as SEPARATE metadata
    assert info["half_width"] >= info["sigma_mult"] * info["sigma"] - 0.05   # floor respected (half_width rounded to 0.1)


def test_interval_not_overconfident_small_n():
    # a small-n class must NOT get a spuriously tight interval (the pooled/floored width protects it)
    note = "ornithine carbamoyltransferase"                    # actually lands in the small-n class
    assert u.mech_class(note, ["O"]) == "carbamoyltransfer"
    lo, hi, _, _ = u.prediction_interval(note, ["O"], 0.0, level=95)
    assert (hi - lo) > 15.0


def test_ci95_includes_U_samp():
    s0, _ = u.reaction_sigma("some novel enzyme", ["CCO"], 0.0)
    s30, _ = u.reaction_sigma("some novel enzyme", ["CCO"], 30.0)
    _, hi0, _, i0 = u.prediction_interval("some novel enzyme", ["CCO"], 0.0, U_samp=0.0)
    _, hi30, _, i30 = u.prediction_interval("some novel enzyme", ["CCO"], 0.0, U_samp=30.0)
    assert s30 > s0 + 15
    assert hi30 > hi0 + 15                                   # was: identical half-width
    assert abs(i30["sigma"] - s30) < 0.11                    # same sigma_total as reaction_sigma


def test_levels_monotone_and_flagged():
    hw = {L: u.prediction_interval("some novel enzyme", ["CCO"], 0.0, level=L)[3]["half_width"]
          for L in (68, 90, 95, 99)}
    assert hw[68] < hw[90] < hw[95] < hw[99]                 # was: 68/90/99 all ~half of 95
    assert u.prediction_interval("x", ["CCO"], 0.0, level=90)[3]["level_calibrated"] is False


def test_uncalibrated_class_not_narrower_than_calibrated(monkeypatch):
    monkeypatch.setattr(u, "mech_class", lambda *a, **k: "brand-new-class")
    _, hi, _, info = u.prediction_interval("x", ["CCO"], 0.0)
    assert info["class_calibrated"] is False and info["externally_calibrated"] is False
    assert "not coverage-calibrated" in info["calibration_scope"]
    monkeypatch.undo()
    _, hi_clean, _, _ = u.prediction_interval("some novel enzyme", ["CCO"], 0.0)
    assert hi >= hi_clean - 1e-6


def test_ood_flagged_interval_labelled_not_calibrated():
    # a rare element is an informational OOD flag -> interval explicitly not coverage-calibrated
    sp = {"a": [-1, 0, "CCO"], "m": [-1, 0, "[Se]"], "b": [1, 0, "CC=O"]}
    _, _, _, info = u.prediction_interval("x", [v[2] for v in sp.values()], 0.0, species=sp)
    assert info["ood_flags"] and info["externally_calibrated"] is False
    _, br = u.reaction_sigma("x", [v[2] for v in sp.values()], 0.0, species=sp)
    assert br["externally_calibrated"] is False


def test_coa_keyword_is_whole_word():
    assert u.mech_class("glucoamylase", ["CCO"]) != "CoA-thioester"
    assert u.mech_class("coagulation factor", ["CCO"]) != "CoA-thioester"
    assert u.mech_class("acetyl-CoA synthetase", ["CCO"]) == "CoA-thioester"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  {name} OK")
    print("PASS: uncertainty layer")
