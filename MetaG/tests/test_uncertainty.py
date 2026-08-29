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
    assert info["half_width"] >= info["sigma_mult"] * info["sigma"] - 1e-6   # heavy-tail floor respected


def test_interval_not_overconfident_small_n():
    # a small-n class must NOT get a spuriously tight interval (the pooled/floored width protects it)
    lo, hi, _, _ = u.prediction_interval("carbamoyl-phosphate synthase", ["O"], 0.0, level=95)
    assert (hi - lo) > 15.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  {name} OK")
    print("PASS: uncertainty layer")
