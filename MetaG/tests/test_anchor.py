"""Anchor sub-class detection + correction (metag.routing.anchor). Verifies the 3 systematic classes are
detected STRUCTURALLY (SMARTS on created/destroyed groups) and that anchor_correct subtracts the calibrated
offset; and that a reaction matching no pattern (and an isomerization) is left untouched."""
from metag.routing import anchor

# species dict form: {name: [coeff, charge, smiles]}
PHOSPHAGEN = {"creatineP": [-1, -1, "NP(=O)(O)O"], "creatine": [1, 0, "N"], "Pi": [1, -1, "OP(=O)(O)O"]}
THIOESTER = {"acetate": [-1, -1, "CC(=O)O"], "thiol": [-1, 0, "CS"], "ATP": [-1, -2, "OP(=O)(O)OP(=O)(O)O"],
             "acylS": [1, 0, "CC(=O)SC"], "AMP": [1, -1, "CO"]}
PHOSPHATASE = {"ester": [-1, -1, "COP(=O)(O)O"], "water": [-1, 0, "O"], "alcohol": [1, 0, "CO"],
               "Pi": [1, -1, "OP(=O)(O)O"]}
ISOMERIZATION = {"g6p": [-1, -2, "OCC1OC(O)C(O)C(O)C1OP(=O)(O)O"],
                 "f6p": [1, -2, "OC1(CO)OC(COP(=O)(O)O)C(O)C1O"]}   # phosphate both sides, net 0


def test_subclass_detection():
    assert anchor.subclass(PHOSPHAGEN) == "phosphagen"
    assert anchor.subclass(THIOESTER) == "thioester"
    assert anchor.subclass(PHOSPHATASE) == "phosphatase_monoester"


def test_isomerization_not_anchored():
    # phosphate group present on BOTH sides (net 0) -> no systematic class fires
    assert anchor.subclass(ISOMERIZATION) is None
    assert anchor.anchor_correct(5.0, ISOMERIZATION) is None


def test_anchor_correct_subtracts_offset():
    dG = 60.0
    out = anchor.anchor_correct(dG, PHOSPHAGEN)
    assert out is not None
    dG_corr, sigma, sc = out
    assert sc == "phosphagen"
    assert abs(dG_corr - (dG - anchor.ANCHORS["phosphagen"]["offset"])) < 1e-6
    assert sigma > 0


def test_all_three_offsets_positive():
    # all 3 systematic classes carry a positive solvation offset (sign-consistent)
    for sc in ("phosphagen", "phosphatase_monoester", "thioester"):
        assert anchor.ANCHORS[sc]["offset"] > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  {name} OK")
    print("PASS: anchor detection + correction")
