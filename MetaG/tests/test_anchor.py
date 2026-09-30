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

_ATP = "Nc1ncnc2c1ncn2C3OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C3O"
_AMP = "OCC3OC(n1cnc2c(N)ncnc12)C(O)C3O"
_PPI = "OP(=O)(O)OP(=O)(O)O"
# ATP + acetate -> acetyl-AMP + PPi : free PPi produced + mixed anhydride C(=O)-O-P formed
ADENYLYLATE = {"ATP": [-1, -3, _ATP], "acetate": [-1, -1, "CC(=O)O"],
               "PPi": [1, -3, _PPI], "acAMP": [1, -1, "CC(=O)OP(=O)(O)" + _AMP[1:]]}
# N-adenylate: P-N phosphoramidate formed but PPi produced -> must NOT be mis-caught as phosphagen
N_ADENYLATE = {"ATP": [-1, -3, _ATP], "anth": [-1, -1, "Nc1ccccc1C(=O)O"],
               "PPi": [1, -3, _PPI], "Nadyl": [1, -2, "COP(=O)(O)Nc1ccccc1C(=O)O"]}


CATIONIC_PHOSPHATASE = {"ester": [-1, 1, "C[N+](C)(C)CCOP(=O)(O)O"], "water": [-1, 0, "O"],
                        "choline": [1, 1, "C[N+](C)(C)CCO"], "Pi": [1, -1, "OP(=O)(O)O"]}
ADENYLYLATE_AMINOACID = {"ATP": [-1, -3, _ATP], "ser": [-1, 0, "NC(CO)C(=O)O"],
                         "PPi": [1, -3, _PPI], "serAMP": [1, -1, "NC(CO)C(=O)OP(=O)(O)" + _AMP[1:]]}
# carboxy-phosphate (pyruvate carboxylase, real rxn00250 species): HCO3- + ATP + pyruvate ->
# oxaloacetate + ADP + Pi. Carboxyl created on the C4 skeleton; the consumed bicarbonate (O=C([O-])O,
# itself a carboxyl match) must NOT cancel it -> the >4-heavy skeleton gate. Releases Pi not PPi.
CARBOXYP = {
    "ATP": [-1, -4, "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])O)[C@@H](O)[C@H]1O"],
    "Pyruvate": [-1, -1, "CC(=O)C(=O)[O-]"], "H2CO3": [-1, -1, "O=C([O-])O"],
    "ADP": [1, -3, "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)([O-])[O-])[C@@H](O)[C@H]1O"],
    "Phosphate": [1, -2, "O=P([O-])([O-])O"], "Oxaloacetate": [1, -2, "O=C([O-])CC(=O)C(=O)[O-]"]}


def test_subclass_detection():
    assert anchor.subclass(PHOSPHAGEN) == "phosphagen"
    assert anchor.subclass(THIOESTER) == "thioester_pi"
    assert anchor.subclass(PHOSPHATASE) == "phosphatase_monoester"


def test_phosphatase_cationic_excluded():
    # a monoester bearing its own adjacent permanent cation (phosphocholine) is DETECTED but has no
    # validated offset -> anchor_correct returns None (dG_raw reported), not a mis-applied majority offset
    assert anchor.subclass(CATIONIC_PHOSPHATASE) == "phosphatase_monoester_cationic"
    assert anchor.anchor_correct(10.0, CATIONIC_PHOSPHATASE) is None


def test_adenylylate_detection():
    # ATP-driven acyl-adenylate formation (PPi produced + mixed anhydride) -> adenylylate, split by
    # whether the activated acid is a plain aliphatic carboxylate or carries its own alpha-amino group
    assert anchor.subclass(ADENYLYLATE) == "adenylylate_aliphatic"
    out = anchor.anchor_correct(45.0, ADENYLYLATE)
    assert out is not None and out[2] == "adenylylate_aliphatic"
    assert abs(out[0] - (45.0 - anchor.ANCHORS["adenylylate_aliphatic"]["offset"])) < 1e-6


def test_carboxyphosphate_detection():
    # biotin/ATP carboxylase (pyruvate carboxylase) -> carboxyP; bicarbonate consumed must not cancel
    # the carboxyl created on the C-skeleton (heavy>4 gate); releases Pi not PPi so NOT adenylylate
    assert anchor.subclass(CARBOXYP) == "carboxyP"
    out = anchor.anchor_correct(27.8, CARBOXYP)
    assert out is not None and out[2] == "carboxyP"
    assert abs(out[0] - (27.8 - anchor.ANCHORS["carboxyP"]["offset"])) < 1e-6


def test_adenylylate_aminoacid_split():
    assert anchor.subclass(ADENYLYLATE_AMINOACID) == "adenylylate_aminoacid"
    out = anchor.anchor_correct(45.0, ADENYLYLATE_AMINOACID)
    assert out is not None and out[2] == "adenylylate_aminoacid"


def test_n_adenylate_not_phosphagen():
    # an N-adenylate's P-N bond must NOT be mis-corrected as creatine-kinase phosphagen (PPi guard)
    assert anchor.subclass(N_ADENYLATE) != "phosphagen"


def test_isomerization_not_anchored():
    # phosphate group present on BOTH sides (net 0) -> no systematic class fires
    assert anchor.subclass(ISOMERIZATION) is None
    assert anchor.anchor_correct(5.0, ISOMERIZATION) is None


def test_anchor_correct_applies_signed_offset():
    dG = 60.0
    out = anchor.anchor_correct(dG, PHOSPHAGEN)
    assert out is not None
    dG_corr, sigma, sc, direction = out
    # the PHOSPHAGEN fixture DESTROYS the P-N bond (phosphagen hydrolysis); the class's canonical direction
    # (all TECRDB anchors: kinases) CREATES it -> reverse reading, offset applied with the opposite sign
    assert sc == "phosphagen" and direction == -1
    assert abs(dG_corr - (dG + anchor.ANCHORS["phosphagen"]["offset"])) < 1e-6
    assert sigma > 0


def _reverse(sp):
    return {k: [-v[0], v[1], v[2]] for k, v in sp.items()}


def test_anchor_antisymmetric_under_reversal():
    # writing a reaction backwards must flip the anchored ΔG exactly: anchor(-ΔG, rev) == -anchor(ΔG, fwd).
    # (Detection used to fire on "net change != 0" with a fixed-sign offset: reversed creatine kinase got
    # -56 kJ instead of +56; one-directional classes left reversed reactions uncorrected.)
    for sp in (PHOSPHAGEN, THIOESTER, PHOSPHATASE, ADENYLYLATE, ADENYLYLATE_AMINOACID, CARBOXYP):
        fwd = anchor.anchor_correct(30.0, sp)
        rev = anchor.anchor_correct(-30.0, _reverse(sp))
        assert fwd is not None and rev is not None, sp
        assert fwd[2] == rev[2] and fwd[3] == -rev[3]
        assert abs(fwd[0] + rev[0]) < 1e-9


def test_diester_hydrolysis_not_phosphatase():
    # cAMP phosphodiesterase (diester -> monoester) must not get the phosphatase-monoester offset
    pde = {"cAMP": [-1, -1, "Nc1ncnc2c1ncn2[C@@H]1O[C@@H]3COP(=O)([O-])O[C@H]3[C@H]1O"], "H2O": [-1, 0, "O"],
           "AMP": [1, -2, "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])[O-])[C@@H](O)[C@H]1O"]}
    assert anchor.subclass(pde) is None


def test_water_spelling_irrelevant():
    # explicit-H water must be recognised as water by the hydrolysis classes
    sp = {k: ([v[0], v[1], "[H]O[H]"] if v[2] == "O" else v) for k, v in PHOSPHATASE.items()}
    assert anchor.subclass(sp) == anchor.subclass(PHOSPHATASE) == "phosphatase_monoester"


def test_all_anchors_offsets_signed_consistently():
    # each systematic class carries a nonzero, sign-consistent reference/solvation offset. Most are
    # POSITIVE (pipeline too high: anion-solvation wall). amide_hydrolysis is NEGATIVE (pipeline scores
    # hydrolysis too favorable -> too low), corrected upward -- a physically valid negative offset.
    for sc in anchor.ANCHORS:
        assert anchor.ANCHORS[sc]["offset"] != 0
    assert anchor.ANCHORS["amide_hydrolysis"]["offset"] < 0
    for sc in ("phosphagen", "phosphatase_monoester", "thioester_ppi", "carboxyP"):
        assert anchor.ANCHORS[sc]["offset"] > 0


def test_amide_hydrolysis_detection():
    # acyclic amide + water -> acid + amine fires; a bare carboxylate hydrolysis (no amide) does not
    amide = {"sub": [-1, -1, "CC(=O)NCCC(=O)[O-]"], "w": [-1, 0, "O"],
             "acid": [1, -1, "CC(=O)[O-]"], "amine": [1, 1, "[NH3+]CCC(=O)O"]}
    assert anchor.subclass(amide) == "amide_hydrolysis"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  {name} OK")
    print("PASS: anchor detection + correction")
