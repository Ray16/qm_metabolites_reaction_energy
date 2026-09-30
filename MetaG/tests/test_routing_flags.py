"""Routing changes under A/B stay OFF by default (validated behaviour) and do what they claim when enabled."""
import pytest

P = pytest.importorskip("metag.pipeline")
from metag.routing.truncate import build_truncated_reaction

# PPDK-type core: ATP + pyruvate + H2O -> AMP + PEP + Pi (nucleoside is the spectator)
_ATP = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])O)[C@@H](O)[C@H]1O"
_AMP = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])[O-])[C@@H](O)[C@H]1O"
PPDK = {"H2O": [-1, 0, "O"], "ATP": [-1, -3, _ATP], "Pyr": [-1, -1, "CC(=O)C(=O)[O-]"],
        "Pi": [1, -2, "O=P([O-])([O-])O"], "AMP": [1, -2, _AMP], "PEP": [1, -3, "C=C(OP(=O)([O-])[O-])C(=O)[O-]"]}


def test_ab_flags_default_off():
    for f in ("TRUNC_SPECTATOR_CATIONS", "TRUNC_MAXANION_RETRY", "TRUNC_FG_CUTS"):
        assert P.FLAG_DEFAULTS[f] is False


def test_fg_cuts_never_cap_a_phosphoester_oxygen():
    old = build_truncated_reaction(PPDK)                    # default: may cut C5'-O and H-cap the O
    new = build_truncated_reaction(PPDK, fg_cuts=True)
    assert new is not None
    amp_new = new[0]["AMP_t"][2]
    from rdkit import Chem
    m = Chem.MolFromSmiles(amp_new)
    # the AMP core keeps a C-O-P ester (methyl phosphate or larger), never becomes free H3PO4
    assert m.HasSubstructMatch(Chem.MolFromSmarts("[#6]-[OX2]-P")), amp_new
    assert old is None or old[0]["AMP_t"][2] != amp_new
