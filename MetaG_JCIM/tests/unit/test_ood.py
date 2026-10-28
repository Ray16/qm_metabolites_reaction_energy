"""Out-of-distribution FLAG layer (metag.routing.applicability): surfaces notable structural features for
transparency but NEVER changes sigma -- a "unlike the training set -> widen" floor would fire on the
frontier reactions the method exists to score, defeating coverage. UMA generalizes (universal potential);
the honest uncertainty is a COMPUTED physics sigma, not a structural floor."""
from metag.routing.applicability import ood_assessment
from metag import uncertainty as u

_ATP = "Nc1ncnc2c1ncn2C3OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C3O"

# de-novo pyridine (quinolinate) formation -- multi-bond condensation class
CONDENSATION = {"open": [-1, 0, "O=C(O)CC(=NC(=O)O)C(=O)O"], "quinolinate": [1, 0, "O=C(O)c1cccnc1C(=O)O"]}
# divalent cation (Mg) + phosphate -> coordination-speciation NOTE (not a UMA limit)
METAL = {"ATP": [-1, -2, _ATP], "Mg": [-1, 2, "[Mg+2]"], "MgATP": [1, 0, "[Mg]"]}
# clean carbocyclisation (no heteroaromatic ring, CHNOPS, small) -- must NOT flag
CLEAN = {"a": [-1, 0, "C1CCCCC1=C"], "b": [1, 0, "C1CCC2CCCCC2C1"]}


def test_flags_never_change_sigma():
    # NOTHING in the OOD layer inflates sigma: sigma_floor is always 0, reasons always empty
    for sp in (CONDENSATION, METAL, CLEAN):
        oa = ood_assessment(sp)
        assert oa["sigma_floor"] == 0.0 and oa["reasons"] == []


def test_metal_is_speciation_flag_not_floor():
    oa = ood_assessment(METAL)
    assert oa["ood"] and oa["sigma_floor"] == 0.0                 # flagged, but no sigma effect
    assert any("speciation" in f for f in oa["flags"])            # coordination-speciation note, not "OOD element"


def test_condensation_flag_only():
    oa = ood_assessment(CONDENSATION)
    assert oa["sigma_floor"] == 0.0
    assert any("heterocycle" in f for f in oa["flags"])


def test_clean_not_flagged():
    oa = ood_assessment(CLEAN)
    assert not oa["ood"] and oa["sigma_floor"] == 0.0 and not oa["flags"]


def test_reaction_sigma_never_floored_by_ood():
    # the OOD layer must NEVER floor sigma now (metal reaction: sigma == class value, no floor applied)
    sig, br = u.reaction_sigma("Mg-dependent enzyme", [v[2] for v in METAL.values()], 2.0, species=METAL)
    assert "sigma_floored_from" not in br                          # no floor ever applied
    assert br.get("ood_flags")                                     # but flags surfaced for transparency


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  {name} OK")
    print("PASS: OOD gate")
