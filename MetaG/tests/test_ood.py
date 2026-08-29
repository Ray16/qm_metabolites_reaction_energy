"""Out-of-distribution gate (metag.routing.ood): flags reactions structurally unlike the TECRDB
calibration set and FLOORS sigma (never narrows). Signals validated against known verdicts: fires on
metals + de-novo aromatic N-heterocycle condensations; does NOT fire on O2/aromatic oxidation (UMA
validated accurate) or carbocyclisations."""
from metag.routing.ood import ood_assessment
from metag import uncertainty as u

_ATP = "Nc1ncnc2c1ncn2C3OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C3O"
_PPI = "OP(=O)(O)OP(=O)(O)O"

# de-novo pyridine (quinolinate) formation -- multi-bond condensation class
CONDENSATION = {"open": [-1, 0, "O=C(O)CC(=NC(=O)O)C(=O)O"], "quinolinate": [1, 0, "O=C(O)c1cccnc1C(=O)O"]}
# metal-containing (Mg) -- uncommon element outside CHNOPS
METAL = {"ATP": [-1, -2, _ATP], "Mg": [-1, 2, "[Mg+2]"], "MgATP": [1, 0, "[Mg]"]}
# clean carbocyclisation (no heteroaromatic ring, CHNOPS, small) -- must NOT flag
CLEAN = {"a": [-1, 0, "C1CCCCC1=C"], "b": [1, 0, "C1CCC2CCCCC2C1"]}


def test_metal_flagged_strong():
    oa = ood_assessment(METAL)
    assert oa["ood"] and oa["sigma_floor"] >= 30.0
    assert any("element" in r for r in oa["reasons"])


def test_condensation_flagged():
    oa = ood_assessment(CONDENSATION)
    assert oa["ood"] and oa["sigma_floor"] >= 20.0
    assert any("heterocycle" in r for r in oa["reasons"])


def test_clean_not_flagged():
    oa = ood_assessment(CLEAN)
    assert not oa["ood"] and oa["sigma_floor"] == 0.0


def test_sigma_floored_when_ood_never_narrowed():
    # OOD reaction: sigma is floored up to the OOD level and flagged in the breakdown
    sig, br = u.reaction_sigma("Mg-dependent enzyme", [v[2] for v in METAL.values()], 2.0, species=METAL)
    assert br["ood"] and sig >= 30.0 and "sigma_floored_from" in br
    # clean reaction: OOD does NOT narrow -- sigma stays the class value, no floor applied
    sig2, br2 = u.reaction_sigma("some enzyme", [v[2] for v in CLEAN.values()], 2.0, species=CLEAN)
    assert not br2["ood"] and "sigma_floored_from" not in br2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  {name} OK")
    print("PASS: OOD gate")
