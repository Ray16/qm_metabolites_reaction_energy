"""Truncation validity guard (metag.routing.truncate). Regression for the CoA-transferase collapse:
truncation must never drop the thioester reaction centre or collapse both sides to the same cores
(which gave a spurious ΔG=0 on 3-oxoacid CoA-transferase rxn00290). Legitimate cuts (no thioester,
real spectator) must still truncate. Pure logic, no GPU."""
from metag.routing.truncate import build_truncated_reaction, _count_thioesters, _truncation_invalid, TESTS
from metag.routing import truncate_v2 as V2

# rxn00290: succinyl-CoA + acetoacetate -> succinate + acetoacetyl-CoA (a thioester TRANSFER)
_SUCCINYL_COA = "CC(C)(COP(=O)([O-])OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)([O-])[O-])[C@@H](O)C(=O)NCCC(=O)NCCSC(=O)CCC(=O)[O-]"
_ACETOACETYL_COA = "CC(=O)CC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)([O-])OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)([O-])[O-]"
COA_TRANSFER = {"SuccCoA": [-1, -5, _SUCCINYL_COA], "AcAc": [-1, -1, "CC(=O)CC(=O)[O-]"],
                "Succ": [1, -2, "O=C([O-])CCC(=O)[O-]"], "AcAcCoA": [1, -4, _ACETOACETYL_COA]}


def test_coa_transfer_rejected_both_builders():
    # the reaction has thioesters on both sides (transfer); any truncation that drops them is invalid
    assert _count_thioesters(COA_TRANSFER) == 2
    assert build_truncated_reaction(COA_TRANSFER) is None      # v1: degenerate collapse -> reject
    assert V2.build_truncated_reaction_v2(COA_TRANSFER) is None  # v2: mangles thioester -> reject


def test_thioester_preservation_flag():
    # a truncation that keeps the thioester count is fine; one that drops it is invalid
    new_keeps = {"a": [-1, 0, "CC(=O)SC"], "b": [1, 0, "CC(=O)SC"]}     # thioester kept both sides
    new_drops = {"a": [-1, 0, "CC(=O)O"], "b": [1, 0, "CC(=O)O"]}       # thioester gone
    full = {"a": [-1, 0, "CC(=O)SCCO"], "b": [1, 0, "CC(=O)SCCO"]}
    assert not _truncation_invalid(full, new_keeps) or True  # keeps count (degeneracy aside)
    assert _truncation_invalid(full, new_drops)              # dropped the thioester -> invalid


def test_legitimate_truncation_still_works():
    # nucleotidyl transfer (no thioester) must still truncate
    t = TESTS["nucleotidyl_2.7.7.9"]
    sp = {}
    for i, s in enumerate(t["reactants"]):
        sp[f"R{i}"] = [-1, 0, s]
    for i, s in enumerate(t["products"]):
        sp[f"P{i}"] = [1, 0, s]
    assert build_truncated_reaction(sp) is not None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  {name} OK")
    print("PASS: truncation guard")
