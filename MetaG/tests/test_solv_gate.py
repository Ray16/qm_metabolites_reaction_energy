"""Hydro-lyase water-reference gate (metag.routing.solv_gate.is_hydrolyase). The wired water-reference
correction (pipeline.py, WATER_REF_HYDROLYASE) fires ONLY on this gate, so its specificity is the safety
guarantee: it must FIRE on neutral C=C+H2O<->C-OH hydrations and NEVER on phosphate/anion/isomer reactions
(which are anchored or hide a competing solvation error -- a global water fix regresses them)."""
from metag.routing.solv_gate import is_hydrolyase, needs_smd

# fumarate + H2O <-> malate (stored dehydration direction): a clean hydro-lyase -> MUST fire
HYDRATASE = {"L-Malate": (-1, -2, "O=C([O-])C[C@H](O)C(=O)[O-]"), "H2O": (1, 0, "O"),
             "Fumarate": (1, -2, "O=C([O-])/C=C/C(=O)[O-]")}
# enolase: makes an enol-PHOSPHATE (PEP) -> has phosphorus -> MUST NOT fire
ENOLASE = {"2PG": (-1, -3, "O=C([O-])C(OP(=O)([O-])[O-])CO"), "PEP": (1, -3, "O=C([O-])C(=C)OP(=O)([O-])[O-]"),
           "H2O": (1, 0, "O")}
# phosphatase hydrolysis: consumes water but is a phosphate monoester (anchored) -> MUST NOT fire
PHOSPHATASE = {"ester": (-1, -1, "COP(=O)([O-])[O-]"), "water": (-1, 0, "O"),
               "alcohol": (1, 0, "CO"), "Pi": (1, -1, "OP(=O)([O-])[O-]")}
# isomerase (no net water, groups conserved) -> MUST NOT fire
ISOMERASE = {"g6p": (-1, -2, "OCC1OC(O)C(O)C(O)C1OP(=O)([O-])[O-]"),
             "f6p": (1, -2, "OC1(CO)OC(COP(=O)([O-])[O-])C(O)C1O")}
# amide hydrolysis: consumes water, changes groups, but NOT a C=C<->C-OH hydration -> MUST NOT fire
AMIDE = {"acetamide": (-1, 0, "CC(N)=O"), "water": (-1, 0, "O"), "acetate": (1, -1, "CC(=O)[O-]"),
         "NH3": (1, 0, "N")}


def test_hydrolyase_fires_on_hydratase():
    assert is_hydrolyase(HYDRATASE)


def test_hydrolyase_skips_phosphate_and_anion():
    assert not is_hydrolyase(ENOLASE)         # enol-phosphate (P present)
    assert not is_hydrolyase(PHOSPHATASE)     # phosphate monoester (P present, anchored)


def test_hydrolyase_skips_isomerase_and_amide():
    assert not is_hydrolyase(ISOMERASE)       # no net water, no C=C/C-OH change
    assert not is_hydrolyase(AMIDE)           # net water but not an alkene hydration


def test_needs_smd_still_gates():
    # sanity: the SMD structural gate still fires on a created polar group and skips isomerase
    assert needs_smd(HYDRATASE)[0]
    assert not needs_smd(ISOMERASE)[0]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  {name} OK")
    print("PASS: hydro-lyase water-reference gate")
