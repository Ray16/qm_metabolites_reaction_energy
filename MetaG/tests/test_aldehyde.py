"""Aldehyde-hydration gate (metag.routing.aldehyde). The gem-diol correction fires ONLY for
alpha-EWG-activated aldehydes (glyoxylate/methylglyoxal/formaldehyde), NOT for ordinary alkyl aldehydes
(GAP, acetaldehyde) whose UMA hydration free energy is unreliable -- the physical gate that made the
correction a net positive on the benchmark."""
from metag.routing import aldehyde

HYDRATED = ["O=CC(=O)[O-]",     # glyoxylate
            "CC(=O)C=O",         # methylglyoxal
            "C=O"]               # formaldehyde
NOT_HYDRATED = ["CC=O",          # acetaldehyde (alpha-alkyl)
                "OCC(O)C=O",     # glyceraldehyde-ish alkyl aldehyde
                "CC(=O)[O-]"]    # acetate (no aldehyde at all)


def test_ewg_aldehydes_are_hydrated():
    for smi in HYDRATED:
        assert aldehyde.is_strongly_hydrated(smi), f"{smi} should be gated IN"


def test_alkyl_aldehydes_not_hydrated():
    for smi in NOT_HYDRATED:
        assert not aldehyde.is_strongly_hydrated(smi), f"{smi} should be gated OUT"


def test_gem_diol_adds_water():
    diol = aldehyde.gem_diol("C=O")            # formaldehyde -> methanediol
    assert diol is not None and diol.count("O") >= 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  {name} OK")
    print("PASS: aldehyde gate")
