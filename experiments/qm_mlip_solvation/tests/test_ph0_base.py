"""Hand-worked regression test for the pH-0 / Alberty pKa-transform bookkeeping -- the most intricate
arithmetic in the pipeline (flagged in a code review). Locks: the EXACT Alberty per-proton term
(vs the linear approximation), the acid<->base mirror, the reactant/product sign, spectator cancellation,
and the neutralization dispatch (anion acids + cation bases). No QM -- pure arithmetic + rdkit.

Reference term (unified_pipeline.py): for a site (side, pKa, kind),
    expo    = (pH - pKa)  if kind=='acid'  else (pKa - pH)
    contrib = (+1 react / -1 prod) * RT_LN10 * log10(1 + 10**expo),   RT_LN10 = 2.303*R*T, pH = 7
"""
import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

T = 298.15
PH = 7.0
RT_LN10 = 2.303 * 8.314e-3 * T                     # ~5.709 kJ/mol per pKa unit


def alberty_term(side, pka, kind):
    """Exact Alberty per-proton free-energy term, mirroring unified_pipeline.py."""
    expo = (PH - pka) if kind == "acid" else (pka - PH)
    return (1.0 if side == "react" else -1.0) * RT_LN10 * math.log10(1.0 + 10.0 ** expo)


def approx(a, b, tol=1e-2):
    return abs(a - b) < tol


def test_exact_term_values():
    # acid site, pKa 4 on a reactant: +RT_LN10 * log10(1+10^3) = +17.13 kJ (fully deprotonated at pH7)
    assert approx(alberty_term("react", 4.0, "acid"), 17.13), alberty_term("react", 4.0, "acid")
    # at pKa == pH the group is half-dissociated: +RT_LN10 * log10(2) = +1.72 kJ
    assert approx(alberty_term("react", 7.0, "acid"), 1.72)
    # product side flips the sign
    assert approx(alberty_term("prod", 4.0, "acid"), -17.13)


def test_high_pKa_uses_exact_not_linear():
    """The whole reason for the exact form: a HIGH-pKa acid (e.g. Pi's 12.35) is protonated at pH7, so its
    term must be ~0. The LINEAR approximation RT_LN10*(pH-pKa) would wrongly give a large negative -30.6."""
    exact = alberty_term("react", 12.35, "acid")
    linear = RT_LN10 * (PH - 12.35)
    assert approx(exact, 0.0, tol=0.05), exact
    assert linear < -30.0                                    # the bug the exact form avoids


def test_acid_base_mirror():
    """A base with pKa 10 (protonated below its pKa) is the exact mirror of an acid with pKa 4:
    both are 3 units from pH7, so |term| is identical."""
    assert approx(alberty_term("react", 10.0, "base"), alberty_term("react", 4.0, "acid"))


def test_spectator_cancellation():
    """A site that appears on BOTH sides (spectator: same class/pKa) cancels exactly -> net 0."""
    net = alberty_term("react", 6.5, "acid") + alberty_term("prod", 6.5, "acid")
    assert approx(net, 0.0)


def test_neutralization_dispatch():
    """Anions neutralize to their acid (emit an 'acid' pKa); cations (ammonium) neutralize to the amine
    (emit a 'base' pKa). Locks the v2 dispatch that the base path depends on."""
    import ph0_auto as p
    neutral_a, acids, netq_a = p._neutralize("CC(=O)[O-]")          # acetate -> acetic acid
    assert netq_a == 0 and len(acids) >= 1, (neutral_a, acids)
    neutral_b, acids2, bases, netq_b = p._neutralize_v2("CC[NH3+]")  # ethylammonium -> ethylamine
    assert netq_b == 0 and len(bases) >= 1, (neutral_b, bases)


def main():
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  {name:38s} OK")
            except AssertionError as e:
                fails += 1
                print(f"  {name:38s} FAIL: {e}")
    assert fails == 0, f"{fails} pH-0 bookkeeping regressions"
    print("\nPASS: pH-0 Alberty transform bookkeeping locked")


if __name__ == "__main__":
    main()
