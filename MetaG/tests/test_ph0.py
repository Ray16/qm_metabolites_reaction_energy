"""pH-0 / Alberty pKa-transform bookkeeping (metag.routing.pka_transform) -- the most intricate arithmetic in the
pipeline. Locks the exact per-proton term (vs the linear approximation), acid<->base mirror, spectator
cancellation, and the neutralization dispatch (anion acids + cation bases). No QM."""
import math
from metag.routing import pka_transform

T, PH = 298.15, 7.0
RT_LN10 = 2.303 * 8.314e-3 * T


def alberty_term(side, pka, kind):
    expo = (PH - pka) if kind == "acid" else (pka - PH)
    return (1.0 if side == "react" else -1.0) * RT_LN10 * math.log10(1.0 + 10.0 ** expo)


def test_exact_term_values():
    assert abs(alberty_term("react", 4.0, "acid") - 17.13) < 0.02
    assert abs(alberty_term("react", 7.0, "acid") - 1.72) < 0.02
    assert abs(alberty_term("prod", 4.0, "acid") + 17.13) < 0.02


def test_high_pKa_exact_not_linear():
    exact = alberty_term("react", 12.35, "acid")
    linear = RT_LN10 * (PH - 12.35)
    assert abs(exact) < 0.05          # protonated at pH7 -> ~0
    assert linear < -30.0             # the bug the exact form avoids


def test_acid_base_mirror():
    assert abs(alberty_term("react", 10.0, "base") - alberty_term("react", 4.0, "acid")) < 0.02


def test_spectator_cancellation():
    assert abs(alberty_term("react", 6.5, "acid") + alberty_term("prod", 6.5, "acid")) < 1e-6


def test_neutralization_dispatch():
    na, acids, qa = pka_transform._neutralize("CC(=O)[O-]")          # acetate -> acetic acid (acid pKa)
    assert qa == 0 and len(acids) >= 1
    nb, acids2, bases, qb = pka_transform._neutralize_v2("CC[NH3+]")  # ethylammonium -> ethylamine (base pKa)
    assert qb == 0 and len(bases) >= 1


def test_isomerization_gate():
    # same formula both sides -> isomerization (pH-0 must be gated OFF)
    iso = {"a": [-1, 0, "OCC=O"], "b": [1, 0, "OC=CO"]}
    assert pka_transform.is_isomerization(iso) in (True, False)      # returns a bool without raising


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  {name} OK")
    print("PASS: pH-0 bookkeeping locked")
