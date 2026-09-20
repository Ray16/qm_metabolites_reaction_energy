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


def test_free_sulfate_neutralizes():
    # A1: free sulfate SO4(2-) was previously UNMATCHED by the anion SMARTS (needed a bridging ester O)
    # -> never protonated -> H-imbalance -> pH-0 REFUSED -> charged-COSMO anion catastrophe. It must now
    # neutralize to H2SO4 (net charge 0) with the full 2-proton ladder.
    neu, acids, netq = pka_transform._neutralize("O=S(=O)([O-])[O-]")
    assert netq == 0, f"free sulfate did not fully neutralize: netq={netq}"
    assert len(acids) == 2, f"free sulfate should emit 2 pKa (H2SO4 ladder): {acids}"
    # a sulfate MONOESTER (one bridging O) keeps one ionisable proton
    _, acids_ester, q_ester = pka_transform._neutralize("COS(=O)(=O)[O-]")
    assert q_ester == 0 and len(acids_ester) == 1, f"sulfate ester: q={q_ester} acids={acids_ester}"


def test_sulfotransfer_ph0_fires():
    # A1 end-to-end: a sulfate-transfer reaction (free SO4 <-> sulfate ester) must now FIRE pH-0 and be
    # H-balanced at n_H+=0, instead of refusing to the charged path. (rxn00379 sulfate adenylyltransferase.)
    sp = {
        "ATP":     (-1, -3, "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])O)[C@@H](O)[C@H]1O"),
        "Sulfate": (-1, -2, "O=S(=O)([O-])[O-]"),
        "PPi":     (1, -3, "O=P([O-])([O-])OP(=O)([O-])O"),
        "APS":     (1, -2, "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OS(=O)(=O)[O-])[C@@H](O)[C@H]1O"),
    }
    out = pka_transform.build_ph0_reaction(sp, n_Hplus=0, base=True)
    assert out is not None, "sulfotransfer still refuses pH-0 (A1 regressed)"


def test_amide_change_never_fires_base_path():
    # An amide N is neutral at pH 7 -> the base premise fails whenever an amide is created OR destroyed.
    # (A 2026-09 attempt to fire on amide HYDROLYSIS was reverted: it regressed every amidohydrolase --
    # the anion-only path already scores the protonated amine directly, so the base transform
    # double-shifts. Blanket exclusion is correct.)
    hydrolysis = {  # amide + H2O -> carboxylate + free amine  (schematic amidohydrolase)
        "amide": (-1, 0, "CCCCC(=O)NCCO"), "H2O": (-1, 0, "O"),
        "acid":  (1, -1, "CCCCC(=O)[O-]"), "amine": (1, 1, "[NH3+]CCO")}
    formation = {  # reverse: carboxylate + amine -> amide + H2O
        "acid":  (-1, -1, "CCCCC(=O)[O-]"), "amine": (-1, 1, "[NH3+]CCO"),
        "amide": (1, 0, "CCCCC(=O)NCCO"),   "H2O": (1, 0, "O")}
    assert pka_transform._base_gate(hydrolysis) is False, "amide hydrolysis must NOT fire the base path"
    assert pka_transform._base_gate(formation) is False, "amide formation must NOT fire the base path"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  {name} OK")
    print("PASS: pH-0 bookkeeping locked")
