"""Routing changes under A/B stay OFF by default (validated behaviour) and do what they claim when enabled."""
import pytest
from rdkit import Chem

P = pytest.importorskip("metag.pipeline")
from metag.routing.policy import RoutingPolicy
from metag.routing.truncate import (build_truncated_reaction, has_anomeric_ring_reaction_center,
                                    truncation_radius)

# PPDK-type core: ATP + pyruvate + H2O -> AMP + PEP + Pi (nucleoside is the spectator)
_ATP = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])O)[C@@H](O)[C@H]1O"
_AMP = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])[O-])[C@@H](O)[C@H]1O"
PPDK = {"H2O": [-1, 0, "O"], "ATP": [-1, -3, _ATP], "Pyr": [-1, -1, "CC(=O)C(=O)[O-]"],
        "Pi": [1, -2, "O=P([O-])([O-])O"], "AMP": [1, -2, _AMP], "PEP": [1, -3, "C=C(OP(=O)([O-])[O-])C(=O)[O-]"]}


def test_ab_flags_default_off():
    for f in ("TRUNC_SPECTATOR_CATIONS", "TRUNC_MAXANION_RETRY"):
        assert P.FLAG_DEFAULTS[f] is False
    assert P.FLAG_DEFAULTS["TRUNC_FG_CUTS"] is True    # adopted: faithful C-C-only cuts (correctness)
    assert P.FLAG_DEFAULTS["TRUNC_ANOMERIC_RADIUS"] is True   # adopted: converged glycosidic core
    assert P.effective_config()["free_ppi_pka"] is True     # adopted 2026-10-01: measured free-PPi ladder
    assert P.effective_config()["anhydride_pka"] is False


def test_routing_policy_resolves_environment_once(monkeypatch):
    monkeypatch.setenv("AUTO_TRUNCATE", "0")
    policy = RoutingPolicy.from_runtime(P._flag, P.os.environ)
    monkeypatch.setenv("AUTO_TRUNCATE", "1")
    assert policy.auto_truncate is False
    assert RoutingPolicy.from_runtime(P._flag, P.os.environ).auto_truncate is True


def test_explicit_policy_and_radius_override_are_mutually_exclusive():
    policy = RoutingPolicy.from_runtime(P._flag, P.os.environ)
    with pytest.raises(ValueError, match="cannot be combined"):
        P.route_reaction(
            {"species": {"a": [-1, 0, "CC"], "b": [1, 0, "CC"]}},
            policy=policy,
            trunc_radius=3,
            log=lambda *_: None,
        )


def test_fg_cuts_never_cap_a_phosphoester_oxygen():
    old = build_truncated_reaction(PPDK)                    # default: may cut C5'-O and H-cap the O
    new = build_truncated_reaction(PPDK, fg_cuts=True)
    assert new is not None
    amp_new = new[0]["AMP_t"][2]
    m = Chem.MolFromSmiles(amp_new)
    # the AMP core keeps a C-O-P ester (methyl phosphate or larger), never becomes free H3PO4
    assert m.HasSubstructMatch(Chem.MolFromSmarts("[#6]-[OX2]-P")), amp_new
    assert old is None or old[0]["AMP_t"][2] != amp_new


# Hypoxanthine phosphoribosyltransferase direction as represented in TECRDB:
# PPi + GMP -> PRPP + guanine. Radius 2 drops GMP's conserved 5'-phosphate;
# radius 3 retains the complete ribose-phosphate environment and is converged
# because radius 4 produces the same cores.
_PPI = "O=P([O-])([O-])OP(=O)([O-])O"
_GMP = "Nc1nc2c(ncn2[C@@H]2O[C@H](COP(=O)([O-])[O-])[C@@H](O)[C@H]2O)c(=O)[nH]1"
_PRPP = "O=P([O-])([O-])OC[C@H]1O[C@H](OP(=O)([O-])OP(=O)([O-])O)[C@H](O)[C@@H]1O"
_GUANINE = "Nc1nc2[nH]cnc2c(=O)[nH]1"
PRT = {"PPi": [-1, -3, _PPI], "GMP": [-1, -2, _GMP],
       "PRPP": [1, -4, _PRPP], "Guanine": [1, 0, _GUANINE]}


def test_anomeric_reaction_uses_radius_three_and_is_structurally_converged():
    assert has_anomeric_ring_reaction_center(PRT)
    assert truncation_radius(PRT) == 3
    r2 = build_truncated_reaction(PRT, radius=2, fg_cuts=True)
    r3 = build_truncated_reaction(PRT, radius=3, fg_cuts=True)
    r4 = build_truncated_reaction(PRT, radius=4, fg_cuts=True)
    assert r2 is not None and r3 is not None and r4 is not None
    assert r2[0] != r3[0]
    assert r3 == r4
    assert r3[0]["GMP_t"][2] == Chem.MolToSmiles(Chem.MolFromSmiles(_GMP))


def test_pipeline_adopts_anomeric_radius_three(monkeypatch):
    monkeypatch.setenv("AUTO_TRUNCATE", "1")
    monkeypatch.setenv("ROUTE_FULL", "0")
    monkeypatch.delenv("TRUNC_RADIUS", raising=False)
    rx = {"species": PRT, "n_Hplus": -1, "note": "structural test"}
    routed, routes, truncated = P.route_reaction(rx, log=lambda *_: None)
    assert truncated and routes["trunc_radius"] == 3
    assert routes["trunc_radius_reason"] == "anomeric_ring_reaction_center"
    monkeypatch.setenv("TRUNC_ANOMERIC_RADIUS", "0")
    _, ablated, truncated = P.route_reaction(rx, log=lambda *_: None)
    assert truncated and ablated["trunc_radius"] == 2


def test_non_glycosidic_phosphate_transfer_keeps_default_radius():
    assert not has_anomeric_ring_reaction_center(PPDK)
    assert truncation_radius(PPDK) == 2
