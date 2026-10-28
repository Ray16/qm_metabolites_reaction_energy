"""Reaction-family detection is separate from anchor correction, and cannot steer routing."""
from metag.routing import anchor, reaction_families, truncation_gate

from test_anchor import (
    ADENYLYLATE, ADENYLYLATE_AMINOACID, CARBOXYP, CATIONIC_PHOSPHATASE, ISOMERIZATION, N_ADENYLATE,
    PHOSPHAGEN, PHOSPHATASE, THIOESTER,
)

CASES = [PHOSPHAGEN, THIOESTER, PHOSPHATASE, CATIONIC_PHOSPHATASE, ADENYLYLATE, ADENYLYLATE_AMINOACID,
         CARBOXYP, N_ADENYLATE, ISOMERIZATION]


def _reverse(species):
    return {k: [-c, q, s] for k, (c, q, s) in species.items()}


def test_families_module_is_the_single_detector():
    # anchor re-exports the detector; it must not keep a private copy that could drift
    assert anchor.subclass is reaction_families.subclass
    assert anchor.subclass_extent is reaction_families.subclass_extent
    for sp in CASES:
        sc, direction, extent = reaction_families.subclass_extent(sp)
        rsc, rdirection, rextent = reaction_families.subclass_extent(_reverse(sp))
        assert (rsc, rdirection, rextent) == (sc, -direction, extent)


def test_families_module_carries_no_offsets():
    assert not hasattr(reaction_families, "ANCHORS")
    assert not hasattr(reaction_families, "anchor_correct")


def _physics_gate_passes(monkeypatch):
    monkeypatch.setattr(truncation_gate, "_n_ion_change", lambda sp: 0)
    monkeypatch.setattr(truncation_gate.pfa, "is_isomerization", lambda sp: False)
    monkeypatch.setattr(truncation_gate, "_rc_in_ring", lambda sp: True)
    monkeypatch.setattr(truncation_gate, "_any_floppy_linker", lambda sp: False)


def test_prefer_full_ignores_family_when_anchors_off(monkeypatch):
    _physics_gate_passes(monkeypatch)

    def forbidden(_):
        raise AssertionError("family detection consulted for routing with anchors off")

    monkeypatch.setattr(reaction_families, "subclass", forbidden)
    assert truncation_gate.prefer_full(PHOSPHAGEN) is True


def test_prefer_full_keeps_calibrated_route_when_anchors_on(monkeypatch):
    _physics_gate_passes(monkeypatch)
    assert truncation_gate.prefer_full(PHOSPHAGEN, anchor_correct=True) is False
    assert truncation_gate.prefer_full({"x": [-1, 0, "CCO"], "y": [1, 0, "CC=O"]}, anchor_correct=True)
