"""Cycle-closure harness (metag.tools.cycle_closure): a consistent state function closes exactly, an
inconsistent one is detected and localized, protonation states map to one compound, and a network with
no cycles has zero dof."""
import numpy as np
from metag.tools.cycle_closure import closure_report, compound_key, cycle_basis, stoich_matrix

# A -> B, B -> C, A -> C form one cycle; D -> E is isolated.
A, B, C, D, E = "CCO", "CC=O", "CC(=O)O", "c1ccccc1", "Oc1ccccc1"
F = {A: -10.0, B: 5.0, C: -30.0, D: 0.0, E: -7.0}


def _rx(rid, r, p, shift=0.0, sigma=1.0):
    return {"rid": rid, "species": {"r": [-1, 0, r], "p": [1, 0, p]}, "dG": F[p] - F[r] + shift, "sigma": sigma}


def test_consistent_network_closes():
    rep = closure_report([_rx("ab", A, B), _rx("bc", B, C), _rx("ac", A, C), _rx("de", D, E)])
    assert rep["dof"] == 1 and rep["n_cycles"] == 1
    assert rep["chi2"] < 1e-8
    assert abs(rep["worst_cycles"][0]["closure_error"]) < 1e-8


def test_inconsistency_detected():
    rep = closure_report([_rx("ab", A, B), _rx("bc", B, C), _rx("ac", A, C, shift=12.0), _rx("de", D, E)])
    cyc = rep["worst_cycles"][0]
    assert abs(abs(cyc["closure_error"]) - 12.0) < 1e-6          # the whole 12 kJ shows up in the cycle
    assert set(cyc["reactions"]) == {"ab", "bc", "ac"}
    assert rep["chi2_per_dof"] > 10
    assert "de" not in {w["rid"] for w in rep["worst_reactions"]}  # isolated reaction carries no residual


def test_protonation_states_are_one_compound():
    assert compound_key("CC(=O)[O-]") == compound_key("CC(=O)O")
    assert compound_key("CC(=O)O") != compound_key("CC(=O)OC")


def test_no_cycles_zero_dof():
    rep = closure_report([_rx("ab", A, B), _rx("de", D, E)])
    assert rep["dof"] == 0 and rep["chi2_per_dof"] is None and rep["n_cycles"] == 0


def test_cycle_basis_integer_and_in_nullspace():
    recs = [_rx("ab", A, B), _rx("bc", B, C), _rx("ac", A, C)]
    S, _ = stoich_matrix(recs)
    basis = cycle_basis(S)
    assert len(basis) == 1
    v = np.zeros(3)
    for j, c in basis[0].items():
        assert isinstance(c, int)
        v[j] = c
    assert np.allclose(S @ v, 0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  {name} OK")
    print("PASS: cycle closure")
