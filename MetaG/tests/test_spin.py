"""Ground-state spin multiplicity for the UMA `spin` field (metag.energetics.conformers.spin_multiplicity).

Regression guard for a real bug: O2 is a TRIPLET ground state, but a bare `O=O` SMILES implies a
singlet. Computed as singlet, UMA puts O2 ~115 kJ/mol too high, driving every O2-consuming
oxygenase/oxidase reaction that-many-times too negative (error scaled linearly with O2 stoichiometry
on ModelSEED rxn00024/00054/00057). Pure logic, no GPU. Run: PYTHONPATH=. python tests/test_spin.py
"""
from metag.energetics.conformers import spin_multiplicity as sm


def test_spin_multiplicity():
    cases = {
        ("O=O", 0): 3,          # dioxygen -> triplet ground state (the bug)
        ("O", 0): 1,            # water -> singlet
        ("CC(=O)[O-]", -1): 1,  # acetate -> singlet
        ("O=C(O)O", 0): 1,      # carbonic acid -> singlet (even electrons)
        ("[O-][O]", -1): 2,     # superoxide -> doublet (odd electrons)
        ("[N]=O", 0): 2,        # nitric oxide -> doublet (odd electrons)
        ("Nc1ccccc1O", 0): 1,   # o-aminophenol (oxygenase substrate) -> singlet
    }
    for (smi, q), exp in cases.items():
        got = sm(smi, q)
        assert got == exp, f"{smi} q={q}: spin mult {got}, expected {exp}"
    print(f"OK: {len(cases)} spin-multiplicity cases (O2 triplet guarded)")


if __name__ == "__main__":
    test_spin_multiplicity()
