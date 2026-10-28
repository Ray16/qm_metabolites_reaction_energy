"""Structural tests for experimental ring-chain enumeration; not aqueous-energy validation."""
from rdkit import Chem

from metag.routing.ring_chain import has_ring_chain, ring_forms


def _formula(smi):
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    return CalcMolFormula(Chem.MolFromSmiles(smi))


def test_oxoglutaramate_gives_both_ring_configurations_as_isomers():
    for smi in ("NC(=O)CCC(=O)C(=O)O", "NC(=O)CCC(=O)C(=O)[O-]"):
        forms = ring_forms(smi)
        assert len(forms) == 2 and len(set(forms)) == 2              # both new-centre configurations
        for f in forms:
            assert _formula(f) == _formula(smi)                     # isomer: no water, no proton
            assert Chem.MolFromSmiles(f).GetRingInfo().NumRings() == 1
            assert Chem.GetFormalCharge(Chem.MolFromSmiles(f)) == Chem.GetFormalCharge(Chem.MolFromSmiles(smi))


def test_exclusions_outside_supported_structure_domain():
    for smi in ("N[C@@H](CCC(N)=O)C(=O)O",            # glutamine: amide but no keto/aldehyde
                "CC(=O)C(=O)O",                       # pyruvate: no amide
                "OC[C@H](O)[C@@H](O)C=O",             # open sugar: hemiacetal (O-H nucleophile) excluded
                "NC(=O)CC(=O)C(=O)O",                 # would be a 4-ring
                "CC/C=C\\CC1C(=O)CCC1CC(=O)NCC(=O)O"):  # ring ketone (fused bicycle) excluded
        assert not has_ring_chain(smi) and ring_forms(smi) == []
