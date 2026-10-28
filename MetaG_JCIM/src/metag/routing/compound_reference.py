"""Reaction-independent compound references with explicit proton bookkeeping.

Uses the existing declared pKa provider, not new fitted constants. Consistent
references do not establish microscopic parameter accuracy or complete pools.
"""
from rdkit import Chem

from metag.chem import formula_charge, is_balanced, reaction_residual
from metag.routing import pka_transform as pk

POLICY_VERSION = "2026-10-11-compound-reference-v1"


def compound_reference(smiles, charge):
    """Choose a reference from this compound alone; retain its model scope."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or Chem.GetFormalCharge(molecule) != charge:
        raise ValueError("Compound input charge/SMILES mismatch")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    molecule = Chem.RemoveHs(molecule)
    canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
    reference, acids, bases, reference_charge = pk._neutralize_v2(canonical)
    if reference is None:
        raise ValueError("Compound protonation reference construction failed")
    original_formula, original_charge = formula_charge(canonical)
    reference_formula, actual_reference_charge = formula_charge(reference)
    if actual_reference_charge != reference_charge:
        raise ValueError("Reference charge mismatch")
    heavy_original = {key:value for key,value in original_formula.items() if key != 'H'}
    heavy_reference = {key:value for key,value in reference_formula.items() if key != 'H'}
    proton_uptake = reference_formula.get('H',0) - original_formula.get('H',0)
    if heavy_original != heavy_reference or reference_charge-original_charge != proton_uptake:
        raise ValueError("Reference change is not exclusively proton exchange")
    # The thermochemical reference selects recognized acid/base states only;
    # permanent ions remain charged and their reservoir terms stay explicit.
    return {'policy':POLICY_VERSION, 'input_smiles':canonical, 'input_charge':charge,
            'reference_smiles':reference, 'reference_charge':reference_charge,
            'proton_uptake':proton_uptake,
            'acid_pkas':list(acids), 'base_pkas':list(bases),
            'pka_provider':pk.pka_model_config(),
            'scope':'Existing independent-site reference model; class transfer and coupled/missing states remain unvalidated.'}


def build_compound_reference_reaction(reaction):
    """Assemble references consistently, preserving H+ and balance exactly."""
    if reaction.get('pka_sites'):
        raise ValueError("compound_reference mode requires per-compound reference ownership; flat user pka_sites are unsupported")
    if not is_balanced(reaction['species'], reaction.get('n_Hplus',0)):
        raise ValueError("compound_reference requires an element/charge-balanced input")
    scored, terms, provenance = {}, [], {}
    n_hplus = reaction.get('n_Hplus',0)
    for name,(coefficient,charge,smiles) in reaction['species'].items():
        model = compound_reference(smiles,charge)
        provenance[name] = model
        scored[name] = [coefficient,model['reference_charge'],model['reference_smiles']]
        n_hplus -= coefficient*model['proton_uptake']
        if coefficient == 0:
            continue
        side = 'react' if coefficient < 0 else 'prod'
        for kind,values in [('acid',model['acid_pkas']),('base',model['base_pkas'])]:
            terms.extend([side,pka,kind,abs(coefficient)] for pka in values)
    if not is_balanced(scored,n_hplus):
        residual,charge = reaction_residual(scored,n_hplus)
        raise ValueError(f"Compound-reference reaction does not balance: {residual}, charge={charge}")
    rewritten = dict(reaction,species=scored,n_Hplus=n_hplus,pka_sites=terms,explicit=False)
    return rewritten,provenance
