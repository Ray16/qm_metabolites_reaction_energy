"""Retained source chirality and safe rejection of ambiguous fragment models."""
import copy
import json
from pathlib import Path

import pytest
from rdkit import Chem

from metag.routing.cofactor_cores import _cysteine_core, _PEPTIDE_CYS, cofactor_ring
from metag.routing.planner import route_reaction
from metag.tools.preflight import audit_reactions
from metag.validation.stereochemistry import inspect_stereochemistry, unresolved_species

FIXTURE = Path(__file__).parents[1]/'regression/fixtures/routing_stereo_recovery.json'


def examples():
    return json.loads(FIXTURE.read_text())


def cys_labels(molecule):
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return sorted(molecule.GetAtomWithIdx(match[3]).GetProp('_CIPCode')
                  for match in molecule.GetSubstructMatches(_PEPTIDE_CYS))


@pytest.mark.parametrize('rid', ['rxn00070','rxn00086','rxn01336','rxn01363'])
def test_actual_resolved_inputs_remain_resolved_after_routing(rid):
    rx=examples()[rid];original=copy.deepcopy(rx)
    assert not unresolved_species(rx['species'])
    routed,routes,_=route_reaction(rx,log=lambda *_:None)
    assert routes['input_balanced'] and not routes['errors']
    assert not unresolved_species(routed['species'])
    assert rx==original
    assert audit_reactions({rid:rx})['reactions'][rid]['status']=='review'


@pytest.mark.parametrize('oxidized', [False,True])
def test_cysteine_core_copies_each_source_center(oxidized):
    rx=examples()['rxn00070']
    label='Oxidized glutathione' if oxidized else 'GSH'
    molecule=Chem.MolFromSmiles(rx['species'][label][2])
    core,charge=_cysteine_core(molecule,oxidized)
    assert charge==0
    assert cys_labels(Chem.MolFromSmiles(core))==cys_labels(molecule)
    assert inspect_stereochemistry(core)['status']=='no_unresolved_detected'
    reordered=Chem.RenumberAtoms(molecule,list(reversed(range(molecule.GetNumAtoms()))))
    assert _cysteine_core(reordered,oxidized)==(core,charge)


def test_an_inverted_source_center_produces_an_inverted_core():
    molecule=Chem.MolFromSmiles(examples()['rxn00070']['species']['GSH'][2])
    original=_cysteine_core(molecule,False)[0]
    index=molecule.GetSubstructMatches(_PEPTIDE_CYS)[0][3]
    molecule.GetAtomWithIdx(index).InvertChirality()
    inverted=_cysteine_core(molecule,False)[0]
    assert inverted!=original
    assert cys_labels(Chem.MolFromSmiles(inverted))!=cys_labels(Chem.MolFromSmiles(original))


def test_unknown_source_stereo_is_never_assigned_from_a_named_couple():
    molecule=Chem.MolFromSmiles(examples()['rxn00070']['species']['GSH'][2])
    Chem.RemoveStereochemistry(molecule)
    core,_=_cysteine_core(molecule,False)
    assert inspect_stereochemistry(core)['status']=='unresolved'
    rx=examples()['rxn00070'];rx['species']['GSH'][2]=Chem.MolToSmiles(molecule)
    assert audit_reactions({'r':rx})['reactions']['r']['status']=='blocked_unresolved_stereochemistry'


def test_mixed_disulfide_does_not_acquire_a_homodimer_core():
    rx=examples()['rxn00070'];gsh=rx['species']['GSH'][2]
    # Attach an ethylthio partner to the actual source sulfur without changing
    # any cysteinyl centre. The molecule has only one peptide cysteine unit.
    molecule=Chem.MolFromSmiles(gsh);sulfur=next(a.GetIdx() for a in molecule.GetAtoms() if a.GetAtomicNum()==16)
    edit=Chem.RWMol(molecule)
    added=edit.AddAtom(Chem.Atom(16));c1=edit.AddAtom(Chem.Atom(6));c2=edit.AddAtom(Chem.Atom(6))
    for a,b in [(sulfur,added),(added,c1),(c1,c2)]:edit.AddBond(a,b,Chem.BondType.SINGLE)
    edit.GetAtomWithIdx(sulfur).SetNumExplicitHs(0);edit.GetAtomWithIdx(sulfur).SetNoImplicit(False)
    mixed=edit.GetMol();Chem.SanitizeMol(mixed)
    assert _cysteine_core(mixed,True) is None
    species=copy.deepcopy(rx['species']);species['Oxidized glutathione'][2]=Chem.MolToSmiles(mixed)
    transformed=cofactor_ring(species)
    assert transformed['GSH'][2]==gsh
    assert transformed['Oxidized glutathione'][2]==species['Oxidized glutathione'][2]


@pytest.mark.parametrize('rid', ['rxn01336','rxn01363'])
def test_ambiguous_generic_cut_falls_back_while_retaining_other_cores(rid):
    rx=examples()[rid];routed,routes,truncated=route_reaction(rx,log=lambda *_:None)
    assert routes['stereochemistry_truncation_rejected']
    assert not truncated
    assert routes['cofactor_ring']
    assert not unresolved_species(routed['species'])
    # Nicotinamide stays reduced to its existing small core, so the fallback
    # does not unnecessarily reintroduce the complete NAD spectator.
    assert Chem.MolFromSmiles(routed['species']['NAD'][2]).GetNumHeavyAtoms()<15
