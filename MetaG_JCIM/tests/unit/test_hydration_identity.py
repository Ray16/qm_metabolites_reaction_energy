"""A molecular partition sum must not count equivalent hydration-site labels."""
import math
from unittest.mock import Mock
import pytest
from rdkit import Chem
from metag.routing import aldehyde_hydration as ah
from metag.energetics.aqueous_states import hydrate_all_sites


def fold(smiles, implicit):
    energies, sigmas, routes = {'x':0.}, {'x':0.}, {'warnings':[]}
    hydrate_all_sites(None, {}, 'x', 0, smiles, energies, sigmas, 0., (), 1, 1,
                      lambda _:None, routes, implicit=implicit,
                      water_reference=lambda _:0., acid_transform=lambda _:0., calibrate=False)
    return energies, routes


def test_equivalent_glyoxal_sites_are_one_molecular_state():
    result=ah.enumerate_hydration_states('O=CC=O')
    assert result.complete and result.n_subsets==3 and len(result.states)==2
    assert [n for n,_ in result.states]==[1,2]
    implicit=Mock(return_value=(0.,0.))
    energies,routes=fold('O=CC=O',implicit)
    assert implicit.call_count==2
    # Carbonyl + one unique monohydrate + one dihydrate, all isoenergetic.
    assert energies['x']==pytest.approx(-ah.RT*math.log(3))
    assert energies['x']!=-ah.RT*math.log(4)
    assert routes['hydration_diagnostics']['x']['status']=='complete'
    assert not routes.get('incomplete_hydration')


def test_non_equivalent_sites_remain_distinct_and_reordering_invariant():
    smi='CC(=O)C=O'
    m=Chem.MolFromSmiles(smi)
    reordered=Chem.MolToSmiles(Chem.RenumberAtoms(m,list(reversed(range(m.GetNumAtoms())))),canonical=False)
    a,b=map(ah.enumerate_hydration_states,[smi,reordered])
    assert a.states==b.states and len(a.states)==3
    assert a.complete and b.complete


def test_construction_failure_is_not_hidden_by_deduplication(monkeypatch):
    original=ah._hydrate
    monkeypatch.setattr(ah,'_hydrate',lambda m,pairs:None if len(pairs)==2 else original(m,pairs))
    enumeration=ah.enumerate_hydration_states('O=CC=O')
    assert enumeration.n_subsets==3 and len(enumeration.states)==1
    assert enumeration.construction_failures==1 and not enumeration.complete
    implicit=Mock(return_value=(0.,0.))
    energies,routes=fold('O=CC=O',implicit)
    implicit.assert_not_called()
    assert routes['incomplete_hydration']==['x'] and energies['x']==0.


@pytest.mark.parametrize('failed_energy',[None,float('nan'),float('inf')])
def test_hydration_retains_failed_minimum_diagnostics(failed_energy):
    def implicit(*args,diagnostics,**kwargs):
        diagnostics['thermal']={'reason':'unresolved sampled minimum'}
        return failed_energy,None
    energies,routes=fold('CC=O',implicit)
    state=routes['hydration_diagnostics']['x']['states'][0]
    assert state['status']=='no_estimate'
    assert state['species']['thermal']['reason']=='unresolved sampled minimum'
    assert routes['incomplete_hydration']==['x'] and energies['x']==0.
