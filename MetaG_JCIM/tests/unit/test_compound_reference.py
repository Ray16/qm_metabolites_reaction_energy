"""Compound reference bookkeeping must be independent of reaction context."""
from copy import deepcopy
import pytest

from metag.assembly import assemble_reaction
from metag.chem import is_balanced
from metag.routing.compound_reference import compound_reference, build_compound_reference_reaction
from metag.routing.planner import route_reaction
from metag.tools.audit_routes import audit_route_panel


@pytest.mark.parametrize('smiles,charge', [('CC(=O)O',0),('CC(=O)[O-]',-1),
    ('N',0),('[NH4+]',1),('C[N+](C)(C)C',1),('O',0),
    ('[NH3+]CC(=O)[O-]',0),('NCC(=O)O',0)])
def test_every_reference_change_is_only_proton_exchange(smiles,charge):
    row=compound_reference(smiles,charge)
    exchange={'input':[-1,charge,smiles],
              'reference':[1,row['reference_charge'],row['reference_smiles']]}
    assert is_balanced(exchange,-row['proton_uptake'])


def test_acid_and_base_drawings_resolve_same_references_and_terms():
    for left,qleft,right,qright in [('CC(=O)O',0,'CC(=O)[O-]',-1),
                                   ('N',0,'[NH4+]',1),
                                   ('NCC(=O)O',0,'[NH3+]CC(=O)[O-]',0)]:
        a=compound_reference(left,qleft);b=compound_reference(right,qright)
        for field in ('reference_smiles','reference_charge','acid_pkas','base_pkas'):
            assert a[field]==b[field]


def test_permanent_cations_are_retained_without_invented_base_constants():
    row=compound_reference('C[N+](C)(C)C',1)
    assert row['reference_charge']==1 and row['proton_uptake']==0
    assert row['base_pkas']==[]


def test_acid_identity_cancels_assembly_and_sampling():
    rx={'species':{'acid':[-1,0,'CC(=O)O'],'ion':[1,-1,'CC(=O)[O-]']},'n_Hplus':1}
    rewritten,models=build_compound_reference_reaction(rx)
    assert rewritten['n_Hplus']==0 and is_balanced(rewritten['species'])
    energy=assemble_reaction(rewritten['species'],{'acid':123.4,'ion':123.4},
        {'acid':2.,'ion':2.},n_hplus=0,proton_energy=-1100.,
        pka_sites=rewritten['pka_sites'],ph=7.,rt_ln10=5.708,
        sampling_groups={'acid':'same','ion':'same'})
    assert abs(energy.total)<1e-12 and energy.sampling_sigma==0


def test_reference_models_are_context_extent_and_reversal_independent(monkeypatch):
    monkeypatch.setenv('METAG_ROUTING_MODE','compound_reference')
    first={'species':{'ammonium':[-1,1,'[NH4+]'],'ammonia':[1,0,'N']},'n_Hplus':1}
    second={'species':{'ammonium':[2,1,'[NH4+]'],'ammonia':[-2,0,'N']},'n_Hplus':-2}
    a,ra,_=route_reaction(first,log=lambda _:None)
    b,rb,_=route_reaction(second,log=lambda _:None)
    assert ra['compound_references']['ammonium']==rb['compound_references']['ammonium']
    assert a['n_Hplus']==b['n_Hplus']==0
    assert is_balanced(a['species']) and is_balanced(b['species'])
    panel=audit_route_panel({'forward':first,'reverse_twice':second})
    assert panel['compound_reference_consistency']['passed']
    assert panel['panel_gate']['passed']


def test_user_unowned_terms_cannot_silently_enter_reference_mode():
    rx={'species':{'a':[-1,0,'CCO'],'b':[1,0,'CCO']},'pka_sites':[['prod',4.3]]}
    with pytest.raises(ValueError,match='ownership'):
        build_compound_reference_reaction(rx)


def test_unsupported_reference_is_not_replaced_with_legacy_route(monkeypatch):
    monkeypatch.setenv('METAG_ROUTING_MODE','compound_reference')
    rx={'species':{'a':[-1,0,'CP(=O)(O)O'],'b':[1,0,'CP(=O)(O)O']}}
    report=audit_route_panel({'unsupported':rx})
    row=report['reactions']['unsupported']
    assert row['routes']['compound_reference_status']=='failed'
    assert 'species_scored' not in row
    assert not report['panel_gate']['passed']
    assert report['compound_reference_consistency']['unavailable_reactions']==['unsupported']


def test_owned_model_audit_detects_term_and_proton_mutations(monkeypatch):
    monkeypatch.setenv('METAG_ROUTING_MODE','compound_reference')
    rx={'species':{'acid':[-1,0,'CC(=O)O'],'ion':[1,-1,'CC(=O)[O-]']},'n_Hplus':1}
    report=audit_route_panel({'acid_identity':rx})
    from metag.validation.compound_references import audit_compound_references
    altered=deepcopy(report['reactions'])
    altered['acid_identity']['pka_sites_scored'][0][1]+=1.
    audit=audit_compound_references({'acid_identity':rx},altered)
    assert not audit['passed'] and any('terms_differ' in e['reason'] for e in audit['errors'])
    altered=deepcopy(report['reactions'])
    altered['acid_identity']['n_Hplus_scored']+=1
    audit=audit_compound_references({'acid_identity':rx},altered)
    assert not audit['passed'] and any('bookkeeping_mismatch' in e['reason'] for e in audit['errors'])


def test_shared_structure_with_different_constants_is_detected(monkeypatch):
    monkeypatch.setenv('METAG_ROUTING_MODE','compound_reference')
    rx={'species':{'a':[-1,0,'N'],'b':[1,0,'N']}}
    report=audit_route_panel({'first':rx,'second':rx})
    from metag.validation.compound_references import audit_compound_references
    altered=deepcopy(report['reactions'])
    altered['second']['routes']['compound_references']['a']['base_pkas'][0]+=1
    audit=audit_compound_references({'first':rx,'second':rx},altered)
    assert audit['n_context_dependent_reference_models']==1 and not audit['passed']


def test_reference_builder_rejects_charge_and_element_imbalances():
    with pytest.raises(ValueError,match='charge/SMILES'):
        compound_reference('CC(=O)[O-]',0)
    with pytest.raises(ValueError,match='balanced input'):
        build_compound_reference_reaction({'species':{'a':[-1,0,'CCO'],'b':[1,0,'CO']}})


@pytest.mark.parametrize('value',[None,'unknown',True,float('nan')])
def test_unknown_proton_bookkeeping_is_an_audit_failure(monkeypatch,value):
    monkeypatch.setenv('METAG_ROUTING_MODE','compound_reference')
    rx={'species':{'a':[-1,0,'CCO'],'b':[1,0,'CCO']}}
    report=audit_route_panel({'identity':rx})
    from metag.validation.compound_references import audit_compound_references
    report['reactions']['identity']['n_Hplus_scored']=value
    audit=audit_compound_references({'identity':rx},report['reactions'])
    assert not audit['passed']
    assert any('bookkeeping_mismatch' in row['reason'] for row in audit['errors'])
