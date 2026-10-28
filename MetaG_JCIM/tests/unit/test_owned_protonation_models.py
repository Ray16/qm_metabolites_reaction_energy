"""Owned coupled corrections replace site approximations without double counting."""
import copy
import math

import pytest

from metag.energetics.protonation_models import prepare_models, validate_model
from metag.energetics.protonation import reaction_transform
from metag.routing.planner import route_reaction
from metag.reactions import validate_reaction

CONDITIONS = {'T_K':298.15,'pH':7.,'ionic_strength_M':0.,'pMg':None,'water_activity':1.}
CITRATE = 'O=C(O)CC(O)(CC(=O)O)C(=O)O'
RTLN10 = 2.303*8.314e-3*298.15


def model(smiles=CITRATE, formula='C6H8O7', pkas=None):
    # Synthetic provenance fixture tests validation, not a measured data claim.
    return {'kind':'sequential_macroscopic_acid_pka', 'reference_smiles':smiles,
            'reference_deprotonations':0,'reference_scope':'fully_protonated_carboxylic_acid_family',
            'reference_formula':formula,'pKas':pkas or [3.12927,4.76227,6.39727],
            'conditions':{'T_K':298.15,'ionic_strength_M':0.,'solvent':'water',
                          'standard_state':'solute_1M_activity','magnesium_binding':'absent'},
            'source':{'locator':'synthetic test source','location':'test fixture','artifact_sha256':'0'*64,
                      'determination':'evaluated','parameter_scale':'molarity_activity',
                      'conversion':'explicit synthetic target convention','uncertainty':[.003]*3,
                      'uncertainty_kind':'approximate assessment, not SD','covariance':None,
                      'development_exposure':True}}


def reaction():
    return {'species':{'Citrate':[-1,-3,'O=C([O-])CC(O)(CC(=O)[O-])C(=O)[O-]'],
                       'Acetate':[1,-1,'CC(=O)[O-]'],
                       'Oxaloacetate':[1,-2,'O=C([O-])CC(=O)C(=O)[O-]']},'n_Hplus':0}


def prepared(rx=None, selected=None, conditions=None):
    routed,routes,_ = route_reaction(rx or reaction(),allow_truncate=False,log=lambda *_:None)
    remaining,coupled,diagnostic = prepare_models(selected or {'Citrate':model()}, routed['species'],
                    routed['pka_sites'],routes['protonation_sites_by_species'],conditions or CONDITIONS,rt_ln10=RTLN10)
    return remaining,coupled,diagnostic,routed,routes


def test_source_replacement_has_the_declared_shift_not_an_added_offset():
    remaining,coupled,diagnostic,routed,_ = prepared()
    old = reaction_transform(routed['pka_sites'],7.,RTLN10)
    new = reaction_transform(remaining,7.,RTLN10)+coupled
    assert new-old == pytest.approx(-5.118878243380429,abs=1e-10)
    assert len(remaining)==3
    assert len(diagnostic['species']['Citrate']['removed_independent_terms'])==3


def test_sequential_ladder_is_not_the_independent_site_product():
    m=model('O=C(O)CCC(=O)O','C4H6O4',[5.,5.]);m['source']['uncertainty']=[.003,.003]
    species={'acid':[-1,0,m['reference_smiles']]};owned={'acid':[['react',4.75,'acid',1]]*2}
    _,coupled,_=prepare_models({'acid':m},species,owned['acid'],owned,CONDITIONS,rt_ln10=RTLN10)
    assert coupled == pytest.approx(RTLN10*math.log10(1+100+10000),abs=1e-12)
    assert abs(coupled-RTLN10*2*math.log10(101))>.01


@pytest.mark.parametrize('scale',[2.,.5])
def test_stoichiometric_scaling_and_ownership(scale):
    r=reaction();r['species']={name:[coefficient*scale,charge,smiles] for name,(coefficient,charge,smiles) in r['species'].items()}
    remaining,coupled,_,_,_=prepared(r)
    base_remaining,base_coupled,_,_,_=prepared()
    assert coupled == pytest.approx(scale*base_coupled,abs=1e-12)
    assert reaction_transform(remaining,7.,RTLN10)==pytest.approx(scale*reaction_transform(base_remaining,7.,RTLN10),abs=1e-12)


def test_reversing_reaction_reverses_correction():
    r=reaction();r['species']={name:[-coefficient,charge,smiles] for name,(coefficient,charge,smiles) in r['species'].items()}
    remainder,coupled,*_=prepared(r);base,base_coupled,*_=prepared()
    assert reaction_transform(remainder,7.,RTLN10)+coupled == pytest.approx(
            -reaction_transform(base,7.,RTLN10)-base_coupled,abs=1e-12)


def test_missing_owned_site_cannot_be_silently_replaced():
    _,_,_,routed,routes=prepared()
    with pytest.raises(ValueError,match='absent'):
        prepare_models({'Citrate':model()},routed['species'],[],routes['protonation_sites_by_species'],CONDITIONS,rt_ln10=RTLN10)


def test_declared_conditions_and_reference_identity_are_enforced():
    m=model();m['conditions']['ionic_strength_M']=.1
    with pytest.raises(ValueError,match='condition mismatch'):
        validate_model(m,CITRATE,0,CONDITIONS)
    with pytest.raises(ValueError,match='match the scored'):
        validate_model(model(),'O=C(O)CC(C(=O)O)C(O)C(=O)O',0,CONDITIONS)


@pytest.mark.parametrize('field,value',[('reference_deprotonations',1),('reference_formula','C6H6O7'),
                                        ('pKas',[3.1,4.7]),('pKas',[3.1,float('nan'),6.4])])
def test_incomplete_or_intermediate_reference_refused(field,value):
    m=model();m[field]=value
    with pytest.raises(ValueError):validate_model(m,CITRATE,0,CONDITIONS)


def test_oxaloacetate_requires_its_coupled_carbonyl_states():
    m=model('O=C(O)CC(=O)C(=O)O','C4H4O5',[2.,4.]);m['source']['uncertainty']=[.003,.003]
    with pytest.raises(ValueError,match='carbonyl/tautomer'):
        validate_model(m,m['reference_smiles'],0,CONDITIONS)


@pytest.mark.parametrize('smiles,formula,pkas', [('OC(=O)O','CH2O3',[3.6,10.3]),
                                              ('O=C(O)CC(O)(O)C(=O)O','C4H6O6',[2.,4.])])
def test_carbonate_and_gem_diol_do_not_acquire_an_unowned_macro_anchor(smiles,formula,pkas):
    m=model(smiles,formula,pkas);m['source']['uncertainty']=[.003]*len(pkas)
    with pytest.raises(ValueError):validate_model(m,smiles,0,CONDITIONS)


def test_model_fingerprint_changes_and_input_is_not_mutated():
    source=model();original=copy.deepcopy(source)
    first=prepared(selected={'Citrate':source})[2]['models_sha256']
    assert source==original
    source['pKas'][0]+=.01
    assert prepared(selected={'Citrate':source})[2]['models_sha256']!=first


def test_coupled_pH_derivative_matches_perturbed_polynomial():
    step=1e-5;hi=dict(CONDITIONS,pH=7.+step);lo=dict(CONDITIONS,pH=7.-step)
    numerical=(prepared(conditions=hi)[1]-prepared(conditions=lo)[1])/(2*step)
    assert numerical==pytest.approx(prepared()[2]['reaction_dG_d_pH_kj_per_unit'],abs=1e-7)


def test_flat_user_terms_cannot_supply_invented_ownership():
    rx=reaction();rx.update(pka_sites=[['react',4.75]],species_protonation_models={'Citrate':model()})
    with pytest.raises(ValueError,match='no species ownership'):validate_reaction(rx)


def test_bad_source_model_is_rejected_before_any_qm(monkeypatch):
    from metag import pipeline as p
    def forbidden(*args,**kwargs):raise AssertionError('QM must not start')
    monkeypatch.setattr(p,'_species_free_energies',forbidden)
    rx=reaction();m=model();m['conditions']['T_K']=310.
    rx['species_protonation_models']={'Citrate':m}
    result=p.score_reaction(None,rx,allow_truncate=False,log=lambda *_:None)
    assert result['dG'] is None and 'condition mismatch' in result['suspect']


def test_public_score_replaces_terms_and_keeps_uncertainty_unavailable(monkeypatch):
    from metag import pipeline as p
    from rdkit import Chem
    values = {'CC(=O)O':-601456.8486480737,
              'O=C(O)CC(=O)C(=O)O':-1394208.6893425223,
              CITRATE:-1995679.6093449942}
    energies = {Chem.MolToSmiles(Chem.MolFromSmiles(smiles)):value for smiles,value in values.items()}
    def species_values(predictor, rx, routes, G, sigma, standard, *args):
        for name,(_,_,smiles) in rx['species'].items():
            G[name] = energies[Chem.MolToSmiles(Chem.MolFromSmiles(smiles))]+standard
            sigma[name] = .1
        routes['species_diagnostics'] = {name:{'method':'implicit','source':'test_fixture'} for name in G}
    monkeypatch.setattr(p,'_species_free_energies',species_values)
    monkeypatch.setenv('SOURCE_PKA_REGISTRY', '0')
    baseline = p.score_reaction(None,reaction(),allow_truncate=False,log=lambda *_:None)
    monkeypatch.setenv('SOURCE_PKA_REGISTRY', '1')
    automatic = p.score_reaction(None,reaction(),allow_truncate=False,log=lambda *_:None)
    rx=reaction();rx['species_protonation_models']={'Citrate':model()}
    candidate = p.score_reaction(None,rx,allow_truncate=False,log=lambda *_:None)
    assert baseline['dG']==10.6 and candidate['dG']==5.5
    assert automatic['dG'] == candidate['dG']
    assert automatic['routes']['source_protonation_registry']['automatic_species'] == ['Citrate']
    assert candidate['stages']['pka_transform']==pytest.approx(-16.53)
    diagnostic=candidate['routes']['protonation_diagnostics']
    assert diagnostic['correction_kj']==pytest.approx(-16.534770843397708)
    assert len(diagnostic['terms'])==3
    assert candidate['config']['species_protonation_models_sha256']
    assert candidate['sigma_pred'] is None and candidate['ci95']==[None,None]


def identity_reaction(*, sourced=True):
    rx = {'species': {'left': [-1, 0, CITRATE], 'right': [1, 0, CITRATE]}, 'n_Hplus': 0}
    if sourced:
        rx['species_protonation_models'] = {'left': model(), 'right': model()}
    return rx


def test_one_sided_source_model_cannot_create_energy_in_an_identity_reaction(monkeypatch):
    from metag import pipeline as p

    def forbidden(*args, **kwargs):
        raise AssertionError('QM must not start for inconsistent identical states')

    monkeypatch.setattr(p, '_species_free_energies', forbidden)
    rx = identity_reaction()
    del rx['species_protonation_models']['right']
    result = p.score_reaction(None, rx, allow_truncate=False, log=lambda *_: None)
    assert result['dG'] is None
    assert 'inconsistent protonation models' in result['suspect']


def test_consistent_source_models_cancel_in_an_identity_reaction():
    remainder, coupled, _, _, _ = prepared(identity_reaction(), {'left': model(), 'right': model()})
    assert coupled + reaction_transform(remainder, 7., RTLN10) == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize('mutation', ['temperature', 'structure', 'incomplete_ladder'])
def test_cpu_preflight_checks_the_same_source_contract_as_scoring(mutation):
    from metag.tools.preflight import audit_reactions
    rx = reaction()
    m = model()
    if mutation == 'temperature':
        m['conditions']['T_K'] = 310.
    elif mutation == 'structure':
        m['reference_smiles'] = 'CC(=O)O'
    else:
        m['pKas'] = m['pKas'][:-1]
    rx['species_protonation_models'] = {'Citrate': m}
    row = audit_reactions({'r': rx})['reactions']['r']
    assert row['status'] == 'blocked_source_protonation_model'
    assert row['error']


def test_preflight_reports_the_actual_effective_terms_and_coupled_correction():
    from metag.tools.preflight import audit_reactions
    rx = reaction()
    rx['species_protonation_models'] = {'Citrate': model()}
    row = audit_reactions({'r': rx})['reactions']['r']
    remaining, coupled, _, _, _ = prepared()
    assert row['pka_sites_scored'] == remaining
    assert row['coupled_protonation_kj'] == coupled
    assert row['protonation_reference_models']['Citrate']['assignment']['kind'] == 'sequential_macroscopic_acid_pka'


def test_source_model_errors_do_not_hide_the_prior_stereochemical_exclusion():
    from metag.tools.preflight import audit_reactions
    rx = {'species': {
        'Citrate': [-1, -3, 'O=C([O-])CC(O)(CC(=O)[O-])C(=O)[O-]'],
        'Isocitrate': [1, -3, 'O=C([O-])CC(C(=O)[O-])C(O)C(=O)[O-]']},
        'species_protonation_models': {'Citrate': model()}}
    rx['species_protonation_models']['Citrate']['conditions']['T_K'] = 310.
    row = audit_reactions({'r': rx})['reactions']['r']
    assert row['status'] == 'blocked_unresolved_stereochemistry'
    assert row['stereochemistry']['input']


def test_panel_rejects_different_protonation_assignments_on_the_same_state():
    from metag.tools.audit_routes import audit_route_panel
    report = audit_route_panel({'source': identity_reaction(), 'default': identity_reaction(sourced=False)})
    assert not report['panel_gate']['passed']
    assert report['representation_consistency']['consistent_known_representations']
    assert len(report['protonation_model_consistency']['conflicts']) == 1


def test_panel_model_identity_is_independent_of_labels_direction_and_extent():
    from metag.tools.audit_routes import audit_route_panel
    rx = identity_reaction()
    scaled = {'species': {'x': [2, 0, CITRATE], 'y': [-2, 0, CITRATE]},
              'species_protonation_models': {'x': model(), 'y': model()}}
    report = audit_route_panel({'r1': rx, 'r2': scaled})
    assert report['panel_gate']['passed']
    assert report['protonation_model_consistency']['complete_ownership']


def test_compound_reference_panel_accepts_consistent_sourced_ladders(monkeypatch):
    from metag.tools.audit_routes import audit_route_panel
    monkeypatch.setenv('METAG_ROUTING_MODE', 'compound_reference')
    report = audit_route_panel({'r': identity_reaction()})
    assert report['compound_reference_consistency']['passed']
    assert report['panel_gate']['passed']


def test_cli_rejects_invalid_source_model_before_loading_backend(tmp_path):
    from metag.cli import run_cli
    from unittest.mock import Mock
    rx = identity_reaction()
    rx['species_protonation_models']['left']['conditions']['T_K'] = 310.
    loader = Mock()
    with pytest.raises(SystemExit, match='preflight failed'):
        run_cli(reactions={'r': rx}, model_name='unused', load_model=loader,
                score=Mock(), no_estimate=Mock(), out=str(tmp_path),
                argv=['--require-consistent-panel'])
    loader.assert_not_called()


@pytest.mark.parametrize('reference', [0, 1, 3])
def test_sequential_parameter_derivatives_include_reference_macrostate(reference):
    from metag.energetics.protonation import macroscopic_transform
    pkas = [3.1, 4.8, 6.4]
    def transform(values, ph=5.):
        return macroscopic_transform(values, ph, RTLN10, reference_deprotonations=reference)
    result = transform(pkas)
    h = 1e-5
    for index, derivative in enumerate(result['dG_d_pKa']):
        high, low = pkas.copy(), pkas.copy()
        high[index] += h
        low[index] -= h
        central = (transform(high)['G_transform'] - transform(low)['G_transform']) / (2 * h)
        assert derivative == pytest.approx(central, abs=1e-8)
    central_ph = (transform(pkas, 5. + h)['G_transform'] - transform(pkas, 5. - h)['G_transform']) / (2 * h)
    assert result['dG_d_pH'] == pytest.approx(central_ph, abs=1e-8)
    assert result['dG_d_pH'] == pytest.approx(-math.fsum(result['dG_d_pKa']), abs=1e-12)


def test_source_parameter_sensitivities_follow_reaction_direction_and_extent():
    base = prepared()[2]['species']['Citrate']['reaction_dG_d_pKa_kj_per_unit']
    rx = reaction()
    rx['species'] = {name: [-2*c, q, smi] for name, (c, q, smi) in rx['species'].items()}
    other = prepared(rx)[2]['species']['Citrate']['reaction_dG_d_pKa_kj_per_unit']
    assert other == pytest.approx([-2*value for value in base])
