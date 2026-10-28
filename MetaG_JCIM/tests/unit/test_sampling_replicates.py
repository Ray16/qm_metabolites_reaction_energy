"""Replicate sensitivity requires complete, chemically identical producer records."""
from copy import deepcopy
import json,math
import pytest
from metag.validation.sampling_replicates import audit_sampling_replicates,PASS,R_KJ


def replica(seed,dg=-10.,sigma=.2,**overrides):
    species={'a':[-1,0,'CCO'],'b':[1,0,'COC']}
    conditions={'T_K':298.15,'pH':7.,'ionic_strength_M':0.,'pMg':None,'water_activity':1.,'standard_state':'solutes 1 M; pure liquid water'}
    producer={'schema':1,'source_sha256':'a'*64,'cacheable':True,'xtb_sha256':'b'*64,
      'software':{k:'1.0' for k in ('fairchem-core','torch','rdkit','ase','numpy')},'additional_executable_sha256':{},
      'model':{k:'c'*64 for k in ('checkpoint_sha256','references_sha256','inference_sha256')}}
    result={'dG':dg,'U_samp':sigma,'suspect':None,'original_species':deepcopy(species),'species_scored':deepcopy(species),
      'conditions':conditions,'energy_units':'kJ/mol','quantity':'standard_transformed_reaction_gibbs','reported_rounding_kj':.1,'explicit':False,
      'config':{'conformer_seed_offset':seed,'producer':producer,'physics':'test','conv':[2.5,2,8],'model':'same-model','solv_model':'alpb','conditions':json.dumps(conditions,sort_keys=True)},
      'routes':{'truncated':False,'ph0':False,'cofactor_ring':False,'coa_core':False,'ntp_core':False,'input_balanced':True,'rearrangement_reroute':False,
        'errors':[],'warnings':[],'species_diagnostics':{key:{'method':'implicit','sampling':{'status':'available','seed_offset':seed,'batches_attempted':2}} for key in species}},
      'stages':{'proton':0.,'pka_transform':0.,'species_sum':dg}}
    record={'reaction_id':'r','seed_offset':seed,'status':'estimated','result':result};record.update(overrides);return record


def audit(rows,expected=None,fold=2):
    return audit_sampling_replicates(rows,{'r':[0,1000]} if expected is None else expected,maximum_equilibrium_fold_change=fold)


def test_stable_range_maps_to_equilibrium_ratio_without_averaging_or_ci():
    report=audit([replica(0,-10),replica(1000,-9.5)]);g=report['reactions']['r'];m=g['metrics']
    assert g['status']==PASS and report['summary']['all_within_observed_replicate_tolerance']
    assert m['point_range_kj']==.5 and m['range_over_RT']==pytest.approx(.5/(R_KJ*298.15))
    assert m['equivalent_equilibrium_constant_ratio']==pytest.approx(math.exp(.5/(R_KJ*298.15)))
    assert m['tolerance_kj']==pytest.approx(R_KJ*298.15*math.log(2))
    assert m['pairwise_sampling_comparisons'][0]['difference_over_quadrature_U_samp']==pytest.approx(.5/math.hypot(.2,.2))
    assert 'mean' not in m and 'ci95' not in m


def test_unstable_observed_difference_exceeds_tolerance():
    g=audit([replica(0,-10),replica(1000,-6)])['reactions']['r']
    assert g['comparable'] and g['status']=='exceeds_observed_replicate_tolerance'
    assert not g['within_observed_replicate_tolerance'] and g['metrics']['equivalent_equilibrium_constant_ratio']>2


@pytest.mark.parametrize('failure',['budget_exceeded','no_estimate','failed'])
def test_failure_is_preserved_not_averaged_away(failure):
    g=audit([replica(0),replica(1000,status=failure,result=None,error='specific failure')])['reactions']['r']
    assert g['status']=='incomplete_replicate_set' and g['metrics'] is None
    assert len(g['records'])==2 and g['records'][1]['failure_details']['envelope']['error']=='specific failure'


def test_missing_expected_seed_remains_missing():
    g=audit([replica(0),replica(1000)],{'r':[0,1000,2000]})['reactions']['r']
    assert g['status']=='incomplete_replicate_set' and g['missing_replicates']==[2000]
    assert g['expected_replicates']==[0,1000,2000] and g['metrics'] is None


@pytest.mark.parametrize('change',['condition','weight','source','software','routing','scored','stoichiometry','reversal','units','quantity','proton','pka','method','nested_seed_setting','rounding'])
def test_scientific_changes_are_not_pure_sampling_replicas(change):
    first,second=replica(0),replica(1000);r=second['result']
    if change=='condition':r['conditions']['pH']=8.;r['config']['conditions']=json.dumps(r['conditions'],sort_keys=True)
    if change=='weight':r['config']['producer']['model']['checkpoint_sha256']='d'*64
    if change=='source':r['config']['producer']['source_sha256']='e'*64
    if change=='software':r['config']['producer']['software']['numpy']='2.0'
    if change=='routing':r['routes']['rearrangement_reroute']=True
    if change=='scored':r['species_scored']['a'][2]='COC'
    if change in ('stoichiometry','reversal'):
        for sp in (r['original_species'],r['species_scored']):
            for v in sp.values():v[0]*=2 if change=='stoichiometry' else -1
    if change=='units':r['energy_units']='kcal/mol'
    if change=='quantity':r['quantity']='standard_formation_gibbs'
    if change=='proton':r['stages']['proton']=5.
    if change=='pka':r['stages']['pka_transform']=5.
    if change=='method':r['routes']['species_diagnostics']['a']['method']='explicit'
    if change=='nested_seed_setting':r['config']['implicit_settings']='{"seed_offset":1000}'
    if change=='rounding':r['reported_rounding_kj']=1.
    g=audit([first,second])['reactions']['r'];assert not g['comparable'] and g['metrics'] is None and g['status']!=PASS


def test_sampling_diagnostics_may_vary():
    one,two=replica(0),replica(1000)
    two['result']['routes']['species_diagnostics']['a']['sampling']['batches_attempted']=8
    two['result']['routes']['warnings']=['last random batch hit its declared budget']
    assert audit([one,two])['reactions']['r']['status']==PASS


def test_duplicate_record_is_preserved_as_conflict():
    g=audit([replica(0),replica(0),replica(1000)])['reactions']['r']
    assert g['status']=='replicate_identity_conflict' and g['duplicate_replicates']==[0]
    assert len(g['records'])==3 and g['metrics'] is None


def test_labels_cannot_hide_duplicate_effective_seeds():
    g=audit([replica(0,replicate_id='a'),replica(0,replicate_id='b')],{'r':['a','b']})['reactions']['r']
    assert g['duplicate_seed_offsets']==[0] and g['status']=='replicate_identity_conflict'


def test_envelope_seed_must_match_configuration():
    one,two=replica(0),replica(1000);two['seed_offset']=2000
    g=audit([one,two],{'r':[0,2000]})['reactions']['r']
    assert g['status']=='unverifiable_replicate_set'
    assert 'envelope_seed_disagrees_with_configuration' in g['records'][1]['metadata_issues']


@pytest.mark.parametrize('missing',['producer','weights','source','software','executable','unidentified','full_source_v2'])
def test_missing_provenance_cannot_certify_comparability(missing):
    rows=[replica(0),replica(1000)]
    for row in rows:
        config=row['result']['config'];p=config['producer']
        if missing=='producer':del config['producer']
        if missing=='weights':p['model']=None
        if missing=='source':p['source_sha256']=None
        if missing=='software':p['software']['ase']=None
        if missing=='executable':p['xtb_sha256']=None
        if missing=='unidentified':p['cacheable']=False
        if missing=='full_source_v2':p.update(schema=2,source_scope='raw_species_v1')
    g=audit(rows)['reactions']['r'];assert g['status']=='unverifiable_replicate_set' and not g['comparable']


def test_complete_schema_two_is_supported():
    rows=[replica(0),replica(1000)]
    for row in rows:
        row['result']['config']['producer'].update(schema=2,source_scope='raw_species_v1')
        row['result']['config']['estimator_source_sha256']='f'*64
    assert audit(rows)['reactions']['r']['status']==PASS


@pytest.mark.parametrize('difference',[0.,4.])
def test_zero_sigmas_never_generate_infinite_json(difference):
    report=audit([replica(0,0,sigma=0),replica(1000,difference,sigma=0)])
    pair=report['reactions']['r']['metrics']['pairwise_sampling_comparisons'][0]
    assert pair['quadrature_U_samp_kj']==0
    assert pair['difference_over_quadrature_U_samp']==(0 if difference==0 else None)
    json.dumps(report,allow_nan=False)


def test_equilibrium_ratio_overflow_is_explicit():
    report=audit([replica(0,0),replica(1000,10000)]);m=report['reactions']['r']['metrics']
    assert m['equivalent_equilibrium_constant_ratio'] is None and m['equilibrium_ratio_overflow']
    assert math.isfinite(m['log10_equilibrium_constant_ratio']);json.dumps(report,allow_nan=False)


def test_missing_tolerance_and_single_replica_cannot_pass():
    assert audit([replica(0),replica(1000)],fold=None)['reactions']['r']['status']=='descriptive_only'
    assert audit([replica(0)],{'r':[0]})['reactions']['r']['status']=='insufficient_replicates'


def test_unidentified_failure_blocks_whole_audit_and_is_retained():
    report=audit([replica(0),replica(1000),{'status':'budget_exceeded','result':None,'error':'identity absent'}])
    assert not report['summary']['all_within_observed_replicate_tolerance'] and len(report['unassigned_records'])==1


def test_equivalent_smiles_and_immutable_inputs():
    rows=[replica(0),replica(1000)];rows[1]['result']['species_scored']['a'][2]='OCC'
    original=deepcopy(rows);expected={'r':[0,1000]};before=deepcopy(expected)
    assert audit(rows,expected)['reactions']['r']['status']==PASS
    assert rows==original and expected==before


@pytest.mark.parametrize('fold',[0,.9,-1,float('nan'),float('inf'),True])
def test_invalid_tolerance_is_rejected(fold):
    with pytest.raises(ValueError):audit([replica(0),replica(1000)],fold=fold)


def test_duplicate_expected_ids_are_invalid():
    with pytest.raises(ValueError,match='Duplicate'):audit([],{ 'r':[0,0]})


def test_cli_atomic_output_preserves_inputs_and_refuses_overwrite(tmp_path):
    from metag.tools.audit_sampling import main
    source=tmp_path/'records.json';out=tmp_path/'audit.json'
    source.write_text(json.dumps({'records':[replica(0),replica(1000,-9.5)],'expected_replicates':{'r':[0,1000]}}));old=source.read_bytes()
    main([str(source),'--max-equilibrium-fold-change','2','-o',str(out)])
    assert json.loads(out.read_text())['summary']['all_within_observed_replicate_tolerance']
    assert source.read_bytes()==old and not list(tmp_path.glob('*.tmp'))
    with pytest.raises(SystemExit):main([str(source),'-o',str(out)])


def test_extreme_sampling_sigma_stays_valid_json():
    report=audit([replica(0,sigma=1.79e308),replica(1000,-9,sigma=1.79e308)])
    pair=report['reactions']['r']['metrics']['pairwise_sampling_comparisons'][0]
    assert pair['status']=='sampling_sigma_scale_overflow'
    assert pair['quadrature_U_samp_kj'] is None
    json.dumps(report,allow_nan=False)


def test_missing_route_decisions_cannot_certify_comparability():
    rows=[replica(0),replica(1000)]
    for row in rows:row['result']['routes']={'warnings':[]}
    assert audit(rows)['reactions']['r']['status']=='unverifiable_replicate_set'


def test_malformed_status_is_preserved_as_failure():
    report=audit([replica(0),replica(1000,status=['unexpected'])])
    assert report['reactions']['r']['status']=='incomplete_replicate_set'


def test_common_non_kj_units_are_not_silently_converted():
    rows=[replica(0),replica(1000)]
    for row in rows:row['result']['energy_units']='kcal/mol'
    assert audit(rows)['reactions']['r']['metrics'] is None


def test_reported_stereochemistry_is_not_removed():
    one,two=replica(0),replica(1000)
    one['result']['species_scored']['a'][2]='C[C@H](O)C(=O)O'
    two['result']['species_scored']['a'][2]='C[C@@H](O)C(=O)O'
    assert audit([one,two])['reactions']['r']['status']=='incompatible_replicates'


def test_numerically_complete_but_suspect_result_remains_failed():
    one,two=replica(0),replica(1000)
    two['result']['suspect']='failed chemical identity check'
    assert audit([one,two])['reactions']['r']['status']=='incomplete_replicate_set'


def test_distinct_offsets_with_overlapping_seed_windows_cannot_pass():
    g=audit([replica(0),replica(1)],{'r':[0,1]})['reactions']['r']
    assert g['status']=='overlapping_sampling_seed_windows'
    assert g['overlapping_seed_windows'][0]['shared_seed_window']==[2,8]


def test_nonoverlapping_declared_windows_remain_comparable():
    assert audit([replica(0),replica(8)],{'r':[0,8]})['reactions']['r']['status']==PASS


def test_explicit_cluster_terms_require_independence_evidence_not_offset_labels():
    rows=[replica(0),replica(1000)]
    for row in rows:
        row['result']['routes']['species_diagnostics']['a']={'method':'explicit','sampling':{'status':'unavailable'}}
    g=audit(rows)['reactions']['r']
    assert g['status']=='unverifiable_replicate_set'
    assert any('explicit_cluster_sampling_not_varied' in issue for issue in g['records'][0]['metadata_issues'])


def test_exactly_cancelling_explicit_species_does_not_block_valid_implicit_test():
    rows=[replica(0),replica(1000)]
    for row in rows:
        for field in ('original_species','species_scored'):
            row['result'][field].update(x=[-1,0,'CC'],y=[1,0,'CC'])
        for label in ('x','y'):
            row['result']['routes']['species_diagnostics'][label]={'method':'explicit','sampling':{'status':'unavailable'}}
    assert audit(rows)['reactions']['r']['status']==PASS


def test_deterministic_liquid_water_reference_needs_no_random_seed_trace():
    rows=[replica(0),replica(1000)]
    for row in rows:
        balanced={'a':[-1,0,'CCOC(C)=O'],'b':[1,0,'CCO'],'acid':[1,0,'CC(=O)O'],'water':[-1,0,'O']}
        for field in ('original_species','species_scored'):row['result'][field]=deepcopy(balanced)
        row['result']['routes']['species_diagnostics']['acid']={'method':'implicit','sampling':{'status':'available','seed_offset':row['seed_offset']}}
        row['result']['routes']['species_diagnostics']['water']={'method':'liquid_water_reference','sampling':{'status':'not_applicable'}}
    assert audit(rows)['reactions']['r']['status']==PASS


@pytest.mark.parametrize('change',['wrong_seed','missing_trace','different_state'])
def test_nested_hydration_sampling_and_state_identity_are_verified(change):
    rows=[replica(0),replica(1000)]
    for row in rows:
        for field in ('original_species','species_scored'):
            row['result'][field]['a'][2]='CC=O';row['result'][field]['b'][2]='C=CO'
        row['result']['routes']['hydration_diagnostics']={'a':{'status':'complete','states':[
            {'status':'estimated','n_water':1,'smiles':'CC(O)O','species':{'method':'implicit',
             'sampling':{'status':'available','seed_offset':row['seed_offset']}}}]}}
    state=rows[1]['result']['routes']['hydration_diagnostics']['a']['states'][0]
    if change=='wrong_seed':state['species']['sampling']['seed_offset']=0
    if change=='missing_trace':del state['species']['sampling']
    if change=='different_state':state['smiles']='CCC(O)O'
    g=audit(rows)['reactions']['r'];assert g['status']!=PASS and g['metrics'] is None


def test_actual_per_species_seed_must_match_top_level_label():
    rows=[replica(0),replica(1000)]
    rows[1]['result']['routes']['species_diagnostics']['a']['sampling']['seed_offset']=0
    assert audit(rows)['reactions']['r']['status']=='unverifiable_replicate_set'


def test_cli_guard_exits_nonzero_after_preserving_failed_audit(tmp_path):
    from metag.tools.audit_sampling import main
    source=tmp_path/'missing.json';out=tmp_path/'audit.json'
    source.write_text(json.dumps({'records':[replica(0)],'expected_replicates':{'r':[0,1000]}}))
    with pytest.raises(SystemExit) as exc:
        main([str(source),'--max-equilibrium-fold-change','2','--require-within-tolerance','-o',str(out)])
    assert exc.value.code==1 and out.exists()
    assert json.loads(out.read_text())['reactions']['r']['missing_replicates']==[1000]


def test_cli_guard_requires_explicit_user_tolerance(tmp_path):
    from metag.tools.audit_sampling import main
    source=tmp_path/'input.json';out=tmp_path/'audit.json'
    source.write_text(json.dumps({'records':[replica(0),replica(1000)],'expected_replicates':{'r':[0,1000]}}))
    with pytest.raises(SystemExit) as exc:
        main([str(source),'--require-within-tolerance','-o',str(out)])
    assert exc.value.code==2 and not out.exists()
