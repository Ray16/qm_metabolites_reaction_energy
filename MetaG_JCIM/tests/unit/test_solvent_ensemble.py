import copy,math
import hashlib,json
import pytest
from metag.energetics.provenance import ordered_geometry_record
from metag.energetics.strict_smd import request_record,HARTREE_KJ
from metag.validation.solvent_ensemble import reweight_smd,PRODUCTION_KT_KJ


def inputs(solvents=(0.,0.)):
    minima=[];records={}
    for j,(e,s) in enumerate(zip([0.,2.],solvents)):
        g=ordered_geometry_record(['H','H'],[[0,0,0],[0,0,.7+j*.1]],0,1)
        minima.append({'geometry':g,'reference_smiles':'[H][H]','electronic_kj':e,'solvation_kj':0.,'thermal_kj':0.,'total_kj':e})
        request=request_record(g['symbols'],g['positions_angstrom'],0,'def2-tzvpd',xc='wb97m-v')
        records[g['sha256']]={'status':'complete','request':request,'backend':{'cds_total_convention_verified':True,'producer_sha256':'test'},'gas_Eh':-1.,'smd_Eh':-1.+s/HARTREE_KJ,'smd_cds_Eh':.001,'gas_converged':True,'smd_converged':True,'dgsolv':s}
    cache={'method':'implicit','smi':'[H][H]','settings':{},'G':-PRODUCTION_KT_KJ*math.log1p(math.exp(-2/PRODUCTION_KT_KJ)),
           'meta':{'thermal':{'ensemble':{'validated_minima':minima,'n_reference_thermal_approximation':0}}}}
    return cache,records


def test_uniform_replacement_shift_is_preserved_without_double_counting_cds():
    c,r=inputs((3.,3.));out=reweight_smd(c,r)
    assert out['delta_G_kj']==pytest.approx(3.)
    assert [x['new_weight'] for x in out['minima']]==pytest.approx([x['old_weight'] for x in out['minima']])
    assert out['uncertainty'] is None


def test_reordering_recomputes_partition_and_populations_for_all_states():
    c,r=inputs((5.,-4.));out=reweight_smd(c,r)
    expected=-2.-PRODUCTION_KT_KJ*math.log1p(math.exp(-7/PRODUCTION_KT_KJ))
    assert out['refined_G_kj']==pytest.approx(expected)
    assert out['minima'][0]['old_weight']>out['minima'][1]['old_weight']
    assert out['minima'][0]['new_weight']<out['minima'][1]['new_weight']
    assert sum(x['new_weight'] for x in out['minima'])==pytest.approx(1.)


@pytest.mark.parametrize('damage',['missing','failed','coordinates','basis','producer','source_G','component','duplicate','relaxed','extra'])
def test_incompatible_or_incomplete_records_never_create_a_refined_energy(damage):
    c,r=inputs();keys=list(r)
    if damage=='missing':r.pop(keys[0])
    elif damage=='failed':r[keys[0]]['gas_converged']=False
    elif damage=='coordinates':r[keys[0]]['request']['coordinates_A'][1][2]+=.1
    elif damage=='basis':r[keys[0]]['request']['basis']='def2-svp'
    elif damage=='producer':r[keys[0]]['backend']['producer_sha256']='different'
    elif damage=='source_G':c['G']+=1.
    elif damage=='component':c['meta']['thermal']['ensemble']['validated_minima'][0]['total_kj']+=1.
    elif damage=='duplicate':c['meta']['thermal']['ensemble']['validated_minima'].append(copy.deepcopy(c['meta']['thermal']['ensemble']['validated_minima'][0]))
    elif damage=='relaxed':c['meta']['solv_relax']={'n_relaxed':2}
    elif damage=='extra':r['unrelated']=copy.deepcopy(r[keys[0]])
    with pytest.raises(ValueError):reweight_smd(c,r)


def write_inputs(tmp_path):
    c,r=inputs((3.,3.))
    source=tmp_path/'cache.json';source.write_text(json.dumps(c))
    directory=tmp_path/'solvent';directory.mkdir()
    for i,(key,record) in enumerate(r.items()):
        (directory/f'{i}.json').write_text(json.dumps({'geometry_sha256':key,'status':'complete','record':record}))
    return source,directory,tmp_path/'refined.json'


def test_file_refinement_records_hashes_and_refuses_overwrite(tmp_path):
    from metag.tools.refine_ensemble import refine_from_files
    source,directory,output=write_inputs(tmp_path)
    report=refine_from_files(source,directory,output)
    assert json.loads(output.read_text())==report
    assert report['source_cache_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(report['solvent_file_sha256'])==2
    original=output.read_bytes()
    with pytest.raises(FileExistsError):refine_from_files(source,directory,output)
    assert output.read_bytes()==original


@pytest.mark.parametrize('damage',['duplicate','failed','missing'])
def test_file_refinement_withholds_output_for_invalid_required_set(tmp_path,damage):
    from metag.tools.refine_ensemble import refine_from_files
    source,directory,output=write_inputs(tmp_path)
    p=directory/'0.json'
    if damage=='duplicate':(directory/'duplicate.json').write_bytes(p.read_bytes())
    elif damage=='failed':
        wrapper=json.loads(p.read_text());wrapper['status']='failed';p.write_text(json.dumps(wrapper))
    else:p.unlink()
    with pytest.raises(ValueError):refine_from_files(source,directory,output)
    assert not output.exists()
