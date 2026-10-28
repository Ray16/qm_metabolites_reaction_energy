"""Export preserves exact coordinates and fails before writing invalid records."""
import json
import pytest
from metag.energetics.provenance import ordered_geometry_record
from metag.tools.export_minima import export_minima


def cache(tmp_path):
    path=tmp_path/'cache.json'
    row={'electronic_kj':-10.,'solvation_kj':-2.,'thermal_kj':3.,'total_kj':-9.,
         'geometry':ordered_geometry_record(['O','H'],[[0.,0.,0.],[0.,0.,1.000000000000001]],0,1)}
    data={'smi':'O','q':0,'settings':{'source':'test'},'meta':{'thermal':{'ensemble':{'validated_minima':[row]}}}}
    path.write_text(json.dumps(data));return path,data


def test_export_retains_round_trip_precision_and_component_identity(tmp_path):
    source,data=cache(tmp_path);report=export_minima(source,tmp_path/'export')
    lines=(tmp_path/'export/minimum_0000.xyz').read_text().splitlines()
    assert float(lines[3].split()[-1])==data['meta']['thermal']['ensemble']['validated_minima'][0]['geometry']['positions_angstrom'][1][2]
    assert report['n_minima']==1
    assert report['minima'][0]['total_kj']==-9.
    assert (tmp_path/'export/manifest.json').is_file()


@pytest.mark.parametrize('damage',['missing_geometry','changed_coordinates','inconsistent_total'])
def test_invalid_cache_cannot_publish_an_export(tmp_path,damage):
    source,data=cache(tmp_path);row=data['meta']['thermal']['ensemble']['validated_minima'][0]
    if damage=='missing_geometry':row.pop('geometry')
    elif damage=='changed_coordinates':row['geometry']['positions_angstrom'][1][2]+=1.
    else:row['total_kj']+=1.
    source.write_text(json.dumps(data));destination=tmp_path/'export'
    with pytest.raises(ValueError):export_minima(source,destination)
    assert not destination.exists()


def test_export_never_overwrites_an_existing_directory(tmp_path):
    source,_=cache(tmp_path);destination=tmp_path/'export';destination.mkdir();(destination/'owned.txt').write_text('keep')
    with pytest.raises(FileExistsError):export_minima(source,destination)
    assert (destination/'owned.txt').read_text()=='keep'
