"""CPU reweighting of a recorded species ensemble after complete SMD evaluation.

python -m metag.tools.refine_ensemble CACHE_JSON --records RESULTS_DIRECTORY -o NEW_JSON
The results directory holds geometry_sha256/record wrappers produced by a
reservation-owned SMD evaluator. This command does not submit QM calculations.
"""
import argparse,hashlib,json
from pathlib import Path
from metag.validation.solvent_ensemble import reweight_smd


def refine_from_files(source,records_directory,destination):
    source,records_directory,destination=map(Path,(source,records_directory,destination))
    raw=source.read_bytes();cache=json.loads(raw)
    expected={row['geometry']['sha256'] for row in cache.get('meta',{}).get('thermal',{}).get('ensemble',{}).get('validated_minima',[]) if 'geometry' in row}
    records={};files={}
    for p in sorted(records_directory.glob('*.json')):
        wrapper=json.loads(p.read_text());key=wrapper.get('geometry_sha256')
        if key not in expected:continue
        if key in records:raise ValueError('duplicate solvent result for one geometry')
        if wrapper.get('status')!='complete' or 'record' not in wrapper:raise ValueError('a required solvent calculation failed or is incomplete')
        records[key]=wrapper['record'];files[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    report=reweight_smd(cache,records)
    report.update(source_cache_sha256=hashlib.sha256(raw).hexdigest(),solvent_file_sha256=files)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('--records',type=Path,required=True);p.add_argument('-o','--output',type=Path,required=True);a=p.parse_args(argv)
    r=refine_from_files(a.source,a.records,a.output)
    print(json.dumps({k:r[k] for k in ('status','n_minima','source_G_kj','refined_G_kj','delta_G_kj')},indent=2))


if __name__=='__main__':main()
