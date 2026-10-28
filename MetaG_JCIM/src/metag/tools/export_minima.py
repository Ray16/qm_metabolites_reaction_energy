"""Export dev11 accepted species-cache conformers as XYZ and auditable metadata.

Usage: python -m metag.tools.export_minima SPECIES_CACHE_JSON -o NEW_DIRECTORY
Coordinates are exported from recorded accepted states. This utility does not
revalidate the minimum or infer geometry from an older energy-only cache.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from metag.energetics.provenance import ordered_geometry_record


def export_minima(source, destination):
    source, destination = Path(source), Path(destination)
    raw = source.read_bytes(); cache = json.loads(raw)
    rows = cache.get('meta', {}).get('thermal', {}).get('ensemble', {}).get('validated_minima')
    if not isinstance(rows, list) or not rows:
        raise ValueError('cache has no accepted implicit-ensemble minima')
    prepared=[]
    for index, row in enumerate(rows):
        geometry=row.get('geometry')
        if not isinstance(geometry, dict):
            raise ValueError('accepted geometry missing; an older energy-only cache cannot be reconstructed')
        try:
            canonical=ordered_geometry_record(geometry['symbols'], geometry['positions_angstrom'],
                                              geometry['charge'], geometry['spin_multiplicity'])
        except (KeyError, TypeError) as exc:
            raise ValueError('incomplete accepted geometry') from exc
        if canonical != geometry:
            raise ValueError('accepted geometry identity is mismatched or altered')
        values=[row.get(k) for k in ('electronic_kj','solvation_kj','thermal_kj','total_kj')]
        if any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) for v in values):
            raise ValueError('finite component energies required')
        if not math.isclose(sum(values[:3]), values[3], rel_tol=0, abs_tol=1e-7):
            raise ValueError('component energies do not sum to recorded total')
        lines=[str(len(geometry['symbols'])), f"MetaG accepted gas minimum {index}; charge={geometry['charge']} spin_multiplicity={geometry['spin_multiplicity']}"]
        lines += [symbol+' '+ ' '.join(format(x,'.17g') for x in xyz)
                  for symbol,xyz in zip(geometry['symbols'],geometry['positions_angstrom'])]
        text='\n'.join(lines)+'\n'
        prepared.append((f'minimum_{index:04d}.xyz',text,row))
    # Validate every row before creating any output; never overwrite another run.
    destination.mkdir(parents=True,exist_ok=False)
    outputs=[]
    for name,text,row in prepared:
        (destination/name).write_text(text)
        outputs.append({'xyz_file':name,'xyz_sha256':hashlib.sha256(text.encode()).hexdigest(),**row})
    report={'scope':'Recorded accepted gas geometries and their existing component energies; no new minimum, ground-state, solvent or ensemble-completeness validation.',
            'source_cache_sha256':hashlib.sha256(raw).hexdigest(),'source_smiles':cache.get('smi'),
            'source_charge':cache.get('q'),'source_settings':cache.get('settings'),
            'n_minima':len(outputs),'minima':outputs}
    (destination/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('-o','--output',type=Path,required=True)
    args=parser.parse_args(argv)
    report=export_minima(args.source,args.output)
    print(json.dumps({'n_minima':report['n_minima'],'source_cache_sha256':report['source_cache_sha256'],'output':str(args.output)},indent=2))


if __name__=='__main__':main()
