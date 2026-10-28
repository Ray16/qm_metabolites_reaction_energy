"""Replace solvent components on a complete, recorded sampled ensemble.

This CPU algebra does not resample conformers, validate ground states, alter
production routing, or transfer the old model's uncertainty calibration.
"""
import math
from collections.abc import Mapping
from metag.energetics.provenance import ordered_geometry_record,content_hash
from metag.energetics.strict_smd import request_record,_valid_record

# Same rounded scale as production conformers.boltz; no temperature transform.
PRODUCTION_KT_KJ = 2.4789


def _finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def _partition(values):
    lowest=min(values)
    exponentials=[math.exp(-(x-lowest)/PRODUCTION_KT_KJ) for x in values]
    total=math.fsum(exponentials)
    return lowest-PRODUCTION_KT_KJ*math.log(total),[x/total for x in exponentials]


def reweight_smd(cache, records):
    """Require every accepted geometry and one consistent converged SMD protocol.

    ``records`` maps exact ordered-geometry hashes to strict SMD records. No
    failed/missing state is dropped and no partial partition is renormalized.
    The stored cache energy must equal its recorded component partition, so
    solvent-relaxed, auxiliary or incompatible caches cannot be corrected.
    """
    if not isinstance(cache,Mapping) or not isinstance(records,Mapping):
        raise ValueError('species cache and solvent records must be mappings')
    meta=cache.get('meta',{});settings=cache.get('settings',{})
    if cache.get('method')!='implicit' or meta.get('solv_relax') or settings.get('via'):
        raise ValueError('requires a primary gas-minimum implicit ensemble')
    ensemble=meta.get('thermal',{}).get('ensemble',{})
    minima=ensemble.get('validated_minima')
    if not isinstance(minima,list) or not minima or ensemble.get('n_reference_thermal_approximation')!=0:
        raise ValueError('complete validated per-minimum thermal records required')
    keys=[];old=[];new=[];rows=[];identity=None;protocol=None;producer=None
    for row in minima:
        geometry=row.get('geometry',{})
        try:
            canonical=ordered_geometry_record(geometry['symbols'],geometry['positions_angstrom'],geometry['charge'],geometry['spin_multiplicity'])
        except (KeyError,TypeError) as exc:raise ValueError('accepted geometry missing') from exc
        if canonical!=geometry:raise ValueError('accepted geometry hash or coordinates altered')
        key=geometry['sha256']
        if key in keys:raise ValueError('duplicate exact geometry cannot count twice')
        keys.append(key)
        species_identity=(tuple(geometry['symbols']),geometry['charge'],geometry['spin_multiplicity'],row.get('reference_smiles'))
        if identity is not None and identity!=species_identity:raise ValueError('one ensemble must describe one declared microspecies')
        identity=species_identity
        values=[row.get(k) for k in ('electronic_kj','solvation_kj','thermal_kj','total_kj')]
        if not all(_finite(v) for v in values) or not math.isclose(math.fsum(values[:3]),values[3],abs_tol=1e-7,rel_tol=0):
            raise ValueError('recorded component energies must be finite and close')
        if key not in records:raise ValueError('missing solvent record; incomplete ensembles cannot be reweighted')
        record=records[key]
        if not isinstance(record,Mapping):raise ValueError('invalid solvent record')
        request=record.get('request',{});backend=record.get('backend',{})
        try:
            expected=request_record(geometry['symbols'],geometry['positions_angstrom'],geometry['charge'],request['basis'],xc=request['functional'],mult=geometry['spin_multiplicity'])
        except (KeyError,TypeError) as exc:raise ValueError('solvent protocol missing') from exc
        if not _valid_record(record,expected,backend):raise ValueError('failed or geometry/protocol-mismatched SMD record')
        signature=content_hash({k:v for k,v in expected.items() if k!='coordinates_A'})
        backend_signature=content_hash(backend)
        if protocol is not None and (protocol!=signature or producer!=backend_signature):raise ValueError('mixed solvent protocols or producers')
        protocol,producer=signature,backend_signature
        old.append(values[3]);new.append(values[0]+record['dgsolv']+values[2])
        rows.append({'geometry_sha256':key,'old_solvation_kj':values[1],'new_solvation_kj':record['dgsolv'],'old_total_kj':old[-1],'new_total_kj':new[-1]})
    if set(records)!=set(keys):raise ValueError('solvent records must match exactly the declared ensemble')
    old_g,old_weights=_partition(old);new_g,new_weights=_partition(new)
    if not _finite(cache.get('G')) or not math.isclose(cache['G'],old_g,abs_tol=1e-7,rel_tol=0):
        raise ValueError('source energy does not match recorded partition; no correction applied')
    for row,a,b in zip(rows,old_weights,new_weights):row.update(old_weight=a,new_weight=b)
    return {'status':'complete','source_smiles':cache.get('smi'),'n_minima':len(rows),'source_G_kj':old_g,'refined_G_kj':new_g,'delta_G_kj':new_g-old_g,
            'thermal_scale_kj':PRODUCTION_KT_KJ,'solvent_protocol_sha256':protocol,'solvent_producer_sha256':producer,'minima':rows,
            'uncertainty':None,'scope':'Fixed recorded sampled ensemble, existing electronic/RRHO terms, all solvent terms replaced and populations recomputed. No resampling, missing-basin bound, reaction correction or validated interval.'}
