"""Structure-dependent sampling budgets; no QM backend or environment reads."""
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def sampling_budget(smi, sample_scale=1.0, keep_scale=1.0):
    """FIX 1: scale conformer sampling to molecular flexibility (rotatable bonds).
    Rigid species are cheap; floppy sugars/chains need many seeds + a big pool or the
    Boltzmann ensemble is under-sampled (glycosyl failed at 2×48; step5c used 5×128)."""
    m = Chem.MolFromSmiles(smi)
    nrot = rdMolDescriptors.CalcNumRotatableBonds(m) if m is not None else 0
    # pools are generous — conformer gen + batched UMA relax are cheap; only the xtb
    # solvation on `keep` conformers scales linearly (threaded across cores).
    # SAMPLE_SCALE (env) multiplies seed count + pool for stress-testing convergence.
    sc = sample_scale                                    # read once at import, same value as the cache key
    if nrot <= 3:
        seeds, keep, pool = [1, 2], 10, 96
    elif nrot <= 7:
        seeds, keep, pool = [1, 2, 3], 14, 192
    else:
        seeds, keep, pool = [1, 2, 3, 4, 5, 6], 18, 320
    if sc != 1:
        seeds = list(range(1, max(2, int(round(len(seeds) * sc))) + 1))
        pool = int(round(pool * sc))
    # KEEP_SCALE (A/B, default 1): the physics review found `keep` too small for floppy sugars/cofactors, so
    # the Boltzmann ensemble under-counts conformational entropy (G too high). SAMPLE_SCALE scaled seeds+pool
    # but NOT keep; this scales how many low-E conformers are relaxed + solvated. Part of the cache key.
    if keep_scale != 1:
        keep = int(round(keep * keep_scale)); pool = max(pool, keep * 8)
    return seeds, keep, pool



def sampling_diagnostics(trajectory, successful_seeds, minimum_trajectory, *, seeds, hits,
                         required_hits, maximum, tolerance):
    """Describe the actual plateau criterion; it cannot bound unvisited basins."""
    return {
        'status': 'available',
        'stop_reason': 'plateau_criterion_met' if hits >= required_hits else 'batch_cap_reached',
        'ensemble_trajectory_kj': [float(value) for value in trajectory],
        'minimum_gas_energy_trajectory_kj': [float(value) for value in minimum_trajectory],
        'successful_batch_indices': list(successful_seeds),
        'batches_attempted': seeds,
        'consecutive_plateau_hits': hits,
        'required_plateau_hits': required_hits,
        'maximum_batches': maximum,
        'tolerance_kj': tolerance,
        'last_ensemble_increment_kj': abs(float(trajectory[-1] - trajectory[-2])) if len(trajectory) > 1 else None,
        'criterion': 'both cumulative ensemble free energy and minimum gas electronic energy change less than tolerance',
        'uncertainty_scope': 'U_samp is the spread of the last three successful cumulative estimates; not a bound on unsampled basins or model error',
    }


def species_diagnostics(metadata, source):
    """Keep missing historical diagnostics explicit; do not infer a plateau."""
    from copy import deepcopy
    result = {'method': 'implicit', 'source': source,
            'sampling': deepcopy(metadata.get('sampling', {
                'status': 'unavailable', 'reason': 'sampling trajectory and stopping criterion were not recorded'})),
            'thermal': deepcopy(metadata.get('thermal', {'status': 'unavailable'}))}
    if 'solv_relax' in metadata:
        result['solv_relax'] = deepcopy(metadata['solv_relax'])
    return result
