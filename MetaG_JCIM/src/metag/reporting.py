"""CPU uncertainty diagnostics, separate from molecular energies and point assembly."""


def uncertainty_diagnostics(note, original_species, scored_species, routes, stages,
                            energy, sampling_spread, config):
    """Return historical width diagnostics without changing the reaction energy.

    Class assignment uses original structures and annotation. Feature-model
    inference uses shared primitive-extent normalization and no annotations.
    The caller applies the publication contract only to a valid point estimate.
    """
    from metag.uncertainty import reaction_sigma, prediction_interval, uq_model, feature_interval
    smiles = [value[2] for value in original_species.values()]
    sigma, breakdown = reaction_sigma(note, smiles, sampling_spread,
                                       species=original_species, config=config)
    lower, upper, center, info = prediction_interval(
        note, smiles, energy, level=95, species=original_species,
        U_samp=sampling_spread, config=config)
    if uq_model() == 'features':
        from metag.uq_features import primitive_reaction_features
        raw, extent = primitive_reaction_features(original_species, scored_species,
                                                  routes, stages, sampling_spread)
        class_sigma = sigma
        sigma, lower, upper, info = feature_interval(raw, energy, level=95,
            species=original_species, config=config, extent=extent)
        center = round(energy, 1)
        breakdown = {'uq_model': 'features', 'features': raw,
                     'class_model': dict(breakdown, sigma=class_sigma)}
    return sigma, breakdown, lower, upper, center, info
