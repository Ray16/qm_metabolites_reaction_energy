"""Aqueous-state mixtures evaluated through explicit species-energy callbacks.

Hydration and experimental ring-chain enumeration propagate conditional
sampling uncertainty through their partition sums. This module does not load a QM model,
read runtime switches, or decide whether either route should be enabled.
"""
import math
import numpy as np

from metag.energetics.errors import SpeciesRearranged


def mixture_sampling_uncertainty(state_energies, input_sigmas, rt, jacobian=None):
    """First-order sampling propagation through F = -RT log sum exp(-E_i/RT).

    ``jacobian[i,j]`` is dE_i/dx_j for the underlying independent estimates x.
    It is essential when calibrated relative energies share a reference x_0.
    These are conditional model populations, not measured populations. This
    accounts for neither missing states nor solvation/pKa/calibration errors.
    """
    energies = np.asarray(state_energies, dtype=float)
    sigmas = np.asarray(input_sigmas, dtype=float)
    if (energies.ndim != 1 or not energies.size or not np.isfinite(energies).all()
            or sigmas.ndim != 1 or not np.isfinite(sigmas).all() or (sigmas < 0).any()
            or not math.isfinite(rt) or rt <= 0):
        raise ValueError("Mixture energies, sampling sigmas and RT must be finite; sigmas >= 0, RT > 0")
    derivative = np.eye(len(energies)) if jacobian is None else np.asarray(jacobian, dtype=float)
    if derivative.shape != (len(energies), len(sigmas)) or not np.isfinite(derivative).all():
        raise ValueError("Mixture Jacobian must have one row per state and one column per input")
    weights = np.exp(-(energies - energies.min()) / rt)
    populations = weights / weights.sum()
    sensitivity = populations @ derivative
    sigma = float(np.linalg.norm(sensitivity * sigmas))
    return sigma, {"method": "first_order_partition_derivatives",
                   "conditional_populations": populations.tolist(),
                   "input_sensitivities": sensitivity.tolist(),
                   "scope": "sampling only; independent input estimates; fixed state set, reservoir and calibration parameters; not a prediction interval"}


def hydrate_all_sites(pu, rx, name, q, smi, G, sig, std, seeds, keep, pool, log, routes, *,
                      implicit, water_reference, acid_transform, calibrate):
    """CARBONYL_HYDRATION_ALL: fold the carbonyl <-> gem-diol equilibria of species `name` into G[name] by
    EXACT enumeration of its hydration microstates (every subset of its hydratable sites, each computed).
    On the pH-0 route each state carries its own acid transform (a gem-diol of an alpha-oxo acid is a weaker
    acid) before mixing; the carbonyl form's transform, which the reaction-level pKa sites already apply,
    is removed again. HYDRATION_CAL maps each hydration event with the K_hyd calibration."""
    from metag.routing import aldehyde_hydration as _ah
    n_sites = _ah.n_hydration_sites(smi)
    if n_sites > _ah.MAX_HYDRATION_SITES:
        routes["warnings"].append(
            f"{name}: {n_sites} hydratable sites exceeds complete-enumeration limit "
            f"{_ah.MAX_HYDRATION_SITES}; hydration ensemble not evaluated")
        routes.setdefault("incomplete_hydration", []).append(name)
        return
    enumeration = _ah.enumerate_hydration_states(smi)
    states = enumeration.states
    if not enumeration.complete:
        routes["warnings"].append(
            f"{name}: {enumeration.construction_failures} hydration subset(s) not constructible")
        routes.setdefault("incomplete_hydration", []).append(name)
        return
    if not states:
        return
    on_ph0 = bool(rx.get("pka_sites")) and q == 0
    t_c = acid_transform(smi) if on_ph0 else 0.0
    g_w = water_reference(pu) + std
    folded, sds = [], []
    diagnostic = {"n_site_subsets": enumeration.n_subsets, "n_unique_states": len(states),
                  "states": [], "status": "incomplete"}
    routes.setdefault("hydration_diagnostics", {})[name] = diagnostic
    for n, hyd in states:
        state_info = {"smiles": hyd, "n_water": n, "status": "incomplete", "species": {}}
        diagnostic["states"].append(state_info)
        try:
            Gd, sd = implicit(pu, q, hyd, seeds, keep, pool, log, f"{name}(hydrate{n})", routes["warnings"],
                              diagnostics=state_info["species"])
        except SpeciesRearranged as e:
            state_info.update(status="identity_unresolved", reason=str(e))
            routes["warnings"].append(f"hydration state omitted (not a minimum): {e}")
            routes.setdefault("incomplete_hydration", []).append(name)
            continue
        if (Gd is None or not math.isfinite(Gd) or sd is None
                or not math.isfinite(sd) or sd < 0):
            state_info["status"] = "no_estimate"
            routes["warnings"].append(f"hydration state omitted (QM failed): {name} {hyd}")
            routes.setdefault("incomplete_hydration", []).append(name)
            continue
        dg = Gd + std - n * g_w - G[name]                 # hydration free energy of this state (neutral form)
        if calibrate:
            a, b = _ah.HYDRATION_CAL
            dg = a * dg + n * b                           # calibration is per hydration event
        t_d = acid_transform(hyd) if on_ph0 else 0.0
        folded.append((n, G[name] + n * g_w + dg + t_d - t_c))
        state_info.update(status="estimated", relative_g_kj=float(dg + t_d - t_c))
        sds.append(sd)
    if not folded:
        return
    Geff = _ah.mixture_G_states(G[name], folded, g_w)
    log(f"    [hydration: {name} {len(folded)} state(s) -> shift {Geff - G[name]:+.1f}]")
    diagnostic.update(status="complete" if len(folded) == len(states) else "incomplete",
                      mixture_shift_kj=float(Geff - G[name]))
    # E_i = (1-a)*G_carbonyl + a*(G_hydrate_i-n_i*G_water)
    #       + n_i*b + acid-state corrections. Reusing G_carbonyl couples
    # every calibrated hydrated state; propagate from independent raw inputs.
    slope = _ah.HYDRATION_CAL[0] if calibrate else 1.0
    jacobian = np.eye(len(folded) + 1) * slope
    jacobian[:, 0] = 1.0 - slope
    jacobian[0, 0] = 1.0
    sigma, uncertainty = mixture_sampling_uncertainty(
        [G[name]] + [g - n * g_w for n, g in folded],
        [sig[name]] + sds, _ah.RT, jacobian)
    diagnostic["sampling_uncertainty"] = uncertainty
    diagnostic["carbonyl_population"] = uncertainty["conditional_populations"][0]
    for state, population in zip((s for s in diagnostic["states"] if s["status"] == "estimated"),
                                 uncertainty["conditional_populations"][1:]):
        state["conditional_population"] = population
    G[name] = Geff
    sig[name] = sigma


def fold_ring_chain(pu, rx, name, q, smi, G, sig, seeds, keep, pool, log, routes, *,
                    implicit, acid_transform, standard_state, temperature):
    """RING_CHAIN: G[name] <- -RT ln[exp(-G[name]/RT) + Σ_ring exp(-G_ring/RT)] over the lactam/hemiaminal
    ring forms of routing/ring_chain.py. On the pH-0 route each form carries its own acid transform (the
    ring's alpha-hydroxy acid vs the open alpha-keto acid) relative to the open form's, which the
    reaction-level pKa sites already apply. A ring form that fails or is not a minimum fails the species
    closed (incomplete ensemble), like hydration."""
    from metag.routing.ring_chain import enumerate_ring_forms
    enumeration = enumerate_ring_forms(smi)
    if not enumeration.complete:
        routes["warnings"].append(
            f"{name}: incomplete ring-chain enumeration ({enumeration.total_states} states, "
            f"limit {enumeration.state_limit}, "
            f"{enumeration.construction_failures} construction failure(s)); ensemble not evaluated")
        routes.setdefault("incomplete_ring_chain", []).append(name)
        return
    forms = enumeration.forms
    if not forms:
        return
    on_ph0 = bool(rx.get("pka_sites")) and q == 0
    t_open = acid_transform(smi) if on_ph0 else 0.0
    states, sds = [G[name]], []
    for k, rsmi in enumerate(forms, 1):
        try:
            Gr, sd = implicit(pu, q, rsmi, seeds, keep, pool, log, f"{name}(ring{k})", routes["warnings"])
        except SpeciesRearranged as e:
            routes["warnings"].append(f"ring form omitted (not a minimum): {e}")
            routes.setdefault("incomplete_ring_chain", []).append(name)
            continue
        if Gr is None:
            routes.setdefault("incomplete_ring_chain", []).append(name)
            continue
        if standard_state:
            Gr += standard_state                            # G[name] already carries the 1 M term
        states.append(Gr + ((acid_transform(rsmi) - t_open) if on_ph0 else 0.0))
        sds.append(sd or 0.0)
    if len(states) == 1:
        return
    rt = 8.314e-3 * temperature
    g0 = min(states)
    Geff = g0 - rt * math.log(sum(math.exp(-(g - g0) / rt) for g in states))
    log(f"    [ring-chain: {name} {len(states) - 1} ring form(s) -> shift {Geff - G[name]:+.1f}]")
    routes.setdefault("ring_chain", []).append(name)
    sigma, uncertainty = mixture_sampling_uncertainty(states, [sig[name]] + sds, rt)
    routes.setdefault("ring_chain_diagnostics", {})[name] = {"sampling_uncertainty": uncertainty}
    G[name] = Geff
    sig[name] = sigma
