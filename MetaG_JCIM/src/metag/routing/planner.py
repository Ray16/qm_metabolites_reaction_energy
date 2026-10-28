"""Reaction route planning without model loading or species-energy evaluation.

The returned reaction and route diagnostics retain the pipeline API's schema.
This layer depends on molecular structure, routing policy and protonation rules;
it never imports the QM pipeline or uses observed reaction energies.
"""
import os
from rdkit import Chem

from metag.chem import is_balanced, reaction_residual, fmt_num
from metag.config import flag as _flag
from metag.reactions import validate_reaction


def route_reaction(
    reaction,
    allow_truncate=True,
    trunc_radius=None,
    log=print,
    force_neutral=False,
    policy=None,
):
    """Structural routing -- pure logic, no QM: cofactor ring cores -> CoA / NTP cores -> full-vs-truncate
    gate -> spectator truncation -> pH-0 transform (-> optional zwitterion / neutral-QM invariant).

    Returns (rx, routes, truncated): `rx` is the rewritten reaction score_reaction scores (the input dict
    is never mutated), `routes` records which transforms fired plus any errors/warnings.

    BALANCE GUARD: every rewrite must keep the reaction element- and charge-balanced with its proton
    count in n_H+ -- the one invariant all of them share. A step whose output breaks a balance its input
    had is REVERTED and recorded in routes["reverted"]. Without it an element leak (e.g. a mixed disulfide
    replaced by a symmetric core, dropping the CoA half) reached QM and was caught, if at all, only by
    the |ΔG| sanity bound."""
    from metag.routing.policy import RoutingPolicy

    validate_reaction(reaction)
    from metag.routing.pka_transform import pka_model_config
    pka_model_config()  # invalid selections/artifacts fail before any QM work
    if policy is None:
        policy = RoutingPolicy.from_runtime(_flag, os.environ, trunc_radius)
    elif trunc_radius is not None:
        raise ValueError("trunc_radius cannot be combined with an explicit routing policy")
    elif not isinstance(policy, RoutingPolicy):
        raise TypeError("policy must be a RoutingPolicy")
    from metag.routing.production_policy import validate_production_policy
    validate_production_policy(policy)
    from metag.routing.sugar_states import prepare_declared_sugar_inputs
    prepared, sugar_declarations = prepare_declared_sugar_inputs(reaction)
    rx = dict(prepared)
    rx.setdefault("note", "")
    rx.setdefault("n_Hplus", 0)
    # Routing describes a chemical event, independent of how many times the
    # equation is written. Several core builders require unit coefficients.
    # Normalize for planning, then restore the caller's extent before assembly.
    from metag.reactions import unit_stoichiometry
    _unit_species, _route_extent = unit_stoichiometry(rx["species"])
    if _route_extent != 1.0:
        rx = dict(rx, species=_unit_species, n_Hplus=rx["n_Hplus"] / _route_extent)
        if rx.get("pka_sites"):
            rx["pka_sites"] = [list(site[:2]) + [site[2] if len(site) > 2 else "acid",
                                (site[3] if len(site) > 3 else 1.0) / _route_extent]
                               for site in rx["pka_sites"]]
    rx.setdefault("explicit", False)
    if isinstance(rx["explicit"], list):
        rx["explicit"] = set(rx["explicit"])
    truncated = False                                        # did AUTO_TRUNCATE actually fire?
    routes = {"cofactor_ring": False, "coa_core": False, "ntp_core": False, "prefer_full": False,
              "truncated": False, "trunc_radius": None, "ph0": False, "zwitterion_ph0": False,
              "errors": [], "warnings": [], "reverted": [], "input_balanced": True}
    routes["routing_mode"] = policy.mode
    routes["stereo_preservation_policy"] = "2026-10-18-source-centres-and-defined-fragments"
    if sugar_declarations:
        routes['declared_sugar_pools'] = sugar_declarations
        routes['generic_truncation_disabled_for_declared_sugar_pools'] = True
    if not is_balanced(rx["species"], rx["n_Hplus"]):
        res, dq = reaction_residual(rx["species"], rx["n_Hplus"])
        routes["input_balanced"] = False
        routes["warnings"].append(f"input reaction not element/charge balanced (residual {res}, charge "
                                  f"{fmt_num(dq) if dq is not None else '?'}): ΔG is not meaningful")
        log(f"  !! {routes['warnings'][-1]}")
    rx = _route_cores(rx, routes, policy, log)
    rx = _route_truncation(rx, routes, policy, allow_truncate and not sugar_declarations, log)
    truncated = routes["truncated"]
    if policy.mode == "compound_reference":
        from metag.routing.compound_reference import build_compound_reference_reaction, POLICY_VERSION
        routes["compound_reference_policy"] = POLICY_VERSION
        try:
            rx, references = build_compound_reference_reaction(rx)
            routes["compound_references"] = references
            routes["protonation_sites_by_species"] = {
                name: [[('react' if rx['species'][name][0] < 0 else 'prod'), pka, kind,
                         abs(rx['species'][name][0])]
                       for kind, values in (('acid', row['acid_pkas']), ('base', row['base_pkas']))
                       for pka in values] for name, row in references.items()}
            routes["compound_reference_status"] = "complete"
            routes["ph0"] = any(row['acid_pkas'] or row['base_pkas'] for row in references.values())
        except (ValueError, RuntimeError) as exc:
            routes["compound_reference_status"] = "failed"
            routes["errors"].append("compound_reference: " + str(exc))
            log("  [compound-reference failure; no fallback to reaction-dependent references]")
    else:
        rx = _route_ph0(rx, routes, policy, log)
        rx = _route_neutral_reference(rx, routes, policy, force_neutral, log)
    if _route_extent != 1.0:
        rx = dict(rx, species={n: [c * _route_extent, q, s] for n, (c, q, s) in rx["species"].items()},
                  n_Hplus=rx["n_Hplus"] * _route_extent)
        if rx.get("pka_sites"):
            rx["pka_sites"] = [list(site[:2]) + [site[2] if len(site) > 2 else "acid",
                                (site[3] if len(site) > 3 else 1.0) * _route_extent]
                               for site in rx["pka_sites"]]
        if routes.get('protonation_sites_by_species'):
            routes['protonation_sites_by_species'] = {
                name: [list(site[:3]) + [site[3] * _route_extent] for site in sites]
                for name, sites in routes['protonation_sites_by_species'].items()}
    return rx, routes, truncated


def _balance_guard(step, before, after, routes, log, reset=None):
    """Keep `after` if it is balanced (or if `before` already was not); otherwise revert to `before`,
    record the error, and reset the step's route flags."""
    if after is before or not is_balanced(before["species"], before["n_Hplus"]):
        return after
    if is_balanced(after["species"], after["n_Hplus"]):
        return after
    res, dq = reaction_residual(after["species"], after["n_Hplus"])
    msg = (f"{step}: rewritten reaction not balanced (residual {res}, charge "
           f"{fmt_num(dq) if dq is not None else '?'}) -> step reverted")
    routes["reverted"].append(msg)                        # designed "not applicable" outcome, not an error
    log(f"  !! {msg}")
    for k, v in (reset or {}).items():
        routes[k] = v
    return before



def _route_cores(rx, routes, policy, log):
    """Apply structural cofactor, CoA and nucleotide cores, checking balance after each."""
    _pre = rx
    if policy.cofactor_ring:
        try:
            from metag.routing.cofactor_cores import cofactor_ring
            new = cofactor_ring(rx["species"])
            if new is not rx["species"]:
                # Coring changes species CHARGES; n_H+ must move by the negative of the charge change
                # to keep charge closure (Σcoeff·q + n_H+ = 0). This is a NO-OP for symmetric
                # NAD(P)+/NADH swaps (the +2/-2 shifts cancel) but CORRECTS asymmetric double-redox
                # (2 GSH : 1 GSSG) where the old "n_H+ preserved" left a -2 imbalance -> ~+2340 kJ
                # proton leak (glutathione reductase rxn00070/00086 -> +2363; fixed -> ~+22, exp ~+15).
                dq = (sum(c * q for (c, q, s) in new.values())
                      - sum(c * q for (c, q, s) in rx["species"].values()))
                rx = dict(rx, species=new, n_Hplus=rx["n_Hplus"] - dq,
                          note=rx.get("note", "") + " [RINGCOFACTOR]")
                routes["cofactor_ring"] = True
                log(f"  [cofactor-ring: NAD(P) -> nicotinamide ring model"
                    + (f"; n_H+ {fmt_num(rx['n_Hplus']+dq)}->{fmt_num(rx['n_Hplus'])} (charge closure)]" if dq else "]"))
        except Exception as e:
            log(f"  [cofactor-ring error: {e}; unchanged]"); routes["errors"].append(f"cofactor_ring: {e}")
    rx = _balance_guard("cofactor_ring", _pre, rx, routes, log, reset={"cofactor_ring": False})
    # The optional CoA core preserves the reactive acyl-thioester region.
    _pre = rx
    if policy.coa_core:
        try:
            from metag.routing.coa_core import coa_core
            new = coa_core(rx["species"])
            if new is not rx["species"]:
                rx = dict(rx, species=new, note=rx.get("note", "") + " [COA-CORE]"); routes["coa_core"] = True
                log("  [coa-core: acyl-S-CoA -> acyl-S-CH3 (pantetheine-ADP scaffold capped)]")
        except Exception as e:
            log(f"  [coa-core error: {e}; unchanged]"); routes["errors"].append(f"coa_core: {e}")
    rx = _balance_guard("coa_core", _pre, rx, routes, log, reset={"coa_core": False})
    # Cap a conserved nucleoside while retaining the reacting polyphosphate.
    _pre = rx
    if policy.ntp_core:
        try:
            from metag.routing.ntp_core import ntp_core
            new = ntp_core(rx["species"])
            if new is not rx["species"]:
                rx = dict(rx, species=new, note=rx.get("note", "") + " [NTP-CORE]"); routes["ntp_core"] = True
                log("  [ntp-core: nucleoside-5'-phosphate -> methyl polyphosphate (adenosine capped)]")
        except Exception as e:
            log(f"  [ntp-core error: {e}; unchanged]"); routes["errors"].append(f"ntp_core: {e}")
    rx = _balance_guard("ntp_core", _pre, rx, routes, log, reset={"ntp_core": False})
    return rx


def _route_truncation(rx, routes, policy, allow_truncate, log):
    """Choose full molecules or a balanced reactive core under the routing policy."""
    _prefer_full = False
    # Experimental ring-chain species require their full intramolecular closure partners.
    if allow_truncate and policy.auto_truncate and policy.ring_chain:
        from metag.routing.ring_chain import has_ring_chain
        _rc = [nm for nm, (c, q, s_) in rx["species"].items() if has_ring_chain(s_)]
        if _rc:
            _prefer_full = True
            routes["prefer_full"] = True
            routes["ring_chain_full"] = _rc
            log(f"  [route: ring-chain species {', '.join(_rc)} -> full molecules (no truncation)]")
    if allow_truncate and policy.auto_truncate and policy.route_full and not _prefer_full:
        try:
            from metag.routing.truncation_gate import prefer_full
            _prefer_full = prefer_full(rx["species"], anchor_correct=policy.anchor_correct)
        except Exception as e:
            _prefer_full = False; routes["errors"].append(f"prefer_full: {e}")
        routes["prefer_full"] = bool(_prefer_full)
        if _prefer_full:
            log("  [route: prefer FULL molecule (compact ring, charge-conserved, no floppy linker) -> skip truncation]")
    _pre = rx
    if allow_truncate and policy.auto_truncate and not _prefer_full:
        try:
            from metag.routing.truncate import build_truncated_reaction, truncation_radius
            _radius_override = policy.trunc_radius_explicit
            _rad = policy.trunc_radius
            if policy.trunc_anomeric_radius and not _radius_override:
                _adaptive_rad = truncation_radius(rx["species"], default=_rad)
                if _adaptive_rad != _rad:
                    _rad = _adaptive_rad
                    routes["trunc_radius_reason"] = "anomeric_ring_reaction_center"
                    log("  [truncation radius 3: preserve the complete reacting sugar environment]")
            tr = build_truncated_reaction(rx["species"], radius=_rad, fg_cuts=policy.trunc_fg_cuts)
            if tr is None and policy.trunc_v2:   # v2: global-map truncation for the
                from metag.routing.truncate_global import build_truncated_reaction_v2   # multi-coeff/unequal-side cases
                tr = build_truncated_reaction_v2(rx["species"], radius=_rad)
                if tr is not None:
                    log("  [v2 global-map truncation engaged]")
            if tr is None and policy.trunc_maxanion_retry:
                # PROTONATION-CONSISTENT RETRY: the removed spectator must be identical on both sides, but sources
                # draw the same moiety with different protons (ModelSEED: ATP's AMP part as [O-], ADP's as O), which
                # fails the removed-fragment consistency guard. Retry on the max-anion (fully deprotonated) forms --
                # only when pH-0 will re-neutralize the cores afterwards, so the scored microspecies is unchanged in
                # kind; the core's n_H+ comes from its own H/charge and the balance guard checks the result.
                from metag.routing.pka_transform import _canonicalize_maxanion, is_isomerization
                if policy.ph0_auto and not rx.get("pka_sites") and not is_isomerization(rx["species"]):
                    canon = {}
                    for nm, (c, q, s_) in rx["species"].items():
                        cs = _canonicalize_maxanion(s_)
                        canon[nm] = [c, Chem.GetFormalCharge(Chem.MolFromSmiles(cs)), cs]
                    if canon != {k: list(v) for k, v in rx["species"].items()}:
                        tr = build_truncated_reaction(canon, radius=_rad, fg_cuts=policy.trunc_fg_cuts)
                        if tr is not None:
                            routes["truncated_maxanion"] = True
                            log("  [truncation on max-anion forms (protonation-consistent spectators)]")
            # VALIDITY GUARD (general): a truncation removes CONSERVED spectator parts and caps cut C-C
            # bonds FAR from functional groups. It must NEVER cut a bond ON a CATIONIC center: capping a
            # quaternary ammonium [N+](C)(C)(C) with H yields a PRIMARY ammonium CH2-NH3+ -- a different
            # molecule (carnitine betaine -> primary amine; silently wrong ΔG, the failed 3-dehydrocarnitine
            # core just exposed it). GENERAL symptom for ANY cation (N+, S+, guanidinium, sulfonium...):
            # a positively-charged atom in the core carries MORE H than the same atom in the full species.
            # Match the truncation's 'X_t' core name back to the original 'X'. (Root-cause fix would live in
            # truncate.py's cut-site selection; this is the safety guard + the species-failure retry below.)
            def _truncation_mangles_cation(orig, cored):
                # symptom = a positively-charged atom LOST heavy-atom coordination vs the full species
                # (quaternary N+ degree 4 -> bare [N+] degree 1 = an undervalent, chemically-invalid
                # cation). Element-agnostic (N+, S+, guanidinium C...); compares the max cation heavy-degree.
                from rdkit import Chem as _C
                def cation_deg(smi):
                    m = _C.MolFromSmiles(smi)
                    return sorted(a.GetDegree() for a in m.GetAtoms() if a.GetFormalCharge() > 0) if m else []
                for nm, (c, q, s) in cored.items():
                    onm = nm[:-2] if nm.endswith("_t") else nm
                    if onm not in orig:
                        continue
                    dc, do = cation_deg(s), cation_deg(orig[onm][2])
                    # mangled = a RETAINED cation lost coordination. A cation absent from the core (dc empty) was
                    # removed WITH the spectator -- identical on both sides, so it cancels: not a mangle (this
                    # branch used to reject ~89% of the truncations the guard blocked, e.g. SAM/SAH, carnitine)
                    removed_with_spectator = not dc and policy.trunc_spectator_cations
                    if do and not removed_with_spectator and (not dc or max(dc) < max(do)):
                        return nm
                return None
            if tr is not None:
                bad = _truncation_mangles_cation(rx["species"], tr[0])
                if bad:
                    log(f"  [auto-truncation REJECTED: mangles a cationic center ({bad}) -> full molecules]")
                    routes["warnings"].append(f"truncation rejected: mangles a cationic center ({bad})")
                    tr = None
            if tr is not None:
                from metag.validation.stereochemistry import unresolved_species
                if not unresolved_species(rx['species']):
                    introduced = unresolved_species(tr[0])
                    if introduced:
                        routes['stereochemistry_truncation_rejected'] = introduced
                        routes['warnings'].append('truncation rejected: introduces unresolved stereochemical identity')
                        log('  [auto-truncation REJECTED: undefined core stereochemistry -> retain pre-truncation molecules]')
                        tr = None
            if tr is not None:
                rx = dict(rx, species=tr[0], n_Hplus=tr[1], explicit=False,
                          note=rx.get("note", "") + " [AUTO-TRUNCATED]")
                truncated = True
                routes.update(truncated=True, trunc_radius=_rad)
                log(f"  [auto-truncated -> {len(tr[0])} core species, n_H+={tr[1]}]")
            else:
                log(f"  [auto-truncation fallback: full molecules]")
        except Exception as e:
            log(f"  [auto-truncation error: {e}; full molecules]"); routes["errors"].append(f"truncation: {e}")
    rx = _balance_guard("truncation", _pre, rx, routes, log, reset={"truncated": False, "trunc_radius": None})
    truncated = routes["truncated"]
    return rx


def _route_ph0(rx, routes, policy, log):
    """Build neutral reference species and their protonation terms after core selection."""
    _pre = rx
    if policy.ph0_auto and not rx.get("pka_sites"):
        try:
            from metag.routing.pka_transform import build_ph0_reaction, is_isomerization
            if is_isomerization(rx["species"]) and not policy.ph0_isomerase:
                # ISOMERASE GATE (PH0_ISOMERASE=1 lifts it: A/B under ALPB, where the neutral species are the
                # better-described ones -- the gate was set on the COSMO baseline): pH-0 hurts isomerizations (no anion-solvation change to
                # fix; neutralising spectator anions only injects sampling noise). Skip.
                log("  [pH0-auto: isomerization -> gated OFF (pH-0 would only add noise)]")
            else:
                # UNIFIED build: anions ALWAYS neutralized; basic amines additionally when the
                # internal base gate passes (PH0_BASES, default-on) -> covers the net-proton
                # deamination/transaminase/lyase class the old anion-only path refused.
                why = []
                owned = {}
                out = build_ph0_reaction(rx["species"], rx["n_Hplus"],
                                         base=policy.ph0_bases, why=why, site_ownership=owned)
                if out is not None:
                    ns, pks, nh = out
                    rx = dict(rx, species=ns, n_Hplus=nh, pka_sites=pks, explicit=False,
                              note=rx.get("note", "") + " [pH0]")
                    routes["ph0"] = True
                    routes['protonation_sites_by_species'] = owned
                    log(f"  [pH0 -> {len(pks)} pKa sites, n_H+={fmt_num(nh)}]")
                else:
                    reason = why[0] if why else "no ionizable site"
                    routes["ph0_skipped"] = reason
                    log(f"  [pH0: {reason} -> unchanged]")
        except Exception as e:
            log(f"  [pH0-auto error: {e}; unchanged]"); routes["errors"].append(f"ph0: {e}")
    rx = _balance_guard("ph0", _pre, rx, routes, log, reset={"ph0": False})
    if rx is _pre and not routes['ph0']:
        routes.pop('protonation_sites_by_species', None)
    return rx


def _route_neutral_reference(rx, routes, policy, force_neutral, log):
    """Apply the configured neutral-reference guard to remaining ionizable charges."""
    _neutral_all = policy.neutral_qm or force_neutral     # force_neutral: re-route after a rearranged species
    _pre = rx
    previous_ownership = routes.get('protonation_sites_by_species')
    previous_ph0 = routes['ph0']
    if policy.zwitterion_ph0 or _neutral_all:
        try:
            from metag.routing.pka_transform import build_ph0_reaction, has_zwitterion, has_ionized
            _needs = (lambda sp: has_ionized(sp)) if _neutral_all else (lambda sp: has_zwitterion(sp))
            if _needs(rx["species"]) and rx.get("pka_sites"):
                routes["warnings"].append("charged species reaches QM (user-supplied pka_sites; not re-routed)")
            elif _needs(rx["species"]):
                owned = {}
                out = build_ph0_reaction(rx["species"], rx["n_Hplus"], base=True, force_base=True,
                                         site_ownership=owned)
                if out is not None and not _needs(out[0]):
                    ns, pks, nh = out
                    rx = dict(rx, species=ns, n_Hplus=nh, pka_sites=pks,
                              explicit=False, note=rx.get("note", "") + " [pH0-zwitterion]")
                    routes["ph0"] = True; routes["zwitterion_ph0"] = True
                    routes['protonation_sites_by_species'] = owned
                    log(f"  [zwitterion invariant -> neutral-microspecies path, {len(pks)} pKa sites, n_H+={fmt_num(nh)}]")
                else:
                    routes["warnings"].append("zwitterion reaches QM (neutral-microspecies build refused)")
                    log("  !! zwitterion reaches QM: neutral-microspecies build refused")
        except Exception as e:
            routes["errors"].append(f"zwitterion_ph0: {e}")
    rx = _balance_guard("zwitterion_ph0", _pre, rx, routes, log, reset={"zwitterion_ph0": False})
    if rx is _pre and routes.get('protonation_sites_by_species') is not previous_ownership:
        routes['ph0'] = previous_ph0
        if previous_ownership is None:
            routes.pop('protonation_sites_by_species', None)
        else:
            routes['protonation_sites_by_species'] = previous_ownership
    return rx
