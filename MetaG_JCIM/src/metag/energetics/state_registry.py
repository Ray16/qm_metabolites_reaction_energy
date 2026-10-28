"""Explicit, condition-specific chemical-state reference models.

This layer does not enumerate chemistry, infer pKas, bridge conditions, fit
residuals or activate production routes. State gaps are complete relative free
energies. A measured gap replaces a calculated gap; it is never an offset added
to that gap. Macrostates already contain their microscopic partition functions.

Reservoir counts are relative to a common molecular skeleton. An edge a -> b
consumes ``counts(b) - counts(a)`` protons, waters and Mg ions. Activities are
dimensionless; pMg always means *free* Mg activity, not total concentration.
Calculated standard gaps must already subtract standard reservoir chemical
potentials. Concentration-ratio references remain conditional quantities.
"""
from __future__ import annotations

import math
import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass

from metag.energetics.microstates import R_KJ, combine


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0 or nonnegative and number < 0:
        raise ValueError(f"{name} outside allowed range")
    return number


def _tuple(value, name):
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain explicit identities, not a string")
    return tuple(value)


def _text(value, name, *, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{name} must be an explicit string")


def _unknown_text(value):
    if not isinstance(value, str):
        return False
    text = value.strip().lower().replace("_", " ")
    return text in {"none", "null", "n/a", "na", "?"} or text.startswith(
        ("unknown", "unspecified", "not reported", "not available", "missing"))


@dataclass(frozen=True)
class Conditions:
    temperature_K: float
    solvent: str
    ionic_strength_M: float | None = None
    pH: float | None = None
    pMg_free: float | None = None
    water_activity: float | None = None
    activity_scale: str = "dimensionless_activity"
    recipe: str = ""
    standard_state: str | None = None

    def __post_init__(self):
        _number(self.temperature_K, "temperature_K", positive=True)
        _text(self.solvent, "solvent")
        _text(self.recipe, "recipe", allow_empty=True)
        for name in ("ionic_strength_M", "pH", "pMg_free", "water_activity"):
            value = getattr(self, name)
            if value is not None:
                _number(value, name, positive=name == "water_activity",
                        nonnegative=name == "ionic_strength_M")
        if self.activity_scale not in {"dimensionless_activity", "conditional_concentration"}:
            raise ValueError("unknown activity_scale")
        if self.standard_state is not None:
            _text(self.standard_state, "standard_state")

    def fingerprint(self) -> str:
        """Canonical condition ID; equal numeric int/float values share an ID."""
        data = asdict(self)
        for key in ("temperature_K", "ionic_strength_M", "pH", "pMg_free", "water_activity"):
            if data[key] is not None:
                data[key] = float(data[key])
                if data[key] == 0:
                    data[key] = 0.0
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return "state-conditions-v1:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class State:
    id: str
    scope: str = "microscopic"
    smiles: str | None = None
    proton_count: int = 0
    water_count: int = 0
    magnesium_count: int = 0
    charge: int = 0
    members: tuple[str, ...] = ()
    degeneracy: float = 1.0
    domain_issues: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "members", _tuple(self.members, "members"))
        object.__setattr__(self, "domain_issues", _tuple(self.domain_issues, "domain_issues"))
        _text(self.id, "state id")
        for member in self.members:
            _text(member, "macrostate member id")
        if self.smiles is not None:
            _text(self.smiles, "state smiles")
        if not self.id or self.scope not in {"microscopic", "macroscopic"}:
            raise ValueError("state requires an id and microscopic/macroscopic scope")
        for name in ("proton_count", "water_count", "magnesium_count", "charge"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
            _number(getattr(self, name), name)
        _number(self.degeneracy, "degeneracy", positive=True)
        if self.degeneracy != 1:
            raise ValueError("complete state gaps already include multiplicity; extra degeneracy would double count")
        if self.scope == "macroscopic" and (self.degeneracy != 1 or self.smiles is not None):
            raise ValueError("a macrostate is an ensemble, with no single SMILES or extra degeneracy")
        if self.scope == "macroscopic" and not self.members:
            raise ValueError("macrostate member identities must be specified")
        if self.scope == "microscopic" and self.members:
            raise ValueError("microscopic state cannot contain a macrostate member list")
        if len(set(self.members)) != len(self.members):
            raise ValueError("duplicate macrostate members")


@dataclass(frozen=True)
class Source:
    id: str
    locator: str
    conditions: Conditions
    determination: str = "measured"
    location: str = ""
    artifact_sha256: str | None = None
    uncertainty: float | None = None
    uncertainty_kind: str | None = None
    prior_development_exposure: bool | None = None
    dependencies: tuple[str, ...] = ()
    reference_scope: str = "unspecified"

    def __post_init__(self):
        object.__setattr__(self, "dependencies", _tuple(self.dependencies, "dependencies"))
        _text(self.id, "source id")
        _text(self.locator, "source locator")
        for dependency in self.dependencies:
            _text(dependency, "source dependency")
        if not isinstance(self.conditions, Conditions):
            raise ValueError("source conditions must be a Conditions record")
        if self.prior_development_exposure is not None and type(self.prior_development_exposure) is not bool:
            raise ValueError("development exposure must be bool or None")
        if not self.id or not self.locator:
            raise ValueError("source id and primary/calculation locator are required")
        if self.determination not in {"measured", "derived", "calculated"}:
            raise ValueError("source must identify measured, derived or calculated determination")
        if self.reference_scope not in {"microscopic", "macroscopic", "unspecified"}:
            raise ValueError("unknown source reference scope")
        if self.uncertainty is not None:
            _number(self.uncertainty, "source uncertainty", nonnegative=True)
            if not self.uncertainty_kind:
                raise ValueError("source uncertainty convention is required")
            _text(self.uncertainty_kind, "uncertainty_kind")
        if self.determination == "derived" and not self.dependencies:
            raise ValueError("derived references must identify their input dependencies")
        if self.artifact_sha256 is not None:
            _text(self.artifact_sha256, "artifact_sha256")
            if len(self.artifact_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.artifact_sha256.lower()):
                raise ValueError("artifact_sha256 must be a SHA256 hex digest")


@dataclass(frozen=True)
class Transition:
    id: str
    initial: str
    final: str
    kind: str
    value: float | None
    source: Source
    scope: str = "microscopic"
    # The source and target must match all these axes. pH/pMg are independently
    # required for conditional gaps when the source fixes those reservoirs.
    condition_axes: tuple[str, ...] = ("temperature_K", "solvent", "ionic_strength_M", "activity_scale", "standard_state")
    reservoir_convention: str = ""

    def __post_init__(self):
        object.__setattr__(self, "condition_axes", _tuple(self.condition_axes, "condition_axes"))
        for key, value in (("transition id", self.id), ("initial state", self.initial), ("final state", self.final)):
            _text(value, key)
        _text(self.reservoir_convention, "reservoir_convention", allow_empty=True)
        if not isinstance(self.source, Source):
            raise ValueError("transition source must be a Source record")
        if not self.id or self.initial == self.final:
            raise ValueError("transition needs an id and two distinct states")
        if self.kind not in {"pKa", "log10K", "standard_gap_kj", "conditional_gap_kj"}:
            raise ValueError("unknown reference quantity")
        if self.scope not in {"microscopic", "macroscopic"}:
            raise ValueError("unknown reference scope")
        if self.value is not None:
            _number(self.value, "transition value")
        allowed = set(Conditions.__dataclass_fields__) - {"recipe"}
        if not set(self.condition_axes) <= allowed:
            raise ValueError("unknown source-condition axis")
        if not {"temperature_K", "solvent", "ionic_strength_M", "activity_scale", "standard_state"} <= set(self.condition_axes):
            raise ValueError("temperature, solvent, ionic strength, standard state and reference scale matching are mandatory")
        if self.source.determination == "calculated" and self.kind not in {"standard_gap_kj", "conditional_gap_kj"}:
            raise ValueError("calculated state gaps must be recorded in energy units")


@dataclass(frozen=True)
class Anchor:
    state_id: str
    energy_kj: float
    conditions: Conditions
    scope: str
    locator: str
    includes_population_correction: bool = False

    def __post_init__(self):
        _text(self.state_id, "anchor state id")
        _text(self.locator, "anchor locator")
        if not isinstance(self.conditions, Conditions):
            raise ValueError("anchor conditions must be a Conditions record")
        if type(self.includes_population_correction) is not bool:
            raise ValueError("anchor population-correction declaration must be bool")
        _number(self.energy_kj, "anchor energy")
        if self.scope not in {"microscopic", "macroscopic"} or not self.locator:
            raise ValueError("anchor needs explicit scope and provenance")
        if self.includes_population_correction:
            raise ValueError("an already population-transformed anchor would double count states")


@dataclass(frozen=True)
class Registry:
    id: str
    states: tuple[State, ...]
    transitions: tuple[Transition, ...]
    scope_note: str = ""

    def __post_init__(self):
        object.__setattr__(self, "states", _tuple(self.states, "states"))
        object.__setattr__(self, "transitions", _tuple(self.transitions, "transitions"))
        _text(self.id, "registry id")
        if any(not isinstance(s, State) for s in self.states) or any(not isinstance(t, Transition) for t in self.transitions):
            raise ValueError("registry requires explicit State and Transition records")
        if not self.id or not self.states:
            raise ValueError("registry id and states required")
        state_map = {s.id: s for s in self.states}
        if len(state_map) != len(self.states):
            raise ValueError("duplicate state identity")
        if len({t.id for t in self.transitions}) != len(self.transitions):
            raise ValueError("duplicate transition identity")
        members = [member for s in self.states for member in (s.members if s.scope == "macroscopic" else (s.id,))]
        if len(set(members)) != len(members):
            raise ValueError("macrostate and microscopic populations overlap")
        for edge in self.transitions:
            if edge.initial not in state_map or edge.final not in state_map:
                raise ValueError("transition references an unknown state")
            a, b = state_map[edge.initial], state_map[edge.final]
            if a.scope != b.scope or a.scope != edge.scope:
                raise ValueError("microscopic/macroscopic reference mapping is inconsistent")
            exchange = exchanges(a, b)
            if b.charge - a.charge != exchange["proton"] + 2 * exchange["magnesium"]:
                raise ValueError("charge change disagrees with H/Mg reservoir exchanges")
            if edge.kind == "pKa" and exchange != {"proton": -1, "water": 0, "magnesium": 0}:
                raise ValueError("pKa edge must release exactly one proton")
            if edge.scope == "macroscopic" and (a.degeneracy != 1 or b.degeneracy != 1):
                raise ValueError("macroscopic constants already include state multiplicity")

    def to_dict(self):
        """Portable provenance record; does not serialize an accepted production route."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Rehydrate a portable registry, rerunning identity/provenance validation."""
        states = tuple(State(**{**s, "members": tuple(s.get("members", ())),
                                 "domain_issues": tuple(s.get("domain_issues", ()))}) for s in data["states"])
        edges = []
        for raw in data["transitions"]:
            source = raw["source"]
            source = Source(**{**source, "conditions": Conditions(**source["conditions"]),
                               "dependencies": tuple(source.get("dependencies", ()))})
            edges.append(Transition(**{**raw, "source": source,
                "condition_axes": tuple(raw.get("condition_axes", Transition.__dataclass_fields__["condition_axes"].default))}))
        return cls(data["id"], states, tuple(edges), data.get("scope_note", ""))


def exchanges(initial: State, final: State) -> dict[str, int]:
    """Reservoir molecules consumed in initial -> final; negative means release."""
    return {"proton": final.proton_count - initial.proton_count,
            "water": final.water_count - initial.water_count,
            "magnesium": final.magnesium_count - initial.magnesium_count}


def _condition_issues(source: Conditions, target: Conditions, axes) -> list[str]:
    issues = []
    for name in axes:
        a, b = getattr(source, name), getattr(target, name)
        if a is None or b is None or _unknown_text(a) or _unknown_text(b):
            issues.append(f"missing_condition:{name}")
        elif a != b:
            issues.append(f"condition_mismatch:{name}")
    # Recipe matching is deliberately explicit, not an inferred concentration,
    # ionic-strength or solvent-isotope bridge.
    if _unknown_text(source.recipe) or _unknown_text(target.recipe):
        issues.append("missing_condition:recipe")
    elif source.recipe != target.recipe:
        issues.append("condition_mismatch:recipe")
    return issues


def _structural_issues(states) -> list[str]:
    """Check supplied molecular identities without inventing microstate structures."""
    from rdkit import Chem
    from collections import Counter

    issues, formulas, identities, isotope_formulas = [], [], {}, []
    for state in states:
        issues.extend(f"{state.id}:{issue}" for issue in state.domain_issues)
        if state.smiles is None:
            if state.scope == "microscopic":
                issues.append(f"{state.id}:missing_structure_identity")
            continue
        mol = Chem.MolFromSmiles(state.smiles)
        if mol is None:
            issues.append(f"{state.id}:invalid_smiles")
            continue
        if Chem.GetFormalCharge(mol) != state.charge:
            issues.append(f"{state.id}:structure_charge_mismatch")
        identity = Chem.MolToSmiles(mol, isomericSmiles=True)
        if identity in identities:
            issues.append(f"{state.id}:duplicate_microscopic_identity:{identities[identity]}")
        identities[identity] = state.id
        if any(a.GetSymbol() == "P" and any(n.GetSymbol() == "N" for n in a.GetNeighbors()) for a in mol.GetAtoms()):
            issues.append(f"{state.id}:phosphoramidate_quarantine")
        composition = Counter(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms())
        isotope_formulas.append((state.id, Counter((a.GetSymbol(), a.GetIsotope())
                                                   for a in mol.GetAtoms() if a.GetIsotope())))
        composition["H"] -= state.proton_count + 2 * state.water_count
        composition["O"] -= state.water_count
        composition["Mg"] -= state.magnesium_count
        if any(value < 0 for value in composition.values()):
            issues.append(f"{state.id}:invalid_reservoir_composition")
        formulas.append((state.id, +composition, state.charge - state.proton_count - 2 * state.magnesium_count))
    if formulas:
        reference_formula, reference_charge = formulas[0][1:]
        for name, formula, charge in formulas[1:]:
            if formula != reference_formula or charge != reference_charge:
                issues.append(f"{name}:reservoir_composition_mismatch")
    if isotope_formulas:
        reference_isotopes = isotope_formulas[0][1]
        for name, labels in isotope_formulas[1:]:
            if labels != reference_isotopes:
                issues.append(f"{name}:isotope_reservoir_mapping_required")
    return issues


def transition_gap(edge: Transition, initial: State, final: State, conditions: Conditions) -> dict:
    """Convert a complete transition reference, preserving missing parameters."""
    issue = _condition_issues(edge.source.conditions, conditions, edge.condition_axes)
    counts = exchanges(initial, final)
    if _unknown_text(edge.source.locator):
        issue.append("missing_source_provenance")
    if edge.source.reference_scope == "unspecified":
        issue.append("missing_reference_scope")
    elif edge.source.reference_scope != edge.scope:
        issue.append("source_reference_scope_mismatch")
    if edge.source.conditions.activity_scale == "conditional_concentration":
        # A native concentration ratio is not an activity equilibrium constant.
        # Preserve a fixed measured/conventional water background instead of
        # silently using that ratio to predict a different solvent activity.
        if counts["water"] or edge.source.conditions.water_activity is not None:
            issue.extend(_condition_issues(edge.source.conditions, conditions, ("water_activity",)))
        if counts["magnesium"] and edge.kind == "log10K" and (
                not edge.reservoir_convention or _unknown_text(edge.reservoir_convention)):
            issue.append("missing_conditional_Mg_reference_convention")
    if edge.value is None:
        issue.append("missing_parameter")
    if (edge.kind == "standard_gap_kj" and edge.source.determination == "calculated"
            and any(counts.values()) and (not edge.reservoir_convention or _unknown_text(edge.reservoir_convention))):
        issue.append("missing_standard_reservoir_convention")
    if edge.kind == "conditional_gap_kj":
        # Conditional measured populations already contain reservoir effects;
        # transforming them again would count these terms twice.
        axes = [key for key, count in (("pH", counts["proton"]), ("pMg_free", counts["magnesium"]),
                                     ("water_activity", counts["water"]))
                if count or getattr(edge.source.conditions, key) is not None]
        issue.extend(_condition_issues(edge.source.conditions, conditions, axes))
        gap = edge.value
    else:
        rt = R_KJ * conditions.temperature_K
        if counts["proton"] and conditions.pH is None:
            issue.append("missing_reservoir:pH")
        if counts["magnesium"] and conditions.pMg_free is None:
            issue.append("missing_reservoir:pMg_free")
        if counts["water"] and conditions.water_activity is None:
            issue.append("missing_reservoir:water_activity")
        if edge.kind == "pKa":
            gap = None if edge.value is None else rt * math.log(10) * edge.value
        elif edge.kind == "log10K":
            gap = None if edge.value is None else -rt * math.log(10) * edge.value
        else:
            gap = edge.value
        if gap is not None and not issue:
            gap += rt * math.log(10) * (counts["proton"] * (conditions.pH or 0)
                                        + counts["magnesium"] * (conditions.pMg_free or 0))
            gap -= rt * counts["water"] * math.log(conditions.water_activity or 1)
    if gap is not None and not math.isfinite(gap):
        issue.append("nonfinite_converted_gap")
    return {"id": edge.id, "initial": edge.initial, "final": edge.final,
            "gap_kj": None if issue else gap, "issues": sorted(set(issue)), "exchanges": counts,
            "source": asdict(edge.source), "quantity": edge.kind,
            "scope": edge.scope, "reservoir_convention": edge.reservoir_convention}


def resolve(registry: Registry, conditions: Conditions, *, reference_state: str,
            model_transition_ids: tuple[str, ...], anchor: Anchor | None = None,
            cycle_tolerance_kj: float = 1e-6) -> dict:
    """Resolve a declared state model and audit redundant edges without refitting.

    Callers explicitly select model edges before inspecting prediction errors.
    All other edges are provenance-labelled audits, never additive offsets;
    derived references are not independent of their recorded dependencies.
    Missing states, incompatible selected references or inconsistent selected
    cycles produce no ensemble estimate. Inconsistent redundant references also
    block acceptance while retaining the selected-model diagnostic populations.
    No statistical uncertainty estimate is invented from marginal source SDs.
    """
    _number(cycle_tolerance_kj, "cycle_tolerance_kj", nonnegative=True)
    if not isinstance(registry, Registry) or not isinstance(conditions, Conditions):
        raise ValueError("resolve requires Registry and Conditions records")
    if anchor is not None and not isinstance(anchor, Anchor):
        raise ValueError("resolve anchor must be an Anchor record")
    states = {s.id: s for s in registry.states}
    edges = {t.id: t for t in registry.transitions}
    if reference_state not in states:
        raise ValueError("unknown reference state")
    if len(set(model_transition_ids)) != len(model_transition_ids) or not set(model_transition_ids) <= set(edges):
        raise ValueError("model transition ids must be unique registered identities")
    issues = _structural_issues(registry.states)
    rt = R_KJ * conditions.temperature_K
    if not math.isfinite(rt) or rt <= 0:
        issues.append("unrepresentable_thermal_scale")
    issues.extend(_condition_issues(conditions, conditions,
                  ("temperature_K", "solvent", "ionic_strength_M", "activity_scale", "standard_state")))
    anchor_energy = 0.0
    if anchor is not None:
        if anchor.state_id != reference_state or anchor.scope != states[reference_state].scope:
            issues.append("anchor_reference_mapping_mismatch")
        axes = ("temperature_K", "solvent", "ionic_strength_M", "activity_scale", "standard_state") + tuple(
            key for key in ("pH", "pMg_free", "water_activity")
            if getattr(anchor.conditions, key) is not None or getattr(conditions, key) is not None)
        issues.extend(f"anchor:{i}" for i in _condition_issues(anchor.conditions, conditions, axes))
        anchor_energy = anchor.energy_kj
    converted = {t.id: transition_gap(t, states[t.initial], states[t.final], conditions) for t in registry.transitions}
    adjacency = {s: [] for s in states}
    selected_pairs = set()
    for id in model_transition_ids:
        item = converted[id]
        pair = frozenset((item["initial"], item["final"]))
        if pair in selected_pairs:
            issues.append(f"duplicate_model_gap:{id}")
        selected_pairs.add(pair)
        if item["issues"]:
            issues.extend(f"{id}:{i}" for i in item["issues"])
            continue
        adjacency[item["initial"]].append((item["final"], item["gap_kj"], id))
        adjacency[item["final"]].append((item["initial"], -item["gap_kj"], id))
    energies = {reference_state: anchor_energy}
    paths = {reference_state: ()}
    pending = deque([reference_state])
    tree_edges = set()
    selected_residuals = {}
    while pending:
        state_id = pending.popleft()
        for next_id, gap, edge_id in adjacency[state_id]:
            value = energies[state_id] + gap
            if not math.isfinite(value):
                issues.append(f"nonfinite_state_energy:{next_id}")
                continue
            if next_id not in energies:
                energies[next_id] = value
                paths[next_id] = paths[state_id] + (edge_id,)
                tree_edges.add(edge_id)
                pending.append(next_id)
            elif edge_id not in tree_edges:
                selected_residuals[edge_id] = value - energies[next_id]
    missing = sorted(set(states) - set(energies))
    issues.extend(f"unresolved_state:{s}" for s in missing)
    for id, residual in selected_residuals.items():
        if not math.isfinite(residual):
            issues.append(f"nonfinite_selected_cycle_residual:{id}")
        elif abs(residual) > cycle_tolerance_kj:
            issues.append(f"inconsistent_selected_cycle:{id}")
    audit = []
    quality_warnings = []
    for id, item in converted.items():
        residual = None
        if item["gap_kj"] is not None and item["initial"] in energies and item["final"] in energies:
            residual = energies[item["final"]] - energies[item["initial"]] - item["gap_kj"]
            if not math.isfinite(residual):
                issues.append(f"nonfinite_cycle_residual:{id}")
                residual = None
            elif abs(residual) > cycle_tolerance_kj:
                issues.append(f"inconsistent_reference_cycle:{id}")
        audit.append({**item, "used_in_model": id in model_transition_ids,
                      "cycle_residual_kj": residual})
        source = item["source"]
        if item["issues"] and id not in model_transition_ids:
            quality_warnings.append(f"{id}:unresolved_audit_reference")
        if source["determination"] == "measured" and source["uncertainty"] is None:
            quality_warnings.append(f"{id}:measurement_uncertainty_not_available")
        if source["determination"] == "derived":
            quality_warnings.append(f"{id}:derived_reference_not_independent_of_dependencies")
        if source["prior_development_exposure"] is True:
            quality_warnings.append(f"{id}:prior_development_exposure")
        elif source["prior_development_exposure"] is None:
            quality_warnings.append(f"{id}:development_exposure_not_established")
    if conditions.activity_scale == "conditional_concentration":
        quality_warnings.append("conditional_concentration_model_not_standard_activity_label")
    if conditions.solvent != "H2O":
        quality_warnings.append("empirical_solvent_record_does_not_establish_production_scorer_support")
    ensemble = None
    if rt > 0 and math.isfinite(rt):
        issues.extend(f"unrepresentable_partition_weight:{state_id}" for state_id, energy in energies.items()
                      if not math.isfinite(energy / rt))
    # Structural/mapping errors are not merely measurement disagreement.
    model_blocked = any(not i.startswith("inconsistent_reference_cycle:") for i in issues)
    if not model_blocked:
        ensemble = combine([{"name": s.id, "G": energies[s.id], "degeneracy": s.degeneracy}
                            for s in registry.states], conditions.temperature_K)
        ensemble.pop("sigma")
        ensemble["uncertainty_status"] = "not_propagated; source covariance and reference dependencies required"
        if (not math.isfinite(ensemble["G"]) or
                not all(math.isfinite(value) for value in ensemble["populations"].values())):
            issues.append("nonfinite_ensemble_result")
            ensemble = None
    return {"registry_id": registry.id, "status": "accepted" if not issues else "no_estimate",
            "issues": sorted(set(issues)), "conditions": asdict(conditions),
            "condition_id": conditions.fingerprint(),
            "cycle_tolerance_kj": cycle_tolerance_kj,
            "reference_state": reference_state, "reference_scope": states[reference_state].scope,
            "anchor": None if anchor is None else asdict(anchor),
            "relative_only": anchor is None, "state_energies_kj": energies, "state_paths": paths,
            "model_transition_ids": list(model_transition_ids), "transition_audit": audit,
            "reference_quality_warnings": sorted(set(quality_warnings)),
            "cycle_audit_complete": all(item["cycle_residual_kj"] is not None for item in audit),
            "ensemble": ensemble if not issues else None,
            "diagnostic_selected_ensemble": ensemble,
            "scope_note": registry.scope_note,
            "limitations": ["Source-condition equality is required; no condition bridge supplied.",
                            "Conditional concentration references are not standard activity free energies.",
                            "Reference/source independence and microscopic identity mapping remain caller responsibilities.",
                            "Acceptance is internal consistency of declared states, not complete species coverage or scorer validation.",
                            "Cycle tolerance is a declared numerical acceptance threshold, not a confidence interval.",
                            "No production route activation, model fitting or pKa substitution."]}
