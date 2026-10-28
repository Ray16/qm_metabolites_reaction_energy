"""Convert an explicit chemical reaction reference into declared protonation pools.

This is a reference-data utility, not a production scoring route. It accepts a
source chemical standard Gibbs energy (including any written H+), uses declared
sequential macroscopic acid ladders, and returns a transformed value at the SAME
temperature, zero ionic strength, zero free Mg activity and unit water activity.
No parameter is inferred, fitted, or taken from production's generic pKa rules.

All Gibbs energies are kJ/mol of the written reaction. Solute standards are 1 M
or 1 mol/kg; water is pure liquid. At infinite dilution c/c0 = rho*m/m0, with
rho numerically in kg/L. Consequently G_c = G_m - RT*delta_n_aq*ln(rho).
Written H+ counts in delta_n_aq; liquid water does not. A molal acid pKa becomes
pKa_c = pKa_m - log10(rho). Pooling then adds sum(nu*RT*ln(alpha_reference))
and -nu_Hplus*RT*ln(10)*pH. These conventions make AH -> A- + H+ close to zero
when both sides are represented by the same acid/base pool.

Numerical conversion is not reference admission. Source validity, molecular
identity, completeness of state ladders and input uncertainties remain explicit
caller responsibilities. A source's reported +/- is never called one-sigma.
The returned Jacobian supports J C J^T with a justified FULL input covariance;
no independence, missing-state bound or confidence interval is invented here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from metag.energetics.protonation import macroscopic_transform

R_KJ = 0.00831446261815324
STANDARDS = {"molar_1M", "molal_1mol_per_kg"}


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must identify its source or state")


@dataclass(frozen=True)
class AcidLadder:
    """One complete DECLARED proton-count model, ordered by proton loss.

    A state denotes its entire proton-count macrostate, including any tautomer
    multiplicity implicit in the measured macroscopic constants. It must not be
    silently identified with one computed geometry or a neutral tautomer.
    """

    pool_id: str
    state_ids: tuple[str, ...]
    first_state_charge: int
    pkas: tuple[float, ...]
    temperature_K: float
    standard_state: str
    source: str
    state_scope: str
    limitations: tuple[str, ...] = ()
    ionic_strength: float = 0.0

    def __post_init__(self):
        for name in ("pool_id", "source", "state_scope"):
            _text(getattr(self, name), name)
        if isinstance(self.state_ids, str) or isinstance(self.pkas, str):
            raise ValueError("states and pKas must be explicit sequences")
        object.__setattr__(self, "state_ids", tuple(self.state_ids))
        object.__setattr__(self, "pkas", tuple(self.pkas))
        if isinstance(self.limitations, str):
            raise ValueError("limitations must be a sequence")
        object.__setattr__(self, "limitations", tuple(self.limitations))
        if len(self.state_ids) != len(self.pkas) + 1 or len(set(self.state_ids)) != len(self.state_ids):
            raise ValueError("a ladder needs n+1 distinct state identities for n pKas")
        for state in self.state_ids:
            _text(state, "state identity")
        for value in self.pkas:
            _number(value, "pKa")
        if isinstance(self.first_state_charge, bool) or not isinstance(self.first_state_charge, int):
            raise ValueError("first_state_charge must be an integer")
        _number(self.temperature_K, "temperature_K", positive=True)
        if self.standard_state not in STANDARDS:
            raise ValueError("pKa standard state must be molar_1M or molal_1mol_per_kg")
        if _number(self.ionic_strength, "ionic_strength") != 0:
            raise ValueError("this converter supports zero-ionic-strength constants only")


@dataclass(frozen=True)
class PoolTerm:
    coefficient: float
    ladder: AcidLadder
    reference_deprotonations: int
    reference_state_id: str

    def __post_init__(self):
        if _number(self.coefficient, "coefficient") == 0:
            raise ValueError("omit zero-coefficient terms")
        if not isinstance(self.ladder, AcidLadder):
            raise ValueError("term requires an AcidLadder")
        index = self.reference_deprotonations
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.ladder.state_ids):
            raise ValueError("reference_deprotonations is outside the declared ladder")
        if self.reference_state_id != self.ladder.state_ids[index]:
            raise ValueError("reference identity does not match the selected proton-count macrostate")


def molal_to_molar_shift(temperature_K, delta_n_aqueous, density_kg_L):
    """Standard Gibbs shift at infinite dilution; water excluded, H+ included."""
    t = _number(temperature_K, "temperature_K", positive=True)
    n = _number(delta_n_aqueous, "delta_n_aqueous")
    rho = _number(density_kg_L, "density_kg_L", positive=True)
    return -R_KJ * t * n * math.log(rho)


def convert_chemical_reference(
    *, chemical_dg_kj, chemical_standard_state, temperature_K,
    target_temperature_K, target_pH, terms, proton_coefficient,
    water_coefficient, density_kg_L, source, identity_provenance,
    target_ionic_strength=0.0, target_free_magnesium_activity=0.0,
    water_standard_state="pure_liquid", water_activity=1.0,
    limitations=(),
):
    """Return a traceable central value and input sensitivities, never an admission.

    ``proton_coefficient`` is the explicit signed coefficient in the chemical
    reference equation, not an inferred total proton difference between pools.
    Input terms must account for every non-water/non-proton solute. Charge is
    checked; identity_provenance must document the independent elemental/state
    mapping, which this numerical utility does not establish from text labels.
    """
    dg = _number(chemical_dg_kj, "chemical_dg_kj")
    t = _number(temperature_K, "temperature_K", positive=True)
    tt = _number(target_temperature_K, "target_temperature_K", positive=True)
    ph = _number(target_pH, "target_pH")
    nh = _number(proton_coefficient, "proton_coefficient")
    nw = _number(water_coefficient, "water_coefficient")
    rho = _number(density_kg_L, "density_kg_L", positive=True)
    _text(source, "source")
    _text(identity_provenance, "identity_provenance")
    if chemical_standard_state not in STANDARDS:
        raise ValueError("chemical reference standard state is not supported")
    if not math.isclose(t, tt, rel_tol=0, abs_tol=1e-9):
        raise ValueError("temperature normalization requires separate source thermochemistry")
    if _number(target_ionic_strength, "target_ionic_strength") != 0:
        raise ValueError("ionic-strength transformation is not supplied by this converter")
    if _number(target_free_magnesium_activity, "target_free_magnesium_activity") != 0:
        raise ValueError("Mg-bound pools need a separate binding model")
    if water_standard_state != "pure_liquid" or _number(water_activity, "water_activity") != 1:
        raise ValueError("this reference contract requires pure liquid water at unit activity")
    if isinstance(limitations, str):
        raise ValueError("limitations must be a sequence")
    terms = tuple(terms)
    if not terms or not all(isinstance(term, PoolTerm) for term in terms):
        raise ValueError("all solute terms must be explicit PoolTerms")
    pools = {}
    for term in terms:
        ladder = term.ladder
        if not math.isclose(ladder.temperature_K, t, rel_tol=0, abs_tol=1e-9):
            raise ValueError("pKa temperature does not match the chemical reference")
        if ladder.pool_id in pools and pools[ladder.pool_id] != ladder:
            raise ValueError("one pool identity cannot carry conflicting state models")
        pools[ladder.pool_id] = ladder
    charge = math.fsum(term.coefficient * (term.ladder.first_state_charge - term.reference_deprotonations)
                       for term in terms) + nh
    if abs(charge) > 1e-9 * max(1.0, sum(abs(term.coefficient) for term in terms) + abs(nh)):
        raise ValueError("chemical reference is not charge balanced including explicit H+")
    rt, rtln10 = R_KJ * t, R_KJ * t * math.log(10)
    delta_n = math.fsum(term.coefficient for term in terms) + nh
    shift = molal_to_molar_shift(t, delta_n, rho) if chemical_standard_state == "molal_1mol_per_kg" else 0.0
    jacobian = {"chemical_dg_kj": 1.0, "target_pH": -nh * rtln10,
                "density_kg_L": -rt * delta_n / rho if chemical_standard_state == "molal_1mol_per_kg" else 0.0}
    details, pool_terms = [], []
    issues = list(limitations)
    for term in terms:
        ladder = term.ladder
        pka_shift = -math.log10(rho) if ladder.standard_state == "molal_1mol_per_kg" else 0.0
        pkas = [pka + pka_shift for pka in ladder.pkas]
        result = macroscopic_transform(pkas, ph, rtln10,
                                       reference_deprotonations=term.reference_deprotonations)
        nu, index = term.coefficient, term.reference_deprotonations
        value = nu * result["G_transform"]
        pool_terms.append(value)
        probabilities = result["populations"]
        mean_n = math.fsum(n * pop for n, pop in enumerate(probabilities))
        jacobian["target_pH"] -= nu * rtln10 * (mean_n - index)
        for j in range(len(pkas)):
            derivative = nu * rtln10 * (math.fsum(probabilities[j + 1:]) - (1 if index > j else 0))
            key = f"pKa:{ladder.pool_id}:{j}"
            jacobian[key] = jacobian.get(key, 0.0) + derivative
            if ladder.standard_state == "molal_1mol_per_kg":
                jacobian["density_kg_L"] -= derivative / (rho * math.log(10))
        details.append({"pool_id": ladder.pool_id, "coefficient": nu,
                        "reference_state_id": term.reference_state_id,
                        "reference_deprotonations": index, "pkas_molar": pkas,
                        "populations": dict(zip(ladder.state_ids, probabilities)),
                        "reference_fraction": probabilities[index],
                        "pool_transform_kj": result["G_transform"],
                        "reaction_contribution_kj": value, "source": ladder.source,
                        "state_scope": ladder.state_scope})
        issues.extend(ladder.limitations)
    proton_term = -nh * rtln10 * ph
    total = math.fsum([dg, shift, proton_term, *pool_terms])
    if not math.isfinite(total):
        raise ValueError("unrepresentable transformed reference")
    return {"chemical_dg_kj": dg, "standard_state_shift_kj": shift,
            "chemical_dg_molar_kj": dg + shift, "delta_n_aqueous": delta_n,
            "proton_term_kj": proton_term, "pooling_term_kj": math.fsum(pool_terms),
            "transformed_dg_kj": total, "terms": details, "source": source,
            "identity_provenance": identity_provenance,
            "conditions": {"temperature_K": t, "pH": ph, "ionic_strength_M": 0.0,
                           "free_magnesium_activity": 0.0, "solute_standard_state": "molar_1M",
                           "water_standard_state": "pure_liquid", "water_activity": 1.0},
            "written_water_coefficient": nw, "written_proton_coefficient": nh,
            "parameter_jacobian": jacobian,
            "uncertainty_scope": "No total uncertainty or coverage asserted; use full justified covariance J C J^T, including shared pKas and chemical-reference dependence",
            "limitations": list(dict.fromkeys(issues)),
            "admission_status": "numerical_conversion_only_source_review_required"}
