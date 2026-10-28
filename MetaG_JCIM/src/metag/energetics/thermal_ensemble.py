"""Thermal corrections and ensemble bookkeeping with explicit backend callbacks.

Windowed corrections are provisional sampling diagnostics. Final assembly requires
all sampled representatives to have separately validated corrections.
"""
import numpy as np


def _coverage_info(values, corrected, window, maximum):
    """Describe approximation coverage without claiming an error bound.

    Vertical-energy ordering can change after thermal corrections. A small
    Boltzmann weight before those corrections does not establish convergence.
    """
    lowest = min(values)
    eligible = sum(value - lowest < window for value in values)
    return {
        "n_minima": len(values),
        "n_with_thermal_correction": len(corrected),
        "n_reference_thermal_approximation": len(values) - len(corrected),
        "n_within_vertical_window": eligible,
        "thermal_cap_reached": eligible > maximum,
        "n_outside_vertical_window": len(values) - eligible,
        "thermal_window_kj": window,
        "thermal_maximum": maximum,
        "thermal_coverage_scope": "sampled minima only; no bound on omitted thermal corrections or unsampled basins",
    }


class ThermalTrack:
    """Per-minimum RRHO bookkeeping DURING sampling (THERMAL_ENSEMBLE): after every seed batch the unique
    minima within self.window kJ of the lowest E+ΔGsolv (at most self.maximum) get a UMA-Hessian RRHO
    correction, so the convergence test and the sampling-uncertainty trajectory see G = E + ΔGsolv + Gcorr.
    Corrections are keyed by the representative GEOMETRY (not the list index), so a representative replaced
    by a lower-G duplicate (UniqueMinima.add updates E, G and geometry together) gets a new Hessian, and
    removing / adding minima cannot misassign a correction. Minima outside the window carry the lowest
    window member's correction, an approximation whose missing contribution is not bounded.
    Imaginary modes are floored as soft here; the
    Final assembly requires separately validated corrections for EVERY sampled minimum."""

    def __init__(self, pu, template, q, mult, *, correction, combine, window, maximum):
        if maximum < 1 or not np.isfinite(window) or window <= 0:
            raise ValueError("thermal window and maximum must be positive")
        self.correction, self.combine = correction, combine
        self.window, self.maximum = window, maximum
        self.pu, self.q, self.mult = pu, q, mult
        self.syms = [a.GetSymbol() for a in template.GetAtoms()]
        self.corr = {}                                     # geometry key -> provisional Gcorr
        self.validated = {}                                # exact geometry -> strict (Gcorr, info)
        self.n_hessians = 0

    @staticmethod
    def _key(uniq, j):
        return uniq.ref[j].GetConformer().GetPositions().tobytes()

    def _ensure(self, uniq, j):
        k = self._key(uniq, j)
        if k not in self.corr:
            pos = uniq.ref[j].GetConformer().GetPositions()
            g, info = self.correction(self.pu, self.syms, pos, self.q, spin=self.mult, return_info=True)
            self.n_hessians += 1
            if info.get("n_imag") == 0 and np.isfinite(g):
                self.validated[k] = (g, {key: value for key, value in info.items() if key != "_imag_vecs"})
            else:
                # Preserve the provisional estimator for unresolved geometries;
                # only the strict result can satisfy final minimum validation.
                g, _ = self.correction(self.pu, self.syms, pos, self.q, spin=self.mult,
                                       return_info=True, imag_as_soft=True)
            self.corr[k] = g
        return self.corr[k]

    def _totals(self, uniq, override=None):
        order = sorted(range(len(uniq.G)), key=lambda j: uniq.G[j])
        win = [j for j in order if uniq.G[j] - uniq.G[order[0]] < self.window][:self.maximum]
        if override is not None and override[0] not in win:
            win.append(override[0])
        c = {}
        for j in win:
            c[j] = override[1] if (override is not None and j == override[0]) else self._ensure(uniq, j)
        ref = c[order[0]]
        return [uniq.G[j] + c.get(j, ref) for j in range(len(uniq.G))], c, order, win

    def update(self, uniq):
        tot, _, _, _ = self._totals(uniq)
        return self.combine(tot)

    def final(self, uniq, also, therm_valid, j_valid, *, validated=None):
        """therm_valid = validated correction of minimum j_valid (the structure _thermal_at_minimum accepted)."""
        if validated is None or set(validated) != set(range(len(uniq.G))):
            raise ValueError("final thermal ensemble requires validation of every sampled minimum")
        if not all(np.isfinite(g) for g in validated.values()):
            raise ValueError("nonfinite validated thermal correction")
        c = validated
        order = sorted(c, key=lambda j: uniq.G[j])
        win = order
        tot = [value + c[j] for j, value in enumerate(uniq.G)]
        ref = c[order[0]]
        also_tot = {m: [None if v is None else v + c.get(j, ref) for j, v in enumerate(vals)]
                    for m, vals in also.items()}
        spread = [c[j] - c[order[0]] for j in win]
        info = {"n_window": len(win), "n_hessians": self.n_hessians, "hessian_count_scope": "provisional tracker calls only",
                "tracked_in_convergence": False, "provisional_thermal_tracked": True,
                "validated_index_is_lowest": bool(j_valid == order[0]),
                "gcorr_spread_kj": round(float(max(spread) - min(spread)), 2) if spread else 0.0}
        info.update(_coverage_info(uniq.G, c, self.window, self.maximum))
        return self.combine(tot), also_tot, info


def thermal_ensemble(pu, uniq, also, therm_ref, q, mult, template, *,
                     correction, combine, window, maximum, imaginary_tolerance):
    """Strict compatibility evaluator; unresolved/capped ensembles require the pipeline resolver."""
    if len(uniq.G) > maximum:
        raise ValueError("thermal validation budget exceeded")
    syms = [a.GetSymbol() for a in template.GetAtoms()]
    order = sorted(range(len(uniq.G)), key=lambda j: uniq.G[j])
    win = order
    corr = {j: therm_ref for j in range(len(uniq.G))}
    n_imag = 0
    for j in win:
        pos = uniq.ref[j].GetConformer().GetPositions()
        g, info = correction(pu, syms, pos, q, spin=mult, return_info=True)
        if (not np.isfinite(g) or info.get("n_imag", 0)
                or info.get("max_imag_cm", 0.0) > imaginary_tolerance):
            raise ValueError("unresolved thermal minimum requires geometry resolution")
        corr[j] = g
        n_imag += int(info.get("max_imag_cm", 0.0) > imaginary_tolerance)
    G_tot = [uniq.G[j] + corr[j] for j in range(len(uniq.G))]
    also_tot = {m: [None if v is None else v + corr[j] for j, v in enumerate(vals)] for m, vals in also.items()}
    spread = [corr[j] - corr[order[0]] for j in win]
    info = {"n_hessians": len(win), "n_imag_floored": n_imag,
            "gcorr_spread_kj": round(float(max(spread) - min(spread)), 2) if spread else 0.0}
    info.update(_coverage_info(uniq.G, win, window, maximum))
    return combine(G_tot), also_tot, info
