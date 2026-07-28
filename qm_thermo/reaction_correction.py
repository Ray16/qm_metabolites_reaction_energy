"""Reaction-level empirical calibration with leakage-safe held-out scoring.

This is intentionally not part of :mod:`qm_thermo.reactions`: the latter is
the uncalibrated QM baseline.  A correction is learned only from experimental
reaction residuals and is therefore a separately labelled calibrated model.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CalibrationRow:
    reaction_id: str
    signature: str
    reaction_class: str
    predicted_kJ: float
    experimental_kJ: float

    @property
    def residual_kJ(self) -> float:
        return self.experimental_kJ - self.predicted_kJ


def canonical_signature(stoichiometry: dict[str, float]) -> str:
    """Direction-invariant reaction signature, so reverse pairs stay together."""
    forward = ";".join(f"{key}:{value:g}" for key, value in sorted(stoichiometry.items()))
    reverse = ";".join(f"{key}:{-value:g}" for key, value in sorted(stoichiometry.items()))
    return min(forward, reverse)


def _class_shift(rows: Iterable[CalibrationRow], target_class: str,
                 min_signatures: int, shrinkage: float) -> tuple[float, int]:
    """Estimate a shrunk class residual using independent reaction signatures."""
    rows = list(rows)
    class_rows = [row for row in rows if row.reaction_class == target_class]
    signatures = {row.signature for row in class_rows}
    if len(signatures) < min_signatures:
        return 0.0, len(signatures)
    # One number per signature avoids duplicated measurements/directions
    # receiving disproportionate weight.
    by_signature: dict[str, list[float]] = defaultdict(list)
    all_by_signature: dict[str, list[float]] = defaultdict(list)
    for row in class_rows:
        by_signature[row.signature].append(row.residual_kJ)
    for row in rows:
        all_by_signature[row.signature].append(row.residual_kJ)
    class_mean = sum(sum(values) / len(values) for values in by_signature.values()) / len(by_signature)
    global_mean = sum(sum(values) / len(values) for values in all_by_signature.values()) / len(all_by_signature)
    # Empirical-Bayes-like shrinkage protects a sparse class from receiving a
    # large arbitrary offset.  It never has per-metabolite terms.
    weight = len(signatures) / (len(signatures) + shrinkage)
    return weight * class_mean + (1.0 - weight) * global_mean, len(signatures)


def leave_signature_out(rows: Iterable[CalibrationRow], *, min_signatures: int = 4,
                        shrinkage: float = 3.0) -> list[dict]:
    """Out-of-fold class correction, holding every equivalent reaction out.

    A target receives no correction when its class has insufficient *other*
    signatures.  That policy makes this safe for the current ten-reaction set,
    which is a stress test rather than a calibration dataset.
    """
    rows = list(rows)
    output = []
    for row in rows:
        train = [candidate for candidate in rows if candidate.signature != row.signature]
        shift, n_signatures = _class_shift(train, row.reaction_class, min_signatures, shrinkage)
        output.append({
            "reaction_id": row.reaction_id,
            "signature": row.signature,
            "reaction_class": row.reaction_class,
            "predicted_kJ": row.predicted_kJ,
            "experimental_kJ": row.experimental_kJ,
            "correction_kJ": shift,
            "corrected_kJ": row.predicted_kJ + shift,
            "training_signatures_in_class": n_signatures,
            "calibrated": n_signatures >= min_signatures,
        })
    return output
