"""Training-exposure audits independent of prediction errors and QM backends.

Inputs must already use the baseline's documented protonation normalization.
Hydrogen drawing and optional stereochemistry are canonicalized here; water
is retained. A passed audit establishes the specified split, not independent
validation of an estimator developed against the same experimental database.
"""
from functools import lru_cache
from fractions import Fraction
import math

import numpy as np
from rdkit import Chem


@lru_cache(maxsize=16384)
def compound_identity(smiles, stereo_sensitive=True):
    """Canonical audit identity, including explicitly recorded generic fragments.

    A wildcard can occur in a baseline's training data. Retaining its identity
    supports an exposure audit; it does not make that structure QM-scoreable.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unparseable compound structure: {smiles!r}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    mol = Chem.RemoveHs(mol)
    if not stereo_sensitive:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def reaction_signature(pairs, *, stereo_sensitive=True, omit_proton=True):
    """Exact extent/direction-invariant signature; cancellation precedes scaling.

    ``pairs`` contains (normalized SMILES, signed coefficient). Formation and
    reaction targets must remain separately typed by the caller.
    """
    counts = {}
    for smiles, coefficient in pairs:
        if isinstance(coefficient, bool) or not math.isfinite(float(coefficient)):
            raise ValueError("Stoichiometric coefficients must be finite numbers")
        key = compound_identity(smiles, stereo_sensitive)
        if omit_proton and key == "[H+]":
            continue
        counts[key] = counts.get(key, Fraction(0)) + Fraction(str(coefficient))
    values = sorted((key, value) for key, value in counts.items() if value)
    if not values:
        return ()
    scale = abs(values[0][1])
    forward = tuple((key, value / scale) for key, value in values)
    reverse = tuple((key, -value) for key, value in forward)
    return min(forward, reverse)


def _training_index(training):
    index = {}
    for record in training:
        key = record["row"]
        if key in index:
            raise ValueError(f"Duplicate training row {key!r}")
        index[key] = record
    return index


def audit_holdout(training, targets, train_rows, *, heldout_compounds=(),
                  heldout_families=(), stereo_sensitive=False):
    """Check equivalent reactions, family exposure and every compound target.

    ``targets`` can supply signatures and/or audited equivalent training row
    IDs. Compound exposure includes formation targets. Removing stereochemistry
    is conservative for baselines whose active input encoder omits it.
    """
    index = _training_index(training)
    train_rows = list(train_rows)
    if len(set(train_rows)) != len(train_rows):
        raise ValueError("Duplicate row IDs in training split")
    missing = set(train_rows) - index.keys()
    if missing:
        raise ValueError(f"Unknown training rows: {sorted(missing)}")
    kept = set(train_rows)
    held = {compound_identity(s, stereo_sensitive) for s in heldout_compounds}
    families = set(heldout_families)
    signatures = {}
    for row in train_rows:
        item = index[row]
        if item["type"] == "reaction dg":
            key = reaction_signature(item["signature"], stereo_sensitive=stereo_sensitive)
            if key:
                signatures.setdefault(key, []).append(row)
    violations = []
    exposure = []
    collisions = []
    seen = {compound_identity(s, stereo_sensitive)
            for row in train_rows for s in index[row]["compounds"]}
    for target in targets:
        equivalent = set(target.get("published_training_rows", ()))
        if not stereo_sensitive:
            equivalent.update(target.get("encoder_stereo_equivalent_rows", ()))
        if "signature" in target:
            key = reaction_signature(target["signature"], stereo_sensitive=stereo_sensitive)
            if key:
                equivalent.update(signatures.get(key, ()))
            elif reaction_signature(target["signature"], stereo_sensitive=True):
                collisions.append({"target": target["id"], "kind": "encoder_collapsed_zero",
                                   "scope": "representation limitation; does not establish energy interpolation"})
        leaking = sorted(equivalent & kept)
        if leaking:
            violations.append({"kind": "equivalent_reaction", "target": target["id"], "rows": leaking})
        compounds = {compound_identity(s, stereo_sensitive)
                     for s in target.get("normalized_compounds", ())}
        exposure.append({"target": target["id"], "n_compounds": len(compounds),
                         "n_compounds_seen": len(compounds & seen),
                         "all_compounds_seen": bool(compounds) and compounds <= seen})
    for row in train_rows:
        item = index[row]
        overlap = held & {compound_identity(s, stereo_sensitive) for s in item["compounds"]}
        if overlap:
            violations.append({"kind": "heldout_compound", "row": row,
                               "target_type": item["type"], "compounds": sorted(overlap)})
        overlap_family = families & set(item.get("families", ()))
        if overlap_family:
            violations.append({"kind": "heldout_family", "row": row,
                               "families": sorted(overlap_family)})
    return {"passed": not violations, "violations": violations,
            "n_training_reactions": sum(index[row]["type"] == "reaction dg" for row in train_rows),
            "n_training_formations": sum(index[row]["type"] == "formation dg" for row in train_rows),
            "compound_exposure": exposure, "stereo_sensitive": stereo_sensitive,
            "target_representation_collisions": collisions,
            "generic_structure_training_rows": [row for row in train_rows
                if any("*" in compound_identity(s) for s in index[row]["compounds"])],
            "scope": "split integrity only; prior estimator-development exposure is not removed"}


def training_span_exposure(training, train_rows, target_pairs, *, stereo_sensitive=False):
    """Does training stoichiometry constrain a target by Hess-law combinations?

    No energies enter this calculation. Membership is interpolation information,
    not automatically leakage: a reaction holdout can remain constrained by
    retained formation targets or other reaction targets.
    """
    index = _training_index(training)
    rows = list(train_rows)
    if len(rows) != len(set(rows)) or set(rows) - index.keys():
        raise ValueError("Training rows must be unique and known")
    target_pairs = list(target_pairs)
    physical_target = reaction_signature(target_pairs, stereo_sensitive=True)
    if not physical_target:
        raise ValueError("Target has no physical net stoichiometry")
    columns = [reaction_signature(index[row]["signature"], stereo_sensitive=stereo_sensitive)
               for row in rows]
    target = reaction_signature(target_pairs, stereo_sensitive=stereo_sensitive)
    if not target:
        return {"status": "encoder_collapsed_zero", "representation_collision": True,
                "in_training_stoichiometric_span": True, "relative_projection_residual": 0.0,
                "rank": None, "n_training_targets": len(columns),
                "scope": "mathematical zero in collapsed representation; not evidence of chemical interpolation"}
    keys = sorted({key for column in columns + [target] for key, _ in column})
    lookup = {key: i for i, key in enumerate(keys)}
    matrix = np.zeros((len(keys), len(columns)))
    vector = np.zeros(len(keys))
    for j, column in enumerate(columns):
        for key, value in column:
            matrix[lookup[key], j] = float(value)
    for key, value in target:
        vector[lookup[key]] = float(value)
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, vector, rcond=None)
    relative_residual = float(np.linalg.norm(vector - matrix @ coefficients) / np.linalg.norm(vector))
    return {"in_training_stoichiometric_span": relative_residual <= 1e-8,
            "representation_collision": False,
            "relative_projection_residual": relative_residual, "rank": int(rank),
            "n_training_targets": len(columns),
            "scope": "stoichiometric constraint only; no energy labels or prediction errors used"}
