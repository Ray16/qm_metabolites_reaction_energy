"""Per-reaction uncertainty features (UQ_MODEL=features).

Every feature is a deterministic function of the reaction STRUCTURE (the pH-7 input species), the ROUTE
the pipeline took, the size of its pKa transform, and the conformer-sampling spread. No enzyme name, EC
number, note, or reaction id is read, so the scale transfers to poorly annotated reactions (ModelSEED).

The feature list, transforms and order are part of the fitted artifact (data/uq_feature_model.json);
changing anything here requires refitting (metag.tools.fit_uq).
"""
import math

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

FEATURE_VERSION = "2026-10-02a"

_P = Chem.MolFromSmarts("[PX4]")
_SMARTS = {
    "d_POP": "[PX4]-O-[PX4]",
    "d_PN": "[#7]-[PX4]",
    "d_carbonyl": "[CX3](=[OX1])([#6,#1])[#6,#1]",
    "d_aldehyde": "[CX3H1](=O)[#6]",
    "d_ketoacid": "[CX3](=[OX1])[CX3](=O)[OX2H1,OX1-]",
    "d_hemiacetal": "[OX2;R][CX4;R][OX2H1]",
    "d_amine": "[NX3,NX4+;!$(NC=O);!$(N-a);!$(N=*)][CX4]",
    "d_aromN": "[n]",
    "d_guan": "[NX3][CX3](=[NX2,NX3+])[NX3]",
    "d_amide": "[CX3](=[OX1])[NX3]",
    "d_cationN": "[#7+;!H0]",
}
_PATT = {k: Chem.MolFromSmarts(v) for k, v in _SMARTS.items()}
_ANION_O = Chem.MolFromSmarts("[OX1-]")

# order matters: it is the column order of the fitted model
FEATURES = [
    "n_species", "max_rotb", "sum_heavy", "max_heavy", "gross_anionO", "gross_charge",
    "net_abs_charge_change", "max_abs_charge", "abs_pka_transform", "d_PN", "d_POP", "n_multiP_species",
    "max_P_per_species", "d_hemiacetal", "d_cationN", "d_aromN", "d_amine", "d_amide", "d_guan",
    "d_carbonyl", "d_ketoacid", "d_aldehyde", "U_samp", "truncated", "ntp_core", "cofactor_ring",
    "prefer_full", "ph0", "scored_max_heavy",
]
LOG1P = {"sum_heavy", "max_heavy", "max_rotb", "gross_anionO", "gross_charge", "abs_pka_transform",
         "scored_max_heavy", "U_samp"}


def _mols(species):
    out = []
    for c, q, s in species.values():
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out.append((float(c), float(q), m))
    return out


def reaction_features(orig_species, scored_species, routes, stages, U_samp):
    """Raw (untransformed) feature dict. orig_species / scored_species: {name: [coeff, charge, smiles]}."""
    mols = _mols(orig_species)
    heavy = [m.GetNumHeavyAtoms() for _, _, m in mols]
    nP = [len(m.GetSubstructMatches(_P)) for _, _, m in mols]
    f = {
        "n_species": len(mols),
        "max_rotb": max([rdMolDescriptors.CalcNumRotatableBonds(m) for _, _, m in mols] or [0]),
        "sum_heavy": sum(abs(c) * h for (c, _, _), h in zip(mols, heavy)),
        "max_heavy": max(heavy or [0]),
        "gross_anionO": sum(abs(c) * len(m.GetSubstructMatches(_ANION_O)) for c, _, m in mols),
        "gross_charge": sum(abs(c * q) for c, q, _ in mols),
        "net_abs_charge_change": abs(sum(c * q for c, q, _ in mols)),
        "max_abs_charge": max([abs(q) for _, q, _ in mols] or [0]),
        "abs_pka_transform": abs(float((stages or {}).get("pka_transform", 0.0) or 0.0)),
        "n_multiP_species": sum(1 for p in nP if p >= 2),
        "max_P_per_species": max(nP or [0]),
        "U_samp": float(U_samp or 0.0),
    }
    for k, patt in _PATT.items():
        f[k] = abs(sum(c * len(m.GetSubstructMatches(patt)) for c, _, m in mols))
    routes = routes or {}
    for k in ("truncated", "ntp_core", "cofactor_ring", "prefer_full", "ph0"):
        f[k] = float(bool(routes.get(k)))
    scored = _mols(scored_species or {})
    f["scored_max_heavy"] = max([m.GetNumHeavyAtoms() for _, _, m in scored] or [0])
    return f


def transform(raw):
    """Model-space vector (list, FEATURES order); None entries stay None for median imputation."""
    v = []
    for k in FEATURES:
        x = raw.get(k)
        if x is None:
            v.append(None)
        else:
            x = float(x)
            v.append(math.log1p(x) if k in LOG1P else x)
    return v
