"""Inference for the optional acid site-pKa model.

Parameters are calibrated to independent pKa data, not reaction energies.
This empirical model has a restricted chemistry domain; unsupported groups
raise rather than acquiring an unrelated site's parameters. Training lives
in metag.tools.fit_site_pka and is not imported by production routing.
"""
import functools
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType

import numpy as np
from rdkit import Chem


from metag.routing.pka_domain import UnsupportedSiteError, validate_phosphoryl_domain


CLASSES = ["carb_ammonium", "carb_amine", "carb_plain", "carb_oxo", "carb_oxygen", "carb_Oacid", "carb_vinyl", "carb_arom", "carb_formate",
           "Pi_1", "Pi_2", "Pi_3", "mono_1", "mono_2", "anh_1", "anh_2", "internal", "ppi_1", "ppi_2", "ppi_3",
           "ppi_4", "sulf", "sulfate_1", "sulfate_2"]
EXTRA = ["beta_oxo", "beta_oxygen", "coulomb_c"]


def load_site_model():
    """Load the selected artifact strictly; fingerprint the actual bytes used.

    Read on each selection so replacing a file in a long-lived process cannot
    silently retain old parameters while reporting a new artifact fingerprint.
    Parsing is cached by content; callers receive immutable parameters.
    """
    default = Path(__file__).resolve().parents[1] / "data" / "site_pka_model.json"
    path = Path(os.environ.get("PKA_SITE_MODEL_FILE", str(default)))
    try:
        return _parse_model(path.read_bytes())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"cannot load requested site pKa model {path}: {exc}") from exc


@functools.lru_cache(maxsize=8)
def _parse_model(content):
    params = json.loads(content)["params"]
    if not isinstance(params, dict):
        raise ValueError("params must be a mapping")
    for name in CLASSES + EXTRA:
        value = params.get(name)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"missing or nonfinite parameter {name}")
    if params["coulomb_c"] < 0:
        raise ValueError("coulomb_c must be nonnegative")
    return MappingProxyType(params), hashlib.sha256(content).hexdigest()


def site_features(mol, sites):
    """sites = [(o_idx, cls_group)] from pka_transform._anion_sites. Returns list of dicts:
    {"atom": acidic heavy atom idx (C of carboxyl / P / S), "cls": class name, "beta_oxo": 0/1,
     "beta_oxygen": 0/1}. Phosphate/sulfate O's on the same centre get ladder positions 1..k."""
    out = []
    groups = {}
    for o, grp in sites:
        if grp not in {"carboxyl", "sulfonate", "sulfate", "phosphate"}:
            raise UnsupportedSiteError(f"site pKa model does not support {grp}; use PKA_MODEL=table")
        oa = mol.GetAtomWithIdx(o)
        centre = next((n for n in oa.GetNeighbors() if n.GetSymbol() in ("C", "P", "S")), None)
        if centre is None:
            raise UnsupportedSiteError(f"no acid centre for {grp} at atom {o}")
        groups.setdefault((grp, centre.GetIdx()), []).append(o)
    zs = [a.GetAtomicNum() for a in mol.GetAtoms()]
    free_ppi = zs.count(15) == 2 and all(z in (8, 15) for z in zs)
    ppi_count = 0
    for (grp, c), os_ in sorted(groups.items(), key=lambda kv: kv[0][1]):
        ca = mol.GetAtomWithIdx(c)
        if grp == "carboxyl":
            if all(n.GetSymbol() == "O" for n in ca.GetNeighbors()):
                raise UnsupportedSiteError("site pKa model does not support carbonate; use PKA_MODEL=table")
            for o in os_:
                cls, bo, box = _carboxyl_class(mol, ca)
                out.append({"o": o, "atom": c, "cls": cls, "beta_oxo": bo, "beta_oxygen": box})
        elif grp == "sulfonate":
            out += [{"o": o, "atom": c, "cls": "sulf", "beta_oxo": 0, "beta_oxygen": 0} for o in os_]
        elif grp == "sulfate":
            if len(os_) > 2:
                raise UnsupportedSiteError("sulfate has more sites than model parameters")
            out += [{"o": o, "atom": c, "cls": f"sulfate_{k + 1}", "beta_oxo": 0, "beta_oxygen": 0}
                    for k, o in enumerate(sorted(os_))]
        elif grp == "phosphate":
            validate_phosphoryl_domain(mol, ca)
            if any(n.GetSymbol() == "N" for n in ca.GetNeighbors()):
                raise UnsupportedSiteError("site pKa model does not support phosphoramidate; independent state-specific speciation data are required")
            if any(x.GetSymbol() == "C" and any(
                    y.GetSymbol() == "O" and mol.GetBondBetweenAtoms(x.GetIdx(), y.GetIdx()).GetBondTypeAsDouble() == 2
                    for y in x.GetNeighbors())
                   for n in ca.GetNeighbors() if n.GetSymbol() == "O" and n.GetDegree() >= 2
                   for x in n.GetNeighbors()):
                raise UnsupportedSiteError("site pKa model does not support acyl phosphate; use PKA_MODEL=table")
            n_bridge = sum(1 for n in ca.GetNeighbors() if n.GetSymbol() == "O" and n.GetDegree() >= 2)
            to_p = any(n2.GetSymbol() == "P" and n2.GetIdx() != c
                       for n in ca.GetNeighbors() if n.GetSymbol() == "O" for n2 in n.GetNeighbors())
            if len(os_) > 3 - n_bridge:
                raise UnsupportedSiteError("phosphate has more sites than model parameters")
            for k, o in enumerate(sorted(os_)):
                if free_ppi:
                    ppi_count += 1
                    if ppi_count > 4:
                        raise UnsupportedSiteError("pyrophosphate has more sites than model parameters")
                    cls = f"ppi_{ppi_count}"
                elif n_bridge == 0:
                    cls = f"Pi_{k + 1}"
                elif n_bridge == 1:
                    cls = (f"anh_{k + 1}" if to_p else f"mono_{k + 1}")
                else:
                    cls = "internal"
                out.append({"o": o, "atom": c, "cls": cls, "beta_oxo": 0, "beta_oxygen": 0})
    return out


def _carboxyl_class(mol, c):
    alpha = [n for n in c.GetNeighbors() if n.GetSymbol() == "C"]
    if not alpha:
        return "carb_formate", 0, 0
    a = alpha[0]
    cls = "carb_plain"
    if a.GetHybridization() == Chem.HybridizationType.SP3:
        for nb in a.GetNeighbors():                           # alpha-N: amino acid (textbook, fixed classes)
            if nb.GetSymbol() != "N":
                continue
            if nb.GetFormalCharge() == 1 and nb.GetTotalNumHs() >= 1:
                return "carb_ammonium", 0, 0                  # N stays protonated in the QM species
            amide = any(x.GetSymbol() == "C" and any(
                mol.GetBondBetweenAtoms(x.GetIdx(), y.GetIdx()).GetBondTypeAsDouble() == 2 and y.GetSymbol() in "ONS"
                for y in x.GetNeighbors()) for x in nb.GetNeighbors() if x.GetIdx() != a.GetIdx())
            if nb.GetFormalCharge() == 0 and nb.GetTotalNumHs() >= 1 and not nb.GetIsAromatic() and not amide:
                return "carb_amine", 0, 0                     # neutral NH2 in the QM species: microscopic pKa
    if a.GetIsAromatic():
        cls = "carb_arom"
    elif any(nb.GetSymbol() == "O" and any(x.GetSymbol() in ("P", "S") for x in nb.GetNeighbors())
             for nb in a.GetNeighbors()):
        cls = "carb_Oacid"                                   # alpha-O-phosphate/sulfate (2-PG, PEP): 3.5
    else:
        for nb in a.GetNeighbors():
            if nb.GetIdx() == c.GetIdx():
                continue
            b = mol.GetBondBetweenAtoms(a.GetIdx(), nb.GetIdx()).GetBondTypeAsDouble()
            if nb.GetSymbol() == "O" and b == 2:
                cls = "carb_oxo"; break
            if nb.GetSymbol() == "C" and b == 2:
                cls = "carb_vinyl"
        if cls == "carb_plain":
            for nb in a.GetNeighbors():
                if nb.GetSymbol() == "O" and nb.GetIdx() != c.GetIdx():
                    acid_ester = any(x.GetSymbol() in ("P", "S") for x in nb.GetNeighbors())
                    cls = "carb_Oacid" if acid_ester else "carb_oxygen"
                    if acid_ester:
                        break
    bo = box = 0
    if cls in ("carb_plain", "carb_oxygen", "carb_Oacid"):
        for nb in a.GetNeighbors():
            if nb.GetIdx() == c.GetIdx() or nb.GetSymbol() != "C":
                continue
            for x in nb.GetNeighbors():
                if x.GetIdx() == a.GetIdx():
                    continue
                bt = mol.GetBondBetweenAtoms(nb.GetIdx(), x.GetIdx()).GetBondTypeAsDouble()
                if x.GetSymbol() == "O" and bt == 2 and not any(
                        y.GetSymbol() == "O" and y.GetIdx() != x.GetIdx() and y.GetFormalCharge() < 0
                        for y in nb.GetNeighbors()):
                    bo = 1                                          # beta ketone/aldehyde (not carboxylate)
                elif x.GetSymbol() == "O" and bt == 1:
                    box = 1
    return cls, bo, box


def effective_pkas(mol, feats, params):
    """Sequential microscopic pKa's: intrinsic = base[cls] + beta terms; deprotonate in increasing order,
    each raised by coulomb_c / d for every already-deprotonated site."""
    if not feats:
        return []
    dm = Chem.GetDistanceMatrix(mol)
    intr = [params[f["cls"]] + params["beta_oxo"] * f["beta_oxo"] + params["beta_oxygen"] * f["beta_oxygen"]
            for f in feats]
    order = np.argsort(intr)
    done, eff = [], [0.0] * len(feats)
    for j in order:
        # coupling only BETWEEN CARBOXYLATES (what it is calibrated on). Experimental phosphate / PPi /
        # sulfate ladders are macroscopic values that already contain their intra-molecular repulsion
        # (ATP 7.60, PPi 0.9/2.1/6.7/9.3), so adding it there would double-count.
        if not feats[j]["cls"].startswith("carb_"):
            eff[j] = intr[j]; done.append(j); continue
        shift = sum(params["coulomb_c"] / max(dm[feats[j]["atom"]][feats[i]["atom"]], 1.0)
                    for i in done if feats[i]["cls"].startswith("carb_"))
        eff[j] = intr[j] + shift
        done.append(j)
    return eff
