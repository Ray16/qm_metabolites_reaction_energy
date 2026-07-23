"""Load reference data: ModelSEED reactions and the existing-method Delta_r G values.

Used to (a) select reactions whose every species we have QM energies for, and
(b) benchmark QM Delta_r G'^o against ModelSEED GCM, eQuilibrator, and the
experimental (openTECR/TECRDB) values already shipped in the submodule tables.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

from . import config
from .reactions import PROTON_ID, Reaction

# ModelSEED `deltag` and the eQuilibrator MetaNetX_Reaction_Energies.tbl store
# energies in KCAL/MOL (the retrieval script does `.to('kilocal / mole')`, and
# kcal is ModelSEED's house convention). Confirmed empirically: across 227
# TECRDB-matched reactions, col2 * 4.184 ~= experimental dG (e.g. rxn02459
# 7.48 kcal -> 31.3 kJ vs 34.7 kJ measured). We convert to kJ/mol on read so the
# whole pipeline is kJ/mol (QM and TECRDB are already kJ/mol). The raw submodule
# files are left untouched. 1 kcal_th = 4.184 kJ exactly.
KCAL_TO_KJ = 4.184


@dataclass(frozen=True)
class ReferenceReaction:
    """A ModelSEED reaction plus the Delta_r G values from existing methods."""

    reaction: Reaction
    mseed_dG_kJ: float | None       # ModelSEED Delta_r G'^o, converted kcal->kJ
    mseed_dGerr_kJ: float | None


def load_modelseed_reactions(
    db_dir: str | None = None,
) -> dict[str, ReferenceReaction]:
    """Parse all reaction_*.json shards into ReferenceReaction objects.

    The stoichiometry keeps H+ out (the transform handles pH); water is retained.
    ModelSEED `deltag` is the transformed Delta_r G'^o in KCAL/MOL (eQuilibrator
    component-contribution, copied from MetaNetX_Reaction_Energies.tbl col2), with
    sentinel 10000000 for "unknown". We multiply by KCAL_TO_KJ to return kJ/mol.
    """
    db_dir = db_dir or os.path.join(
        config.PROJECT_DIR, "ModelSEEDDatabase", "Biochemistry"
    )
    out: dict[str, ReferenceReaction] = {}
    for path in sorted(glob.glob(os.path.join(db_dir, "reaction_*.json"))):
        with open(path) as fh:
            for rec in json.load(fh):
                # Keep only genuine, single-compartment, mass/charge-balanced
                # chemical reactions. status "OK" => balanced; transport and
                # multi-compartment reactions are not chemical transformations.
                if rec.get("status") != "OK" or rec.get("is_transport"):
                    continue
                if {s["compartment"] for s in rec["stoichiometry"]} - {0}:
                    continue
                stoich: dict[str, float] = {}
                for s in rec["stoichiometry"]:
                    cpd = s["compound"]
                    if cpd == PROTON_ID:
                        continue
                    stoich[cpd] = stoich.get(cpd, 0.0) + s["coefficient"]
                # Drop species that net to zero (appear on both sides).
                stoich = {c: v for c, v in stoich.items() if abs(v) > 1e-9}
                if not stoich:
                    continue
                dG = _clean_dg(rec.get("deltag"))
                dGerr = _clean_dg(rec.get("deltagerr"))
                if dG is not None:
                    dG *= KCAL_TO_KJ          # kcal/mol -> kJ/mol
                if dGerr is not None:
                    dGerr *= KCAL_TO_KJ
                out[rec["id"]] = ReferenceReaction(
                    reaction=Reaction(rec["id"], stoich),
                    mseed_dG_kJ=dG,
                    mseed_dGerr_kJ=dGerr,
                )
    return out


def _clean_dg(value) -> float | None:
    """ModelSEED uses 10000000 as the 'no value' sentinel."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if abs(v) >= 1e6 else v


def load_equilibrator_reaction_dG(
    tbl_path: str = config.EQUILIBRATOR_RXN_TBL,
) -> dict[str, dict]:
    """Parse eQuilibrator MetaNetX_Reaction_Energies.tbl.

    Columns (verified against the generating script and training_data.py):
      col2 = eQuilibrator standard_dg_prime PREDICTION, in KCAL/MOL
      col3 = its uncertainty, KCAL/MOL
      col4 = ln_reversibility_index (dimensionless) -- NOT experimental dG.
    We convert col2/col3 to kJ/mol. col4 is NOT an energy and is NOT returned as
    experiment; real experimental dG comes from the TECRDB extraction, not here.
    Returns {rxn_id: {"eq_dG_kJ", "eq_err_kJ"}}.
    """
    out: dict[str, dict] = {}
    if not os.path.isfile(tbl_path):
        return out
    with open(tbl_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            rxn_id = parts[0]
            try:
                eq_dG = float(parts[1]) * KCAL_TO_KJ   # kcal/mol -> kJ/mol
                eq_err = float(parts[2]) * KCAL_TO_KJ
            except ValueError:
                continue
            out[rxn_id] = {
                "eq_dG_kJ": eq_dG,
                "eq_err_kJ": eq_err,
            }
    return out


def reactions_within(
    compound_ids: set[str], db_dir: str | None = None
) -> dict[str, ReferenceReaction]:
    """Subset of ModelSEED reactions whose every (non-proton) species is available."""
    refs = load_modelseed_reactions(db_dir)
    return {
        rid: rr
        for rid, rr in refs.items()
        if rr.reaction.compounds() <= compound_ids
    }
