"""TECRDB dataset loaders (json only, no rdkit)."""
import json
import os

import torch

from . import paths


def load_tecrdb():
    """Return (reactions, metabolites, ensemble_qm, targets, rxn_ids)."""
    rxns = json.load(open(f"{paths.PIPE}/tecrdb_full_reactions.json"))
    mets = json.load(open(f"{paths.PIPE}/tecrdb_full_metabolites.json"))
    ens = json.load(open(f"{paths.PIPE}/ensemble_tecrdb_full.json"))
    tgt = json.load(open(f"{paths.PIPE}/tecrdb_full_experiment.json"))
    rxn_ids = [r for r in rxns if r in tgt]
    return rxns, mets, ens, tgt, rxn_ids


def stoich_matrix(rxns, rxn_ids, cidx):
    S = torch.zeros(len(rxn_ids), len(cidx))
    for i, r in enumerate(rxn_ids):
        for c, v in rxns[r].items():
            if c in cidx:
                S[i, cidx[c]] = v
    return S


def targets(tgt, rxn_ids):
    """(y, n, sd) per reaction. sd_kJ is None for singly-measured reactions in
    the source json; those are floored to the global median sd (~the TECRDB noise
    scale) so inverse-variance weighting is defined everywhere."""
    y = torch.tensor([tgt[r]["dG_kJ"] for r in rxn_ids], dtype=torch.float32)
    n = torch.tensor([tgt[r].get("n", 1) for r in rxn_ids], dtype=torch.float32)
    raw = [tgt[r].get("sd_kJ") for r in rxn_ids]
    known = [s for s in raw if s is not None and s > 0]
    floor = float(sorted(known)[len(known) // 2]) if known else 6.0
    sd = torch.tensor([s if (s is not None and s > 0) else floor for s in raw],
                      dtype=torch.float32)
    return y, n, sd


def load_conditions(rxn_ids, artifact_path):
    """Per-reaction measurement conditions [pH, ionic_strength, T(scaled), pMg],
    aligned to rxn_ids. Missing values imputed to biochemical-standard references
    (pH 7, I=0.25 M, T=298.15 K, pMg=14 ~ no added Mg). T is centered/scaled so
    the block is O(1). Returns a float tensor (len(rxn_ids), 4)."""
    cond = json.load(open(artifact_path))
    REF = {"p_h": 7.0, "ionic_strength": 0.25, "temperature": 298.15, "p_mg": 14.0}
    rows = []
    for r in rxn_ids:
        c = cond.get(r, {})
        ph = c.get("p_h"); ph = REF["p_h"] if ph is None else ph
        io = c.get("ionic_strength"); io = REF["ionic_strength"] if io is None else io
        t = c.get("temperature"); t = REF["temperature"] if t is None else t
        mg = c.get("p_mg"); mg = REF["p_mg"] if mg is None else mg
        rows.append([(ph - 7.0), io, (t - 298.15) / 25.0, (mg - 14.0) / 7.0])
    return torch.tensor(rows, dtype=torch.float32)
