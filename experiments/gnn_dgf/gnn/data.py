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
    y = torch.tensor([tgt[r]["dG_kJ"] for r in rxn_ids], dtype=torch.float32)
    n = torch.tensor([tgt[r].get("n", 1) for r in rxn_ids], dtype=torch.float32)
    return y, n
