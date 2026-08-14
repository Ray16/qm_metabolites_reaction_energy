#!/usr/bin/env python
"""Step 1: UMA gas-phase reaction ΔE on CLEAN ISOMERIZATIONS (same formula both
sides, charge conserved -> solvation ~cancels, so gas-phase ΔE ≈ aqueous ΔG'°).

This is the first accuracy sanity check on the engine.  Two things it measures:
  (a) is UMA gas-phase ΔE close to experimental ΔG'° when solvation cancels?
  (b) how large is CONFORMER NOISE -- the killer of the prior xTB work (±50 kJ
      on big cofactors)?  We use a real conformer ensemble per species and report
      the spread, so we know whether the signal survives.

Per species: ETKDG conformers -> MMFF pre-tidy -> UMA geometry-opt each ->
Boltzmann-weighted energy (and min, for reference).  ΔE = E(product)-E(reactant).
Energies cached per compound (shared species computed once).

Curated set (high experimental n, spans neutral sugars -> phosphate anions ->
citrate TRIANION), all from TECRDB with measured ΔG'°.

Run (uma env):  CUDA_VISIBLE_DEVICES=1 python scripts/step1_isomerization_dE.py
Quick:          ... --nconf 4 --reactions rxn00223,rxn00558
"""
import argparse
import json
import os
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.optimize import BFGS

from fairchem.core import pretrained_mlip, FAIRChemCalculator

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "artifacts")
os.makedirs(OUT, exist_ok=True)
PIPE = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/pipeline"
EV2KJ = 96.485  # eV -> kJ/mol
KT_KJ = 2.4789  # RT at 298.15 K, kJ/mol

# curated clean isomerizations (reactant -1, product +1, same formula)
CURATED = ["rxn00223",  # glucose -> fructose        n=89  neutral C6H12O6
           "rxn00636",  # mannose -> fructose        n=8   neutral
           "rxn00558",  # G6P -> F6P                 n=50  phosphate anion
           "rxn01106",  # 2-phosphoglycerate -> 3-PG n=72  phosphate anion
           "rxn00747",  # GAP -> DHAP                n=9   phosphate anion
           "rxn00973"]  # citrate -> isocitrate      n=75  TRIANION


def load_tecrdb():
    rxns = json.load(open(f"{PIPE}/tecrdb_full_reactions.json"))
    mets = {m["id"]: m for m in json.load(open(f"{PIPE}/tecrdb_full_metabolites.json"))}
    tgt = json.load(open(f"{PIPE}/tecrdb_full_experiment.json"))
    return rxns, mets, tgt


def conformers(smiles, nconf, seed=1):
    """ETKDG conformers + MMFF pre-tidy. Returns (list[Atoms], total_charge)."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError("rdkit parse failed")
    m = Chem.AddHs(m)
    params = AllChem.ETKDGv3(); params.randomSeed = seed; params.pruneRmsThresh = 0.3
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=nconf, params=params))
    if not cids:
        AllChem.EmbedMolecule(m, randomSeed=seed, useRandomCoords=True); cids = [0]
    try:
        AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=200)
    except Exception:
        pass
    q = Chem.GetFormalCharge(m)
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    out = []
    for cid in cids:
        pos = m.GetConformer(cid).GetPositions()
        out.append(Atoms(symbols=syms, positions=pos))
    return out, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nconf", type=int, default=12)
    ap.add_argument("--reactions", default=",".join(CURATED))
    a = ap.parse_args()
    rxn_ids = a.reactions.split(",")

    rxns, mets, tgt = load_tecrdb()
    print("loading uma-s-1p2 ...", flush=True)
    pu = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda")
    calc = FAIRChemCalculator(pu, task_name="omol")

    # per-compound absolute min-energy cache (kJ/mol) + spread
    cache = {}

    def energy_of(cid):
        if cid in cache:
            return cache[cid]
        smi = (mets.get(cid) or {}).get("smiles")
        confs, q = conformers(smi, a.nconf)
        Es = []
        for atoms in confs:
            atoms.info = {"charge": int(q), "spin": 1}
            atoms.calc = calc
            try:
                BFGS(atoms, logfile=None).run(fmax=0.05, steps=100)
                e = atoms.get_potential_energy() * EV2KJ
                if np.isfinite(e):
                    Es.append(e)
            except Exception:
                continue
        Es = np.array(sorted(Es))
        rel = Es - Es.min()
        w = np.exp(-rel / KT_KJ); w /= w.sum()
        e_boltz = float((w * Es).sum())
        cache[cid] = dict(e_min=float(Es.min()), e_boltz=e_boltz,
                          spread=float(rel.max()), n=len(Es), q=int(q))
        return cache[cid]

    rows = []
    t0 = time.time()
    for rid in rxn_ids:
        st = {c: v for c, v in rxns[rid].items() if abs(v) > 1e-9}
        (r, _), (p, _) = sorted(st.items(), key=lambda x: x[1])   # r:-1, p:+1
        er, ep = energy_of(r), energy_of(p)
        dE_min = ep["e_min"] - er["e_min"]
        dE_boltz = ep["e_boltz"] - er["e_boltz"]
        exp = tgt[rid]["dG_kJ"]; n = tgt[rid].get("n", 1)
        noise = max(er["spread"], ep["spread"])
        rows.append(dict(rxn=rid, reactant=r, product=p, charge=er["q"],
                         exp_dG_kJ=exp, n_meas=n,
                         uma_dE_min_kJ=round(dE_min, 2), uma_dE_boltz_kJ=round(dE_boltz, 2),
                         err_min=round(dE_min - exp, 2), err_boltz=round(dE_boltz - exp, 2),
                         conf_spread_kJ=round(noise, 1),
                         n_conf=(er["n"], ep["n"])))
        print(f"  {rid} q{er['q']:+d}  exp {exp:6.1f}  UMA(min) {dE_min:7.1f}  "
              f"UMA(boltz) {dE_boltz:7.1f}  |err_min| {abs(dE_min-exp):5.1f}  "
              f"conf-spread {noise:4.0f} kJ  ({mets[r]['name'][:14]}->{mets[p]['name'][:14]})",
              flush=True)

    err_min = np.array([r["err_min"] for r in rows])
    err_boltz = np.array([r["err_boltz"] for r in rows])
    print(f"\n==== UMA gas-phase ΔE vs experiment on {len(rows)} isomerizations ====")
    print(f"  MAE(min-conf)   {np.abs(err_min).mean():.1f} kJ/mol   sign {np.mean(np.sign([r['uma_dE_min_kJ'] for r in rows])==np.sign([r['exp_dG_kJ'] for r in rows]))*100:.0f}%")
    print(f"  MAE(Boltzmann)  {np.abs(err_boltz).mean():.1f} kJ/mol")
    print(f"  median conformer spread {np.median([r['conf_spread_kJ'] for r in rows]):.0f} kJ  "
          f"(the noise floor for this protocol)")
    print(f"  NOTE: experimental isomerization ΔG are small (|{np.abs([r['exp_dG_kJ'] for r in rows]).max():.0f}| kJ max); "
          "gas-phase ΔE omits the (small, ~cancelling) solvation+thermal terms.")
    print(f"  total {time.time()-t0:.0f}s")

    json.dump(dict(reactions=rows,
                   mae_min=float(np.abs(err_min).mean()),
                   mae_boltz=float(np.abs(err_boltz).mean()),
                   nconf=a.nconf),
              open(os.path.join(OUT, "step1_isomerization_dE.json"), "w"), indent=2)
    print("wrote artifacts/step1_isomerization_dE.json")


if __name__ == "__main__":
    main()
