#!/usr/bin/env python
"""Score the full TECRDB set and test the pre-registered group-correction claims.

Written before the numbers existed, so the analysis is fixed in advance. This
project has retracted three headline figures (27.7, 7.4, the species-defect
hypothesis) that came from choosing the analysis after seeing the data.

Pre-registered, from the 130-reaction preview:

  P1  absolute QM MAE lands at 30-40 kJ/mol.            (expected; no news)
  P2  mean signed error at d(P-O-P) = -1 is +64 +/- 15. (tests the one bond
      correction we believe in)
  P3  no group other than phosphoanhydride shows |bias| > 15 kJ/mol.
  P4  isodesmic coverage at fingerprint cosine >= 0.90 exceeds 50%, against
      29% on the 130. **If P4 fails the reference-network strategy is dead**
      and should be abandoned rather than tuned.

Group effects are fitted **jointly**, not one at a time: scanning groups
singly on the 130 produced three false leads out of four (carboxylate,
thioester and thiol all evaporated once phosphoanhydride was controlled for).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
sys.path.insert(0, THERMO)

from qm_thermo import config                                          # noqa: E402
from qm_thermo.composite import extract_ensemble_energy               # noqa: E402
from qm_thermo.reactions import Reaction, SpeciesInfo, reaction_dG    # noqa: E402

GROUPS = {
    "phosphoanhydride": "[P]~[O]~[P]",
    "phosphate ester": "[P](~[O])(~[O])~[OX2]~[#6]",
    "thioester": "[CX3](=O)[SX2]",
    "carboxylate": "[CX3](=[OX1])[OX1H0-,OX2H1]",
    "carboxylic ester": "[CX3](=[OX1])[OX2][#6]",
    "amide": "[CX3](=[OX1])[NX3]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=[OX1])[#6]",
    "alcohol": "[OX2H][#6]",
    "acetal/anomeric": "[CX4]([OX2])([OX2])",
    "disulfide": "[SX2][SX2]",
    "thiol": "[SX2H]",
    "C=C": "[CX3]=[CX3]",
    "primary amine": "[NX3;H2][CX4]",
    "aromatic N": "[n]",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--energies", default=os.path.join(THERMO, "mlip", "G_aq_tecrdb_full.json"))
    ap.add_argument("--reactions", default=os.path.join(HERE, "tecrdb_full_reactions.json"))
    ap.add_argument("--experiment", default=os.path.join(HERE, "tecrdb_full_experiment.json"))
    ap.add_argument("--species", default=os.path.join(HERE, "tecrdb_full_species.json"))
    ap.add_argument("--metabolites", default=os.path.join(HERE, "tecrdb_full_metabolites.json"))
    ap.add_argument("--out", default=os.path.join(THERMO, "results", "benchmark",
                                                  "tecrdb_full_scored.json"))
    args = ap.parse_args()

    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")

    bd = json.load(open(args.energies))
    reactions = json.load(open(args.reactions))
    experiment = json.load(open(args.experiment))
    spec = json.load(open(args.species))
    mets = {m["id"]: m for m in json.load(open(args.metabolites))}

    G = {}
    for c, rec in bd.items():
        G[c] = (extract_ensemble_energy(
            rec, temperature_K=config.DEFAULT_CONDITIONS.temperature_K).gibbs_kJ
            if "conformers" in rec else float(rec["G_aq_kJ"]))
    S = {c: SpeciesInfo(c, n_hydrogens=int(v["n_hydrogens"]), charge=int(v["charge"]))
         for c, v in spec.items()}
    C = config.DEFAULT_CONDITIONS

    scored, missing = {}, 0
    for rid, st in reactions.items():
        if any(c not in G for c in st):
            missing += 1
            continue
        scored[rid] = reaction_dG(Reaction(rid, st), G, S, conditions=C).dG_transformed_kJ
    print(f"scored {len(scored)}/{len(reactions)} reactions "
          f"({missing} skipped for a missing compound energy)")

    exp = {r: experiment[r]["dG_kJ"] for r in scored}
    err = {r: scored[r] - exp[r] for r in scored}
    absr = [abs(v) for v in err.values()]
    pz = [abs(exp[r]) for r in scored]
    print(f"\nP1  QM MAE {statistics.mean(absr):.1f}   predict-zero {statistics.mean(pz):.1f}   "
          f"signs {sum(scored[r]*exp[r] > 0 for r in scored)}/{len(scored)}")

    # ---- group deltas --------------------------------------------------------
    pats = {k: Chem.MolFromSmarts(v) for k, v in GROUPS.items()}
    cnt = {}
    for c, m in mets.items():
        mol = Chem.MolFromSmiles(m["smiles"])
        cnt[c] = None if mol is None else {k: len(mol.GetSubstructMatches(p))
                                           for k, p in pats.items()}
    rows, names = [], []
    for rid in scored:
        st = reactions[rid]
        if any(cnt.get(c) is None for c in st):
            continue
        rows.append([sum(v * cnt[c][k] for c, v in st.items()) for k in GROUPS])
        names.append(rid)
    X = np.array(rows, dtype=float)
    y = np.array([err[r] for r in names])

    pop = {k: int((X[:, i] != 0).sum()) for i, k in enumerate(GROUPS)}
    anh = X[:, list(GROUPS).index("phosphoanhydride")]
    m1 = anh == -1
    print(f"\nP2  d(P-O-P) = -1: n={int(m1.sum())}  mean signed "
          f"{y[m1].mean():+.1f} kJ/mol (pre-registered +64 +/- 15)"
          if m1.sum() else "\nP2  no d(P-O-P) = -1 reactions")

    # joint least-squares through the origin, all groups at once
    keep = [i for i, k in enumerate(GROUPS) if pop[k] >= 8]
    coef, *_ = np.linalg.lstsq(X[:, keep], y, rcond=None)
    print(f"\nP3  joint fit (n={len(y)}, {len(keep)} groups with >=8 occurrences)")
    print(f"    {'group':20s} {'n':>4s} {'joint kJ/unit':>14s} {'solo mean signed':>17s}")
    joint = {}
    for j, i in enumerate(keep):
        k = list(GROUPS)[i]
        solo = y[X[:, i] != 0].mean()
        joint[k] = float(coef[j])
        print(f"    {k:20s} {pop[k]:4d} {coef[j]:14.1f} {solo:17.1f}")
    big = [k for k, v in joint.items() if abs(v) > 15 and k != "phosphoanhydride"]
    print(f"    -> groups other than P-O-P with |joint bias| > 15: "
          f"{', '.join(big) if big else 'NONE (P3 holds)'}")

    resid = y - X[:, keep] @ coef
    print(f"    MAE {np.abs(y).mean():.1f} -> {np.abs(resid).mean():.1f} after the joint fit "
          f"(in-sample; not a validated correction)")

    # ---- P4: reference-network coverage -------------------------------------
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = {}
    for rid in names:
        acc = np.zeros(2048)
        for c, v in reactions[rid].items():
            mol = Chem.MolFromSmiles(mets[c]["smiles"])
            if mol is None:
                acc = None
                break
            acc += v * np.array(gen.GetFingerprint(mol), dtype=float)
        if acc is not None and np.linalg.norm(acc) > 0:
            fps[rid] = acc / np.linalg.norm(acc)
    ids = list(fps)
    M = np.array([fps[i] for i in ids])
    sim = M @ M.T
    np.fill_diagonal(sim, -1.0)
    best = sim.max(axis=1)
    for thr in (0.95, 0.90, 0.80):
        print(f"P4  best-reference cosine >= {thr:.2f}: {(best >= thr).mean():.0%} "
              f"of {len(ids)} reactions" + ("   <- pre-registered >50% at 0.90"
                                            if abs(thr - 0.90) < 1e-9 else ""))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"scored_kJ": scored, "experiment_kJ": exp, "error_kJ": err,
               "group_joint_kJ_per_unit": joint, "group_occurrences": pop,
               "reference_cosine_best": {i: float(b) for i, b in zip(ids, best)}},
              open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
