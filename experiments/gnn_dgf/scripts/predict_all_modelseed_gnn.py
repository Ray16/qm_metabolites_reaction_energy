#!/usr/bin/env python
"""GNN Delta_r G'^o for EVERY ModelSEED reaction, with calibrated uncertainty and
an honest reliability tier -- the deliverable that makes the 100%-coverage claim
meaningful (coverage != trust).  Parallels results/eq/modelseed_all_dG.json
(eQuilibrator) and the retrained-dGP sweep, but produced by OUR local graph-only
GNN so it is an independent prediction (double-validation vs the published
ModelSEED dev-branch thermodynamics).

Model: graph-only ensemble (artifacts/checkpoint_graph.pt, xtb-free -> scales to
the whole database).  Prediction per reaction:  dG = sum_i coeff_i * f(compound_i)
where f is the per-compound formation energy (mean over the 12-model ensemble).

Conventions (must match how the GNN was trained on TECRDB apparent dG'^o):
  * Proton cpd00067 is DROPPED from every reaction.  TECRDB targets are apparent
    dG'^o (pH folds H+ in) and H+ was never a training compound; dropping it makes
    the ModelSEED reaction consistent with the learned f.  (H2O cpd00001 IS a
    training compound -> kept.)
  * S@f is already apparent dG'^o at ~pH7 -- no extra Legendre transform (unlike
    dGP/eQ, whose stored numbers we match in meaning, not in derivation).

Uncertainty:  sigma_total^2 = (s*sigma_ens)^2 + tau^2, (s,tau) from the graph-only
held-out calibration (artifacts/uq_calibration_none.json, CPD-DISJOINT = the
extrapolation regime that fits novel ModelSEED compounds).

Reliability tier (per reaction):
  out_of_domain : a participant contains an element NEVER seen in the 453 TECRDB
                  training compounds (in-domain = {C,H,O,N,P,S,I}).  Catches every
                  metal-/halogen-/Se-/B-coordinated compound -- pure extrapolation,
                  do NOT trust the number.
  extrapolation : all elements in-domain, but calibrated sigma is in the top decile
                  (>90th pct of in-domain sigma) -> high model uncertainty.
  in_domain     : all elements in-domain and sigma below that cut.
  unpredictable : a non-proton participant has no usable structure (missing SMILES,
                  R-group '*', or RDKit parse fail) -> dG=None (as dGP/eQ also skip).

Run (gnndgf env; import rdkit before torch):
  CUDA_VISIBLE_DEVICES=1 python scripts/predict_all_modelseed_gnn.py
"""
import _bootstrap  # noqa: F401
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

# rdkit BEFORE torch (features imports rdkit; importing rdkit after torch hits a
# GLIBCXX conflict in this env)
from gnn.features import CompoundGraphs  # noqa: E402  (pulls rdkit first)
from rdkit import Chem, RDLogger          # noqa: E402
import numpy as np                        # noqa: E402
import torch                              # noqa: E402

from gnn import paths                     # noqa: E402
from gnn.model import MPNN, Graph, DEV    # noqa: E402

RDLogger.DisableLog("rdApp.*")

IN_DOMAIN = set("C H O N P S I".split())   # elements present in the 453 TECRDB compounds
PROTON = "cpd00067"
CSV_MAX = None                             # cap for --smoke


def isnull(x):
    return x is None or str(x).strip() in ("", "null", "None", "nan")


def formula_elements(formula):
    return set(re.findall(r"[A-Z][a-z]?", formula or ""))


def load_compounds():
    """cpd_id -> {smiles, charge, formula}.  Active (non-obsolete) only."""
    comps = {}
    for f in sorted(glob.glob(f"{paths.DB}/compound_*.tsv")):
        for r in csv.DictReader(open(f), delimiter="\t"):
            if r.get("is_obsolete") == "1":
                continue
            comps[r["id"]] = dict(smiles=r.get("smiles"), charge=r.get("charge"),
                                  formula=r.get("formula"))
    return comps


def load_reactions():
    """rxn_id -> {stoich(list of (coeff,cpd)), is_transport, balanced}.  Active only.

    `balanced` = ModelSEED's own status field is exactly 'OK'.  A reaction that is
    mass- or charge-imbalanced (status MI:.../CI:...) is a lumped/biomass/generic
    pseudo-reaction whose Delta_r G is undefined for ANY method (eQuilibrator
    emits |dG| up to 44,000 kJ on these; it flags them itself).  We refuse to
    emit a number rather than publish a meaningless one.
    """
    rxns = {}
    pat = re.compile(r"(-?\d+(?:\.\d+)?):(cpd\d+):(\d+)")
    for f in sorted(glob.glob(f"{paths.DB}/reaction_*.tsv")):
        for r in csv.DictReader(open(f), delimiter="\t"):
            if r.get("is_obsolete") == "1":
                continue
            stoich = [(float(c), cpd) for c, cpd, _cmp in pat.findall(r.get("stoichiometry") or "")]
            rxns[r["id"]] = dict(stoich=stoich, is_transport=(r.get("is_transport") == "1"),
                                 balanced=(r.get("status", "").strip() == "OK"))
    return rxns


def featurize_predict(comps, ckpt, chunk=2000):
    """Return f_ens: cpd_id -> np.array(n_ens) of per-compound formation energies,
    and predictable: set of cpd ids that produced a prediction."""
    # which compounds have a usable graph
    good = []
    for cid, c in comps.items():
        smi = c["smiles"]
        if isnull(smi) or "*" in smi:
            continue
        if Chem.MolFromSmiles(smi) is None:
            continue
        good.append(cid)
    print(f"  featurizable compounds: {len(good)} / {len(comps)}")

    # rebuild ensemble
    hp = ckpt["hp"]
    models = []
    for st in ckpt["model_states"]:
        m = MPNN(ckpt["atom_dim"], ckpt["bond_dim"], ckpt["qm_dim"],
                 hp["hidden"], hp["layers"], hp["drop"]).to(DEV)
        m.load_state_dict(st); m.eval(); models.append(m)
    n_ens = len(models)

    f_ens = {}
    for s in range(0, len(good), chunk):
        ids = good[s:s + chunk]
        mets = [{"id": cid, "smiles": comps[cid]["smiles"]} for cid in ids]
        g = Graph(CompoundGraphs(mets, level="none").pack())
        with torch.no_grad():
            fk = torch.stack([mdl(g) for mdl in models]).cpu().numpy()  # (n_ens, n_chunk)
        for j, cid in enumerate(ids):
            f_ens[cid] = fk[:, j]
        print(f"    predicted {min(s + chunk, len(good))}/{len(good)}", flush=True)
    return f_ens, n_ens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoint_graph.pt")
    ap.add_argument("--calib", default="uq_calibration_none.json")
    ap.add_argument("--regime", default="CPD-DISJOINT",
                    choices=["CPD-DISJOINT", "RANDOM"],
                    help="which held-out calibration to use for sigma (frontier=CPD-DISJOINT)")
    ap.add_argument("--out", default="modelseed_all_dG_gnn")
    ap.add_argument("--smoke", action="store_true", help="first 3 compound/reaction shards")
    a = ap.parse_args()

    ckpt = torch.load(paths.artifact(a.checkpoint))
    assert ckpt.get("level") == "none", f"expected graph-only checkpoint, got level={ckpt.get('level')}"
    cal = json.load(open(paths.artifact(a.calib)))[a.regime]["calib"]
    s_cal, tau_cal = cal["scale_s"], cal["tau_kJ"]
    print(f"device={DEV}  checkpoint={a.checkpoint}  calibration[{a.regime}]: "
          f"sigma^2=({s_cal:.2f}*sigma_ens)^2+{tau_cal:.2f}^2")

    comps = load_compounds()
    rxns = load_reactions()
    print(f"active compounds={len(comps)}  active reactions={len(rxns)}")

    f_ens, n_ens = featurize_predict(comps, ckpt)

    # per-compound out-of-domain elements (formula-based; catches metals/halogens)
    oo_by_cpd = {cid: (formula_elements(c["formula"]) - IN_DOMAIN)
                 for cid, c in comps.items()}

    # ---- predict every reaction ------------------------------------------------
    rec = {}
    for rid, r in rxns.items():
        if not r["balanced"]:
            # mass/charge-imbalanced -> dG undefined for any method; refuse to score
            rec[rid] = dict(dG_prime_kJ=None, reason="imbalanced (ModelSEED status != OK)",
                            reliability="imbalanced")
            continue
        net = defaultdict(float)
        for coeff, cpd in r["stoich"]:
            net[cpd] += coeff
        net.pop(PROTON, None)
        parts = {c: v for c, v in net.items() if abs(v) > 1e-9}
        if not parts:
            rec[rid] = dict(dG_prime_kJ=None, reason="empty (proton-only / transport cancels)")
            continue
        missing = [c for c in parts if c not in f_ens]
        if missing:
            rec[rid] = dict(dG_prime_kJ=None, reason="missing structure",
                            missing=missing[:6])
            continue
        dG_k = np.zeros(n_ens)
        for c, v in parts.items():
            dG_k = dG_k + v * f_ens[c]
        dG = float(dG_k.mean())
        sig_ens = float(dG_k.std())
        sig_tot = float(np.sqrt((s_cal * sig_ens) ** 2 + tau_cal ** 2))
        oo = sorted(set().union(*[oo_by_cpd[c] for c in parts]))
        rec[rid] = dict(dG_prime_kJ=round(dG, 2), uncertainty_kJ=round(sig_tot, 2),
                        _sig=sig_tot, oo_elements=oo, is_transport=r["is_transport"])

    # ---- tiering: extrapolation cut = 90th pct of in-domain sigma --------------
    indom_sig = [v["_sig"] for v in rec.values()
                 if v["dG_prime_kJ"] is not None and not v["oo_elements"]]
    cut = float(np.percentile(indom_sig, 90)) if indom_sig else 1e9
    tier_counts = defaultdict(int)
    for v in rec.values():
        if v.get("reliability") == "imbalanced":
            pass  # already set; keep
        elif v["dG_prime_kJ"] is None:
            v["reliability"] = "unpredictable"
        elif v["oo_elements"]:
            v["reliability"] = "out_of_domain"
        elif v["_sig"] > cut:
            v["reliability"] = "extrapolation"
        else:
            v["reliability"] = "in_domain"
        v.pop("_sig", None)
        if not v.get("oo_elements"):
            v.pop("oo_elements", None)
        tier_counts[v["reliability"]] += 1

    # ---- write JSON + CSV ------------------------------------------------------
    outj = os.path.join(paths.RESULTS, "eq", f"{a.out}.json")
    outc = os.path.join(paths.RESULTS, "eq", f"{a.out}.csv")
    os.makedirs(os.path.dirname(outj), exist_ok=True)
    json.dump(rec, open(outj, "w"))
    with open(outc, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["modelseed_rxn", "dG_prime_kJ_per_mol", "uncertainty_kJ",
                     "reliability", "out_of_domain_elements", "is_transport"])
        for rid, v in rec.items():
            wr.writerow([rid, v.get("dG_prime_kJ", ""), v.get("uncertainty_kJ", ""),
                         v["reliability"], "|".join(v.get("oo_elements", [])),
                         int(v.get("is_transport", False))])

    # ---- summary ---------------------------------------------------------------
    print("\n==== reliability tiers (all ModelSEED reactions) ====")
    for t in ("in_domain", "extrapolation", "out_of_domain", "imbalanced", "unpredictable"):
        n = tier_counts[t]
        print(f"  {t:<14s} {n:6d}  ({100*n/len(rec):.1f}%)")
    print(f"  total reactions {len(rec)};  extrapolation cut = sigma > {cut:.1f} kJ (90th pct in-domain)")
    trust = tier_counts["in_domain"] + tier_counts["extrapolation"]
    print(f"  numeric prediction emitted for {trust} balanced reactions "
          f"({tier_counts['in_domain']} in-domain-trustworthy); "
          f"{tier_counts['imbalanced']} imbalanced reactions refused (dG undefined)")

    # cross-check overlap with eQ (double-validation hook)
    eqp = os.path.join(paths.RESULTS, "eq", "modelseed_all_dG.json")
    if os.path.exists(eqp):
        eq = json.load(open(eqp))
        both = [(rid, rec[rid]["dG_prime_kJ"], eq[rid]["dG_prime_kJ"])
                for rid in rec if rid in eq
                and rec[rid]["dG_prime_kJ"] is not None
                and isinstance(eq.get(rid), dict) and eq[rid].get("dG_prime_kJ") is not None]
        if both:
            d = np.array([abs(g - e) for _, g, e in both])
            print(f"\n  vs eQuilibrator on {len(both)} shared predicted reactions: "
                  f"MAE {d.mean():.1f}  median {np.median(d):.1f}  kJ/mol (independent methods)")
    print(f"\nwrote {outj}\n      {outc}")
    summ = dict(tiers=dict(tier_counts), extrap_cut_kJ=cut, n_reactions=len(rec),
                calibration=dict(regime=a.regime, s=s_cal, tau=tau_cal),
                in_domain_elements=sorted(IN_DOMAIN))
    json.dump(summ, open(paths.artifact("modelseed_gnn_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
