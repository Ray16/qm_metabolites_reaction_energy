#!/usr/bin/env python
"""Stage B of the conformer-ensemble upgrade (uma env, GPU): UMA + Boltzmann.

Reads the CREST/xtb ensemble from Stage A (large_dGPredictor_error/build_bench_
ensembles.py) and adds ONLY the UMA electronic energy per conformer, then forms
the conformational free energy:

    G_aq,i   = E_UMA(gas, conf i) + dGsolv(xtb-ALPB, i) + G_RRHO(xtb, i)
    G_aq     = G_min - RT * ln Σ_i exp(-(G_aq,i - G_min)/RT)      (ensemble free G,
                                                     conformational entropy included)

Reaction Delta_rG'^o via the standard qm_thermo Alberty + Debye-Huckel transform,
directly comparable to the dGPredictor + TECRDB values in the collaborator's reactions CSV.
This replaces the single-conformer run_uma_composite_single.py; the difference isolates the
conformational-sampling effect.

Run (uma env, free GPU):
    CUDA_VISIBLE_DEVICES=<n> /homes/rzhu/miniforge3/envs/uma/bin/python run_uma_ensemble.py

Writes:
    uma_workflow/G_aq_ensemble.json                 (per-compound + per-conf)
    results/benchmark/qm_reaction_dG_ensemble.json  (rxn_id -> kJ/mol)
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

from ase import Atoms

THERMO = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc"
BENCH = os.path.join(THERMO, "large_dGPredictor_error")
sys.path.insert(0, THERMO)
sys.path.insert(0, "/homes/rzhu/uma_tools")
from qm_thermo import config                                  # noqa: E402
from qm_thermo.reactions import Reaction, reaction_dG, SpeciesInfo  # noqa: E402
import uma_helper                                             # noqa: E402

EV_TO_KJ = 96.48533212
UMA_MODEL = os.environ.get("UMA_MODEL", "uma-s-1p2")
RT = config.DEFAULT_CONDITIONS.R_kJ * config.DEFAULT_CONDITIONS.temperature_K  # 2.479 kJ/mol

SPECIES_JSON = os.path.join(BENCH, "species.json")
RXN_JSON = os.path.join(BENCH, "reactions.json")
RXN_CSV = os.path.join(BENCH, "top10_reactions_stereo_significant.csv")
# Ensemble to score: the RDKit-ETKDG + xtb ensemble (the standard method; CREST was
# retired as too slow). Overridable via ENS_JSON. Output names track the source.
ENS_JSON = os.environ.get("ENS_JSON", os.path.join(BENCH, "ensemble_fast_xtb.json"))
TAG = os.environ.get(
    "TAG", "ensemble_fast" if "fast" in os.path.basename(ENS_JSON) else "ensemble")

_CALC = None


def _uma_calc():
    global _CALC
    if _CALC is None:
        _CALC = uma_helper.get_calculator("omol", UMA_MODEL)
    return _CALC


def read_xyz_atoms(path):
    lines = open(path).read().splitlines()
    n = int(lines[0])
    syms, pos = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        syms.append(p[0]); pos.append([float(p[1]), float(p[2]), float(p[3])])
    return Atoms(symbols=syms, positions=pos)


def uma_gas_energy_kJ(xyz_path, chg):
    atoms = read_xyz_atoms(xyz_path)
    atoms.info["charge"] = int(chg)
    atoms.info["spin"] = 1
    atoms.calc = _uma_calc()
    return float(atoms.get_potential_energy()) * EV_TO_KJ


def boltzmann_free_energy(g_list):
    """G_ensemble = Gmin - RT ln Σ exp(-(Gi-Gmin)/RT); also return weights."""
    gmin = min(g_list)
    z_terms = [math.exp(-(g - gmin) / RT) for g in g_list]
    Z = sum(z_terms)
    g_ens = gmin - RT * math.log(Z)
    weights = [z / Z for z in z_terms]
    return g_ens, weights


def load_reactions():
    raw = json.load(open(RXN_JSON))
    return {rid: Reaction(rid, {c: float(v) for c, v in st.items()})
            for rid, st in raw.items()}


def load_reference_table():
    dgp, exp, name = {}, {}, {}
    for row in csv.DictReader(open(RXN_CSV)):
        rid = row["modelseed_rxn"]
        name[rid] = row["name"]
        dgp[rid] = float(row["dGpredictor_modelseed_dG_kJ"])
        exp[rid] = float(row["tecrdb_dG_kJ"])
    return dgp, exp, name


def main():
    spec = json.load(open(SPECIES_JSON))
    ensemble = json.load(open(ENS_JSON))
    reactions = load_reactions()
    dgp, exp, name = load_reference_table()

    need = set()
    for rxn in reactions.values():
        need |= {c for c in rxn.compounds() if c != "cpd00067"}
    missing = need - set(ensemble)
    if missing:
        raise SystemExit(f"no CREST ensemble yet for: {sorted(missing)} "
                         f"(run build_bench_ensembles.py first)")

    print(f"=== UMA + CREST ensemble (benchmark set) | model={UMA_MODEL} | "
          f"RT={RT:.3f} kJ/mol | {len(need)} compounds ===")
    G_aq, breakdown = {}, {}
    for c in sorted(need):
        chg = int(spec[c]["charge"])
        confs = ensemble[c]
        g_list, per_conf = [], []
        for cf in confs:
            e_uma = uma_gas_energy_kJ(cf["xyz"], chg)
            g_i = e_uma + cf["dGsolv_kJ"] + cf["G_RRHO_kJ"]
            g_list.append(g_i)
            per_conf.append(dict(conf=cf["conf"], E_UMA_kJ=e_uma,
                                 dGsolv_kJ=cf["dGsolv_kJ"], G_RRHO_kJ=cf["G_RRHO_kJ"],
                                 G_aq_kJ=g_i, n_imag=cf["n_imag"]))
        g_ens, weights = boltzmann_free_energy(g_list)
        for pc, w in zip(per_conf, weights):
            pc["weight"] = w
        neff = 1.0 / sum(w * w for w in weights)          # participation ratio
        G_aq[c] = g_ens
        breakdown[c] = dict(name=spec[c]["name"], charge=chg, n_conf=len(confs),
                            n_eff=neff, G_aq_kJ=g_ens,
                            single_conf_G_aq_kJ=per_conf[
                                min(range(len(g_list)), key=lambda i: g_list[i])]["G_aq_kJ"],
                            conformers=per_conf)
        print(f"{c} q={chg:+d}: {len(confs):2d} conf (n_eff={neff:4.1f})  "
              f"G_aq(ens)={g_ens:12.1f}  ΔG_conf={g_ens - min(g_list):5.1f} kJ  ({spec[c]['name']})")

    json.dump(breakdown, open(os.path.join(THERMO, "uma_workflow",
                                           f"G_aq_{TAG}.json"), "w"), indent=2)

    species = {c: SpeciesInfo(c, n_hydrogens=int(spec[c]["n_hydrogens"]),
                              charge=int(spec[c]["charge"])) for c in G_aq}
    bench = os.path.join(THERMO, "results", "benchmark")
    out_rows = {}
    print(f"\n{'rxn':10s} {'exp':>8s} {'dGPredictor':>12s} {'QM(ens)':>9s} "
          f"{'|QM-exp|':>9s} {'|dGP-exp|':>10s}  name")
    qm_ae, dgp_ae = [], []
    for rid, rxn in reactions.items():
        qm = reaction_dG(rxn, G_aq, species,
                         conditions=config.DEFAULT_CONDITIONS).dG_transformed_kJ
        out_rows[rid] = qm
        e = exp[rid]
        qm_ae.append(abs(qm - e)); dgp_ae.append(abs(dgp[rid] - e))
        print(f"{rid:10s} {e:8.1f} {dgp[rid]:12.1f} {qm:9.1f} "
              f"{abs(qm - e):9.1f} {abs(dgp[rid] - e):10.1f}  {name[rid]}")

    out_path = os.path.join(bench, f"qm_reaction_dG_{TAG}.json")
    json.dump(out_rows, open(out_path, "w"), indent=2)
    src = "CREST" if TAG == "ensemble" else "fast RDKit+xtb"
    mae = lambda v: sum(v) / len(v)
    print(f"\nMAE vs TECRDB experiment (n={len(qm_ae)}):")
    print(f"   dGPredictor (fine-tuned)      = {mae(dgp_ae):6.1f}")
    print(f"   QM (UMA + {src} ensemble)  = {mae(qm_ae):6.1f}   kJ/mol")
    print(f"   QM closer in {sum(1 for a,b in zip(qm_ae,dgp_ae) if a<b)}/{len(qm_ae)} reactions")
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
