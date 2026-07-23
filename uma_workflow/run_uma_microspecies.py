#!/usr/bin/env python
"""Stage B for the corrected pH-7 microspecies (uma env, GPU).

Scores G_aq for the alternative species built by build_microspecies.py using the
identical composite (E_UMA gas + ALPB dGsolv + xtb G_RRHO, Boltzmann-averaged),
so they are directly interchangeable with the production per-compound energies.

Run:  CUDA_VISIBLE_DEVICES=0 /homes/rzhu/miniforge3/envs/uma/bin/python \
          run_uma_microspecies.py

Writes: uma_workflow/G_aq_microspecies.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
BENCH = os.path.join(THERMO, "large_dGPredictor_error")
sys.path.insert(0, THERMO)
sys.path.insert(0, HERE)

import run_uma_ensemble as E   # noqa: E402  (reuse the exact composite)

ENS_JSON = os.path.join(BENCH, "microspecies_xtb.json")
MET_JSON = os.path.join(BENCH, "microspecies_metabolites.json")
OUT = os.path.join(HERE, "G_aq_microspecies.json")


def main():
    ensemble = json.load(open(ENS_JSON))
    meta = {m["id"]: m for m in json.load(open(MET_JSON))}

    print(f"=== UMA microspecies | model={E.UMA_MODEL} | RT={E.RT:.3f} kJ/mol "
          f"| {len(ensemble)} species ===")
    out = {}
    for c in sorted(ensemble):
        chg = int(meta[c]["charge"])
        g_list, per_conf = [], []
        for cf in ensemble[c]:
            e_uma = E.uma_gas_energy_kJ(cf["xyz"], chg)
            g_i = e_uma + cf["dGsolv_kJ"] + cf["G_RRHO_kJ"]
            g_list.append(g_i)
            per_conf.append(dict(conf=cf["conf"], E_UMA_kJ=e_uma,
                                 dGsolv_kJ=cf["dGsolv_kJ"],
                                 G_RRHO_kJ=cf["G_RRHO_kJ"], G_aq_kJ=g_i))
        g_ens, weights = E.boltzmann_free_energy(g_list)
        for pc, w in zip(per_conf, weights):
            pc["weight"] = w
        neff = 1.0 / sum(w * w for w in weights)
        out[c] = dict(name=meta[c]["name"], charge=chg, smiles=meta[c]["smiles"],
                      n_conf=len(g_list), n_eff=neff, G_aq_kJ=g_ens,
                      conformers=per_conf)
        print(f"{c:22s} q={chg:+d}: {len(g_list):2d} conf (n_eff={neff:4.1f})  "
              f"G_aq={g_ens:12.1f}  ({meta[c]['name']})")

    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
