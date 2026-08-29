"""Verbose single-reaction re-run for debugging routing/protonation artifacts.
    CUDA_VISIBLE_DEVICES=4 python analysis/debug_route.py rxn00486
Prints the full pipeline log (routing decisions, per-species G, pH-0 pKa sites) to stdout.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
reactions = json.load(open(os.path.join(HERE, "divergent_inputs.json")))
rid = sys.argv[1]
rx = reactions[rid]
rxin = dict(rx, species={k: tuple(v) for k, v in rx["species"].items()})
from metag.backend.uma import load_uma
from metag.pipeline import score_reaction
pu = load_uma(os.environ.get("METAG_MODEL", "uma-s-1p2p1"))
print(f"### {rid}  {rx['note']}  n_H+={rx['n_Hplus']}  gc={rx['gc']} eq={rx['eq']}")
r = score_reaction(pu, rxin, seeds=(1, 2), keep=8, pool=48, log=print, key=rid)
print(f"\n### RESULT dG={r['dG']} raw={r['dG_raw']} anchor={r['anchor']} class={r['sigma_breakdown'].get('class')} suspect={r['suspect']}")
