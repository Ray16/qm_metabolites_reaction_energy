"""Route every TECRDB reaction under several flag variants (no QM) and collect the union of species
'SMILES qN' that implicit_G would be asked for. Usage: collect_species.py OUT.json"""
import os, sys, json, subprocess
M = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REX = os.environ.get("TECRDB_INPUTS", os.path.join(os.path.dirname(M), "experiments/qm_mlip_solvation/scripts/reactions_tecrdb_std.json"))
VARIANTS = [{"NTP_CORE": "1"}, {"NTP_CORE": "1", "ARYLAMINE_NONBASIC": "1", "PKA_ENV": "1", "ZWITTERION_PH0": "1"}, {"NTP_CORE": "1", "CARBONYL_HYDRATION_ALL": "1", "ARYLAMINE_NONBASIC": "1", "PKA_ENV": "1", "ZWITTERION_PH0": "1"}] or [{}, {"ZWITTERION_PH0": "1"}, {"TRUNC_FG_CUTS": "0", "TRUNC_ANOMERIC_RADIUS": "0"},
            {"PH0_REDOX_PROTON": "0"}, {"TRUNC_ANOMERIC_RADIUS": "0"}, {"PKA_ENV": "1", "ZWITTERION_PH0": "1"}]
if len(sys.argv) > 2:                                   # child: one variant
    sys.path.insert(0, M)
    import metag.pipeline as P
    seen = set()
    def fake(pu, q, smi, *a, **k):
        seen.add(f"{smi} q{int(q)}"); return (0.0, 1.0)
    P.implicit_G = fake
    P.water_ref_G = lambda pu, log=None: 0.0
    rx = json.load(open(REX))
    for rid, r in rx.items():
        try:
            P.score_reaction(None, dict(r, species={k: tuple(v) for k, v in r["species"].items()}),
                             log=lambda *x: None, key=rid)
        except Exception as e:
            print("ERR", rid, repr(e)[:200], file=sys.stderr)
    json.dump(sorted(seen), open(sys.argv[2], "w")); sys.exit()
allsp = set()
for i, v in enumerate(VARIANTS):
    out = f"/tmp/collect_{os.getpid()}_{i}.json"
    subprocess.run([sys.executable, __file__, "child", out], env={**os.environ, **v, "PYTHONPATH": M}, check=True)
    s = json.load(open(out)); print(v, len(s)); allsp |= set(s)
json.dump(sorted(allsp), open(sys.argv[1], "w"), indent=0); print("union", len(allsp))
