"""Aggregate the TECRDB sample-sweep results by structural failure category. For each
category report N, MAE vs exp, and MAE vs exp_sd (an error <= exp_sd is 'as good as the
data'). Also lists the worst misses. Reads logs/sweep_*.log + reactions_tecrdb_all.json.
No GPU. This turns the sweep into the empirical failure map."""
import json, os, re, glob, collections, math

HERE = os.path.dirname(__file__)
S = os.path.join(HERE, "..", "scripts")
allr = json.load(open(os.path.join(S, "reactions_tecrdb_all.json")))
flags = json.load(open(os.path.join(HERE, "..", "..", "..", "pipeline", "tecrdb367_failure_flags.json")))
LOG = os.path.join(HERE, "..", "logs")

pat = re.compile(r"ΔG = ([+-]?\d+\.\d+) ± ([\d.]+) .*?err \[([^\]]+)\]")
results = {}
for f in glob.glob(os.path.join(LOG, "sweep_*.log")):
    rid = os.path.basename(f)[6:-4]
    txt = open(f).read()
    m = pat.search(txt)
    unresolved = "UNRESOLVED" in txt
    if m:
        dG = float(m.group(1)); U = float(m.group(2))
        err = float(m.group(3).split(",")[0])
        results[rid] = dict(dG=dG, U=U, err=err, unresolved=unresolved)

print(f"sweep parsed: {len(results)}/{len(glob.glob(os.path.join(LOG,'sweep_*.log')))} reactions\n")
by_cat = collections.defaultdict(list)
for rid, r in results.items():
    for fl in flags.get(rid, ["CLEAN(tractable)"]):
        by_cat[fl].append((rid, r))

def mae(xs): return sum(abs(x) for x in xs) / len(xs) if xs else float("nan")
print(f"{'category':22s} {'N':>3s} {'MAE':>7s} {'MAE-vs-sd':>10s} {'within-sd':>10s}")
for cat, items in sorted(by_cat.items(), key=lambda x: -len(x[1])):
    errs = [r["err"] for _, r in items]
    sds = [allr[rid].get("exp_sd", 0) or 0 for rid, _ in items]
    within = sum(1 for e, s in zip(errs, sds) if s > 0 and abs(e) <= s)
    excess = [abs(e) - s for e, s in zip(errs, sds)]
    print(f"{cat:22s} {len(items):3d} {mae(errs):7.1f} {mae(excess):10.1f} {within:5d}/{len(items):<4d}")

print("\nworst misses:")
for rid, r in sorted(results.items(), key=lambda x: -abs(x[1]["err"]))[:8]:
    print(f"  {rid}: err {r['err']:+.1f} (ΔG {r['dG']:+.1f}±{r['U']:.1f}, exp {allr[rid]['exp'][0]}, "
          f"sd {allr[rid].get('exp_sd',0)}) {'[UNRESOLVED]' if r['unresolved'] else ''} "
          f"{','.join(flags.get(rid,[]))}")
json.dump(results, open(os.path.join(HERE, "..", "artifacts", "sweep_results.json"), "w"), indent=2)
