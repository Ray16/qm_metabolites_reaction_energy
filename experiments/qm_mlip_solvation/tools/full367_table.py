"""Rolling predicted-vs-measured table for the full-367 AUTO_TRUNCATE run.
Parses logs/full367/*.log (completed = has a ΔG line), joins measured exp dG + sd + EC + flags,
writes a markdown table (sorted by |err|) + a running summary (n, MAE, MAE-within-sd, per-flag).
Re-run any time; safe while the run is in progress."""
import json, os, re, glob, collections

HERE = os.path.dirname(__file__)
S = os.path.join(HERE, "..", "scripts")
LOG = os.path.join(HERE, "..", "logs", "full367")
allr = json.load(open(os.path.join(S, "reactions_tecrdb_all.json")))
flags = json.load(open(os.path.join(HERE, "..", "..", "..", "pipeline", "tecrdb367_failure_flags.json")))
OUT = os.path.join(HERE, "..", "artifacts", "full367_results.md")

pat = re.compile(r"ΔG = ([+-]?\d+\.\d+) ± ([\d.]+) kJ/mol\s+vs exp \[([^\]]+)\]\s+err \[([^\]]+)\]")
rows = []
for f in glob.glob(os.path.join(LOG, "*.log")):
    rid = os.path.basename(f)[:-4]
    txt = open(f, errors="ignore").read()
    m = pat.search(txt)
    if not m:
        continue
    dG = float(m.group(1)); U = float(m.group(2)); err = float(m.group(4).split(",")[0])
    trunc = "trunc" if "auto-truncated" in txt else ("full" if "fallback" in txt else "?")
    unres = "UNRES" if "UNRESOLVED" in txt else ""
    r = allr.get(rid, {})
    sd = r.get("exp_sd", 0) or 0
    fl = ",".join(x for x in flags.get(rid, []) if x != "CLEAN(tractable)") or "clean"
    rows.append(dict(rid=rid, ec=r.get("EC", ""), dG=dG, U=U, exp=r.get("exp", ["?"])[0],
                     sd=sd, err=err, trunc=trunc, unres=unres, flags=fl))

rows.sort(key=lambda r: -abs(r["err"]))
n = len(rows)
mae = sum(abs(r["err"]) for r in rows) / n if n else 0
within = sum(1 for r in rows if r["sd"] > 0 and abs(r["err"]) <= r["sd"])
by = collections.defaultdict(list)
for r in rows:
    by[r["trunc"]].append(abs(r["err"]))

lines = [f"# Full-367 predicted vs measured ΔG (rolling)  —  {n}/367 done",
         "",
         f"**MAE {mae:.1f} kJ/mol** | within exp_sd: {within}/{n} | "
         + " | ".join(f"{k}: n={len(v)} MAE={sum(v)/len(v):.1f}" for k, v in sorted(by.items())),
         "",
         "| rxn | EC | flags | mode | pred ΔG | ±U | exp | sd | err |",
         "|-----|----|-------|------|--------:|---:|----:|---:|----:|"]
for r in rows:
    lines.append(f"| {r['rid']} | {r['ec']} | {r['flags'][:24]} | {r['trunc']}{(' '+r['unres']) if r['unres'] else ''} "
                 f"| {r['dG']:+.1f} | {r['U']:.1f} | {r['exp']:+.2f} | {r['sd']:.1f} | {r['err']:+.1f} |")
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines[:3]))
print(f"...\nwrote {OUT} ({n} reactions)")
