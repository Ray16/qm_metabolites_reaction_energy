#!/usr/bin/env python
"""Did explicit first-shell water fix the anion-solvation error?

Compares three treatments of the same acid/base pairs against experimental pKa:

  bare ALPB       what the pipeline does now  (anionic error +40 kJ/mol at k=1)
  cluster-continuum   n explicit waters on BOTH acid and base, xtb-relaxed,
                      continuum outside; the water count cancels in the
                      deprotonation, only differential stabilisation survives

The proton reference is re-fitted on the CATIONIC pairs for each treatment,
because those involve no anion at all -- so any residual anionic error after
re-referencing is genuine anion physics and not a bookkeeping shift. This is the
same control that showed CPCM-X was breaking the reference rather than fixing
the anions.

Decision rule, set before looking: microsolvation is worth pursuing only if the
anion-specific error drops below ~10 kJ/mol. The reaction-level target is
sqrt(4) x per-species error < 8, so per-species must reach ~4.

Run:  /homes/rzhu/miniforge3/envs/palm/bin/python analyze_microsolv.py [--nwat 3]
"""
import argparse
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RT = 8.314462618e-3 * 298.15
RTLN10 = RT * math.log(10)
MU_H = -1122.8


def load(nwat):
    pairs = json.load(open(os.path.join(HERE, "pka_pairs.json")))
    ens = json.load(open(os.path.join(HERE, "pka_xtb.json")))
    ms = json.load(open(os.path.join(HERE, f"microsolv_n{nwat}.json")))
    return pairs, ens, ms


def bare_G(ens, key):
    """G of the bare species: xtb ALPB electronic + shared G_RRHO."""
    c = ens[key][0]
    return c["dGsolv_kJ"] + c["G_RRHO_kJ"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nwat", type=int, default=3)
    args = ap.parse_args()
    pairs, ens, ms = load(args.nwat)

    # bare reference energies from the production ensembles (UMA-scored G_aq)
    gpath = os.path.join(os.path.dirname(HERE), "uma_workflow", "G_aq_pka.json")
    bare = ({c: r["G_aq_kJ"] for c, r in json.load(open(gpath)).items()}
            if os.path.isfile(gpath) else None)

    def usable(p):
        ok = all(k in ms and ms[k] for k in (p["acid"], p["base"]))
        if bare is not None:
            ok = ok and all(k in bare for k in (p["acid"], p["base"]))
        return ok

    rows = []
    for p in pairs:
        if not usable(p):
            continue
        d_ms = ms[p["base"]]["G"] + MU_H - ms[p["acid"]]["G"]
        r = dict(key=p["key"], kind=p["kind"], group=p["group"],
                 q=abs(int(p.get("q_base", -1))), pKa_exp=p["pKa_exp"],
                 pKa_ms=d_ms / RTLN10)
        if bare is not None:
            d_b = bare[p["base"]] + MU_H - bare[p["acid"]]
            r["pKa_bare"] = d_b / RTLN10
        rows.append(r)

    if not rows:
        raise SystemExit("no usable pairs yet -- microsolvation still running?")

    def refit(col):
        """Shift mu_H so the CATIONIC pairs read zero error; return anionic stats."""
        cat = [r for r in rows if r["kind"] == "cationic"]
        shift = (np.mean([(r[col] - r["pKa_exp"]) for r in cat]) * RTLN10
                 if cat else 0.0)
        out = {}
        for kind in ("anionic", "cationic"):
            e = [((r[col] - r["pKa_exp"]) * RTLN10 - shift)
                 for r in rows if r["kind"] == kind]
            if e:
                out[kind] = (np.mean(e), np.std(e), np.mean(np.abs(e)), len(e))
        return shift, out

    print(f"n = {len(rows)} acid/base pairs with {args.nwat} explicit waters\n")
    print(f"{'treatment':22}{'mu_H shift':>12}{'anion bias':>12}{'anion sd':>10}"
          f"{'anion MAE':>11}{'cation MAE':>12}")
    for col, lab in (("pKa_bare", "bare ALPB"), ("pKa_ms", f"cluster-continuum n={args.nwat}")):
        if col not in rows[0]:
            continue
        shift, st = refit(col)
        a = st.get("anionic"); c = st.get("cationic")
        print(f"{lab:22}{shift:12.1f}{a[0]:12.1f}{a[1]:10.1f}{a[2]:11.1f}"
              f"{(c[2] if c else float('nan')):12.1f}")

    print(f"\n{'pair':16}{'group':11}{'|q|':>4}{'pKa_exp':>9}"
          + ("{:>11}".format("pKa_bare") if "pKa_bare" in rows[0] else "")
          + f"{'pKa_ms':>9}")
    for r in sorted(rows, key=lambda r: (r["kind"], r["group"])):
        line = f"{r['key']:16}{r['group']:11}{r['q']:4d}{r['pKa_exp']:9.2f}"
        if "pKa_bare" in r:
            line += f"{r['pKa_bare']:11.2f}"
        print(line + f"{r['pKa_ms']:9.2f}")

    print("\nby charge of the anion formed (after re-referencing on cations):")
    shift, _ = refit("pKa_ms")
    for q in (1, 2, 3, 4):
        e = [((r["pKa_ms"] - r["pKa_exp"]) * RTLN10 - shift)
             for r in rows if r["kind"] == "anionic" and r["q"] == q]
        if e:
            print(f"   q=-{q}  n={len(e):2d}  mean {np.mean(e):+7.1f}  MAE {np.mean(np.abs(e)):6.1f}")
    print("\n  (bare ALPB reference: q=-1 +41.9, q=-2 +19.9, q=-3 +16.3, q=-4 -34.7)")
    print("  decision rule: pursue explicit solvation only if anion MAE < ~10 kJ/mol")


if __name__ == "__main__":
    main()
