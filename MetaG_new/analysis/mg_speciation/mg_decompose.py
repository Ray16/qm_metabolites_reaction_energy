"""Decompose the openTECR benchmark Legendre transform into pH/I and Mg2+ parts, per measurement.

Replicates MetaG/analysis/build_opentecr_standard.py row selection exactly (imports it), then for every used
measurement computes
  corr_total = dG'0_eQ(7, 0, pMg14) - dG'0_eQ(pH, I, pMg_reported)   (what the builder applied)
  corr_noMg  = dG'0_eQ(7, 0, pMg14) - dG'0_eQ(pH, I, 14)            (same, Mg ignored)
  mg_part    = corr_total - corr_noMg
It also flags rows whose free-text ('additional data' / table comment) mentions Mg but carry no p_mg (Mg present
but NOT transformed), and estimates an UPPER BOUND for those using pMg = -log10(c_Mg,total) (all Mg free).
Run: ~/miniforge3/envs/eqapi/bin/python mg_decompose.py
"""
import csv, json, math, os, re, statistics, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
TC = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(TC, "MetaG", "analysis"))
import build_opentecr_standard as O
B = O.B

MG_RE = re.compile(r"Mg", re.I)
CONC_RE = re.compile(r"c\((?:Mg[^)]*)\)\s*=\s*([0-9.Ee+-]+)\s*(m?M)")


def mg_total_M(text):
    m = CONC_RE.search(text or "")
    if not m:
        return None
    v = float(m.group(1)); return v * (1e-3 if m.group(2) == "mM" else 1.0)


def main():
    R = json.load(open(O.ALL_INP))
    kegg = B.kegg_to_modelseed(B.MSDB); ms = B.load_modelseed_reactions(B.MSDB)
    index = defaultdict(list)
    for rid, st in ms.items():
        index[B.signature(st, {B.PROTON, B.WATER})].append((rid, +1))
        index[B.signature({c: -v for c, v in st.items()}, {B.PROTON, B.WATER})].append((rid, -1))
    comments = {O._key(c): c["comment"] for c in csv.DictReader(open(os.path.join(O.SRC, "sheet_table_comments.csv")))}
    rows, _ = O.opentecr_rows(); tr = B.Transformer(ms)
    out = []
    for r, keq, src in rows:
        eq, ok = B.parse_equation(keq)
        if not ok or any(k not in kegg for k in eq):
            continue
        hits = index.get(B.signature({kegg[k]: v for k, v in eq.items()}, {B.PROTON, B.WATER}), [])
        if not hits:
            continue
        rid, orient = sorted(hits)[0]
        if rid not in R or O.usable(r) or tr.reaction(rid) is None:
            continue
        Kp, ph = B._f(r["K_prime"]), B._f(r["p_h"]); T = B._f(r["temperature"]) or 298.15
        I, pmg = B._f(r["ionic_strength"]), B._f(r["p_mg"]); Iu = 0.25 if I is None else I
        dg_obs = orient * -B.R_KJ * T * math.log(Kp)
        c_tot = tr.correction(rid, ph, Iu, 14.0 if pmg is None else pmg, T)
        c_no = tr.correction(rid, ph, Iu, 14.0, T)
        add, com = r["additional data"] or "", comments.get(O._key(r), "")
        cmg = mg_total_M(add) or mg_total_M(com)
        rec = dict(rid=rid, pH=ph, I=I, pMg=pmg, T=T, dg_obs=dg_obs, corr_total=c_tot, corr_noMg=c_no,
                   mg_part=c_tot - c_no, mg_text_additional=bool(MG_RE.search(add)), mg_text_comment=bool(MG_RE.search(com)),
                   c_Mg_total_M=cmg, additional=add[:200])
        if pmg is None and cmg and cmg > 0:                              # Mg present but untransformed: upper bound
            rec["untransformed_mg_upper"] = tr.correction(rid, ph, Iu, -math.log10(cmg), T) - c_no
        out.append(rec)
    json.dump(out, open(os.path.join(HERE, "per_measurement.json"), "w"), indent=0)
    print("measurements", len(out))


if __name__ == "__main__":
    main()
