"""Where is the UMA pipeline still lacking? Current per-reaction error (logs/production + wired aldehyde
correction), CLASSIFIED by mechanism, ranked by contribution to the total MAE. Answers: which chemistry
classes carry the remaining error, and what to attack next."""
import os, re, sys, json
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(EXP, "scripts"))
import ph0_auto as pfa
import route_anchor as ra
import aldehyde_hydration as ah
d = json.load(open(os.path.join(EXP, "scripts", "reactions_tecrdb_all.json")))


def uma_dG(rid):
    p = os.path.join(EXP, "logs", "production", f"{rid}.log")
    if not os.path.exists(p): return None
    m = re.search(r"ΔG = ([+-]?\d+\.\d+)", open(p, errors="ignore").read())
    return float(m.group(1)) if m else None


def ald_delta(rid):
    p = os.path.join(EXP, "logs", "ah367_on", f"{rid}.log")
    if not os.path.exists(p): return 0.0
    dd = 0.0
    for m in re.finditer(r"\[hydration: (\S+) carbonyl.*?shift ([+-][0-9.]+)\]", open(p, errors="ignore").read()):
        nm, sh = m.group(1), float(m.group(2))
        for sp, (c, q, s) in d[rid]["species"].items():
            if nm == sp or nm == sp + "_t" or nm[:8] == sp[:8]:
                if ah.is_strongly_hydrated(s): dd += c * sh
                break
    return dd


PN = Chem.MolFromSmarts("[#7]-[PX4](=O)")           # phosphoramidate (phosphagen)
POP = Chem.MolFromSmarts("[PX4]-O-[PX4]")            # pyrophosphate / NTP
PANION = Chem.MolFromSmarts("[PX4](=O)([O-])[O-]")   # phosphate dianion
COO = Chem.MolFromSmarts("[CX3](=O)[O-]")
THIOEST = Chem.MolFromSmarts("[#6](=O)[SX2]")        # thioester (CoA)


def _net(rid, patt):
    n = 0
    for c, q, s in d[rid]["species"].values():
        m = Chem.MolFromSmiles(s)
        if m: n += c * len(m.GetSubstructMatches(patt))
    return n


def classify(rid):
    v = d[rid]; note = v.get("note", "").lower(); sp = v["species"]
    smis = " ".join(s for c, q, s in sp.values())
    def has(p): return any(Chem.MolFromSmiles(s) and Chem.MolFromSmiles(s).HasSubstructMatch(p) for c, q, s in sp.values())
    if ra.subclass(sp) == "phosphagen" or _net(rid, PN) != 0:
        return "phosphagen (P–N/Mg)"
    if "mg-prone" in note or (_net(rid, POP) != 0):
        return "Mg / phosphoanhydride (NTP,PPi)"
    if ("c1cc[n+]" in smis.lower() and "C(N)=O" in smis) or any(k in note for k in ["dehydrogenase", "reductase", "oxidase", "oxidoreductase"]):
        return "redox cofactor (NAD/FAD)"
    if has(THIOEST) or "coa" in smis.lower():
        return "CoA / thioester"
    if pfa.is_isomerization(sp):
        return "isomerase (conc-limited)"
    if _net(rid, PANION) != 0:
        return "phosphate anion created/destroyed"
    if _net(rid, COO) != 0 and ("nh" in smis.lower() or "[N" in smis):
        return "amine/carboxylate (deamination)"
    if "huge" in note or "floppy" in note:
        return "huge / floppy"
    # decompose the former "other" grab-bag by mechanism (EC + note)
    ec = v.get("EC") or ""
    if "hydratase" in note or "dehydratase" in note or ec.startswith(("4.2.1", "4.2.99")):
        return "hydratase/dehydratase (flexible)"     # open-chain form not a gas minimum -> solution MD
    if ec.startswith("4") or any(k in note for k in ("aldolase", "lyase", "cycloisomer")):
        return "lyase/aldolase (C–C/C–N)"
    if ec.startswith("2") or "phosphorylase" in note or "transferase" in note:
        return "transferase/phosphorylase"
    if ec.startswith("3") or "hydrolase" in note or "phosphatase" in note or "glycosid" in note:
        return "hydrolase (glycosidase/phosphatase)"
    return "other"


def main():
    rows = []
    for rid, v in d.items():
        u = uma_dG(rid)
        if u is None: continue
        u += ald_delta(rid)
        e = u - v["exp"][0]
        if abs(e) > 200: continue
        rows.append((rid, e, classify(rid)))
    err = np.array([abs(e) for _, e, _ in rows]); n = len(rows)
    print(f"n={n}   MAE={err.mean():.2f}   median={np.median(err):.2f}   |err|>20 tail={int((err>20).sum())}\n")

    # per-class contribution to total absolute error
    cls = defaultdict(list)
    for rid, e, c in rows: cls[c].append((rid, e))
    print(f"{'class':34s} {'n':>3s} {'MAE':>6s} {'median':>6s} {'%oftotal|err|':>13s}")
    tot = err.sum()
    order = sorted(cls, key=lambda c: -sum(abs(e) for _, e in cls[c]))
    for c in order:
        es = np.array([abs(e) for _, e in cls[c]])
        share = 100 * es.sum() / tot
        print(f"{c:34s} {len(es):3d} {es.mean():6.1f} {np.median(es):6.1f} {share:12.1f}%")

    print("\n=== worst 25 reactions (current) ===")
    for rid, e, c in sorted(rows, key=lambda r: -abs(r[1]))[:25]:
        print(f"  {rid} {e:+7.1f}  {c:32s} {d[rid].get('note','')[7:47]}")


if __name__ == "__main__":
    main()


def make_figure():
    """Horizontal bar: each class's contribution to the total |error| (kJ summed), i.e. where the MAE lives."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = []
    for rid, v in d.items():
        u = uma_dG(rid)
        if u is None: continue
        u += ald_delta(rid); e = u - v["exp"][0]
        if abs(e) <= 200: rows.append((rid, e, classify(rid)))
    cls = defaultdict(list)
    for rid, e, c in rows: cls[c].append(abs(e))
    tot = sum(sum(v) for v in cls.values())
    # sort by summed |err| (contribution)
    order = sorted(cls, key=lambda c: sum(cls[c]))
    labels = [f"{c}  (n={len(cls[c])}, MAE {np.mean(cls[c]):.0f})" for c in order]
    share = [100 * sum(cls[c]) / tot for c in order]
    WALL = {"Mg / phosphoanhydride (NTP,PPi)", "phosphagen (P–N/Mg)", "phosphate anion created/destroyed"}
    colors = ["#c0392b" if c in WALL else ("#95a5a6" if "isomerase" in c else "#2c7fb8") for c in order]
    fig, ax = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    y = np.arange(len(order))
    ax.barh(y, share, color=colors, edgecolor="white")
    for yi, s in zip(y, share):
        ax.text(s + 0.4, yi, f"{s:.0f}%", va="center", fontsize=11)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("share of total |error|  (%)", fontsize=13)
    ax.set_title(f"Where the {np.mean([abs(e) for _,e,_ in rows]):.1f} kJ/mol MAE lives  "
                 f"(red = anion/Mg wall ≈ {sum(s for c,s in zip(order,share) if c in WALL):.0f}%)", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False); ax.margins(x=0.08)
    p = os.path.join(EXP, "figures", "where_lacking_by_class.png")
    fig.savefig(p, dpi=300, bbox_inches="tight"); print(f"wrote {p}")
