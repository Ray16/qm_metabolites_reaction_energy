"""ModelSEED thermodynamic-coverage landscape quoted in the manuscript Introduction.

Counts non-obsolete, non-transport ModelSEED reactions; the fraction flagged mass/charge-imbalanced
(status MI/CI); balanced reactions with no stored GC or eQuilibrator estimate; and GC-vs-eQ
disagreement where both are stored. The stored `thermodynamics` values are kcal/mol (see memory
note "cached ModelSEED thermo is kcal"), so thresholds are applied after x4.184 -> kJ/mol.

    python landscape.py [--db DIR] [--out JSON]
"""
import argparse, glob, json, os

KCAL = 4.184
HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "..", "..", "..", "..", "ModelSEEDDatabase", "Biochemistry")


def num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if abs(x) < 1e6 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--out", default=os.path.join(HERE, "landscape.json"))
    a = ap.parse_args()
    rx = [r for f in sorted(glob.glob(os.path.join(a.db, "reaction_*.json"))) for r in json.load(open(f))]
    live = [r for r in rx if not r.get("is_obsolete") and not r.get("is_transport")]
    status = lambda r: r.get("status") or ""
    imbalanced = [r for r in live if "MI" in status(r) or "CI" in status(r)]
    formula_err = [r for r in live if status(r) == "CPDFORMERROR"]
    unscoreable = len(imbalanced) + len(formula_err)
    balanced = [r for r in live if not ("MI" in status(r) or "CI" in status(r)) and status(r) != "CPDFORMERROR"]
    th = lambda r: r.get("thermodynamics") or {}
    has = lambda v: isinstance(v, (list, tuple)) and len(v) > 0 and num(v[0]) is not None
    no_est = [r for r in balanced if num(r.get("deltag")) is None]          # no stored deltag at all
    both = [r for r in live if has(th(r).get("Group contribution")) and has(th(r).get("eQuilibrator"))]
    d_kj = [abs(num(th(r)["Group contribution"][0]) - num(th(r)["eQuilibrator"][0])) * KCAL for r in both]
    out = {
        "n_reactions_total": len(rx), "n_live": len(live),
        "n_imbalanced_MI_CI": len(imbalanced), "n_formula_error": len(formula_err),
        "frac_unscoreable": unscoreable / len(live),
        "n_balanced": len(balanced), "n_balanced_no_estimate": len(no_est),
        "n_both_estimates": len(both),
        "frac_gc_eq_gt10_kJ": sum(x > 10 for x in d_kj) / len(both),
        "frac_gc_eq_gt30_kJ": sum(x > 30 for x in d_kj) / len(both),
        "frac_gc_eq_gt10_kcal_as_kJ_BUG": sum(x / KCAL > 10 for x in d_kj) / len(both),
        "frac_gc_eq_gt30_kcal_as_kJ_BUG": sum(x / KCAL > 30 for x in d_kj) / len(both),
    }
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
