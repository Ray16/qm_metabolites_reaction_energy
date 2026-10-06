"""Per-reaction uncertainty features for MetaG (structure, routing, solvation sensitivity; no annotations).

Every feature is computable at prediction time from the reaction input, the routed species the pipeline
scored, and the species cache -- no enzyme name, EC number, note, or reaction id. Built for the frozen
2026-10-01c TECRDB results (with experimental targets) and the 300-reaction ModelSEED panel (no labels).

    python features.py        -> features_tecrdb.json, features_modelseed.json
"""
import glob
import json
import os
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))                          # MetaG_new/ (self-contained)
ART = os.path.join(PKG, "artifacts")
TECRDB_RESULTS = os.path.join(ART, "results", "metag_opentecr_calibrated")     # exp = openTECR standardized
MODELSEED_RESULTS = os.path.join(ART, "results", "generality")
TECRDB_INPUT = os.path.join(PKG, "src", "metag", "data", "reactions_opentecr_std.json")
MODELSEED_INPUT = os.path.join(ART, "results", "generality_inputs.json")
# The per-species QM cache is a recompute input (~36 MB, GPU-bound), NOT a pinned paper artifact -- it is the
# one thing still in the old tree. Only regenerating the solvation-sensitivity features needs it; override
# with METAG_SPECIES_CACHE if you have it. Absence is non-fatal (those features fall back to empty).
CACHE = os.environ.get("METAG_SPECIES_CACHE",
                       os.path.join(os.path.dirname(PKG), "MetaG", "analysis", "sweep_20261001", "cache"))
PHYSICS = "2026-10-01c"

P = Chem.MolFromSmarts("[PX4]")
POP = Chem.MolFromSmarts("[PX4]-O-[PX4]")
PN = Chem.MolFromSmarts("[#7]-[PX4]")
CARBONYL = Chem.MolFromSmarts("[CX3](=[OX1])([#6,#1])[#6,#1]")          # aldehyde / ketone
ALDEHYDE = Chem.MolFromSmarts("[CX3H1](=O)[#6]")
KETO_ACID = Chem.MolFromSmarts("[CX3](=[OX1])[CX3](=O)[OX2H1,OX1-]")
ENOLIZABLE = Chem.MolFromSmarts("[CX4;!H0][CX3](=[OX1])[#6]")
HEMIACETAL = Chem.MolFromSmarts("[OX2;R][CX4;R][OX2H1]")                 # cyclic sugar anomeric C
AMINE = Chem.MolFromSmarts("[NX3,NX4+;!$(NC=O);!$(N-a);!$(N=*)][CX4]")
AROM_N = Chem.MolFromSmarts("[n]")
GUAN = Chem.MolFromSmarts("[NX3][CX3](=[NX2,NX3+])[NX3]")
THIOESTER = Chem.MolFromSmarts("[CX3](=O)[SX2]")
AMIDE = Chem.MolFromSmarts("[CX3](=[OX1])[NX3]")
ANION_O = Chem.MolFromSmarts("[OX1-]")
CATION_N = Chem.MolFromSmarts("[#7+;!H0]")


def species_cache():
    idx = defaultdict(dict)
    if not os.path.isdir(CACHE):                       # recompute input, not pinned; absence is non-fatal
        return idx
    for path in glob.glob(os.path.join(CACHE, "*.json")):
        d = json.load(open(path))
        st = d.get("settings", {})
        if st.get("physics") == PHYSICS and d.get("method") == "implicit":
            idx[(d["smi"], d["q"])][st.get("solv")] = d["G"]
    return idx


def _mol(smi):
    m = Chem.MolFromSmiles(smi)
    return m


def _net(species, patt):
    t = 0.0
    for c, q, s in species.values():
        m = _mol(s)
        if m is not None:
            t += c * len(m.GetSubstructMatches(patt))
    return t


def _gross(species, patt):
    t = 0.0
    for c, q, s in species.values():
        m = _mol(s)
        if m is not None:
            t += abs(c) * len(m.GetSubstructMatches(patt))
    return t


def structure_features(species):
    """Features of the ORIGINAL (pH-7 input) reaction."""
    mols = [(c, q, _mol(s)) for c, q, s in species.values()]
    mols = [(c, q, m) for c, q, m in mols if m is not None]
    heavy = [m.GetNumHeavyAtoms() for _, _, m in mols]
    rot = [rdMolDescriptors.CalcNumRotatableBonds(m) for _, _, m in mols]
    nP = [len(m.GetSubstructMatches(P)) for _, _, m in mols]
    f = {
        "n_species": len(mols),
        "max_heavy": max(heavy or [0]),
        "sum_heavy": sum(abs(c) * h for (c, _, _), h in zip(mols, heavy)),
        "max_rotb": max(rot or [0]),
        "max_abs_charge": max([abs(q) for _, q, _ in mols] or [0]),
        "gross_charge": sum(abs(c * q) for c, q, _ in mols),
        "net_abs_charge_change": abs(sum(c * q for c, q, _ in mols)),
        "n_multiP_species": sum(1 for p in nP if p >= 2),
        "max_P_per_species": max(nP or [0]),
        "gross_P": sum(abs(c) * p for (c, _, _), p in zip(mols, nP)),
        "d_POP": abs(_net(species, POP)),
        "d_PN": abs(_net(species, PN)),
        "d_carbonyl": abs(_net(species, CARBONYL)),
        "d_aldehyde": abs(_net(species, ALDEHYDE)),
        "d_ketoacid": abs(_net(species, KETO_ACID)),
        "d_enolizable": abs(_net(species, ENOLIZABLE)),
        "d_hemiacetal": abs(_net(species, HEMIACETAL)),
        "d_amine": abs(_net(species, AMINE)),
        "d_aromN": abs(_net(species, AROM_N)),
        "d_guan": abs(_net(species, GUAN)),
        "d_thioester": abs(_net(species, THIOESTER)),
        "d_amide": abs(_net(species, AMIDE)),
        "d_anionO": abs(_net(species, ANION_O)),
        "d_cationN": abs(_net(species, CATION_N)),
        "gross_anionO": _gross(species, ANION_O),
    }
    return f


def scored_features(rec, cache):
    """Features of the routed reaction the pipeline actually scored + solvation-model sensitivity."""
    sp = rec["species_scored"]
    sp = list(sp.items()) if isinstance(sp, dict) else sp
    routes = rec.get("routes") or {}
    stages = rec.get("stages") or {}
    tot = {"alpb": 0.0, "cpcmx": 0.0, "cosmo": 0.0}
    complete = True
    for _, (c, q, s) in sp:
        if s == "O" and q == 0:
            continue                         # liquid water: experimental reference, solvent-independent
        e = cache.get((s, q), {})
        for k in tot:
            if k in e:
                tot[k] += c * e[k]
            else:
                complete = False
    pka = stages.get("pka_transform", 0.0) or 0.0
    f = {
        "U_samp": float(rec.get("U_samp") or 0.0),
        "abs_pka_transform": abs(pka),
        "truncated": float(bool(routes.get("truncated"))),
        "trunc_radius3": float(routes.get("trunc_radius") == 3),
        "ntp_core": float(bool(routes.get("ntp_core"))),
        "cofactor_ring": float(bool(routes.get("cofactor_ring"))),
        "prefer_full": float(bool(routes.get("prefer_full"))),
        "ph0": float(bool(routes.get("ph0"))),
        "n_route_warnings": len(routes.get("warnings") or []),
        "scored_max_heavy": max([(_mol(s).GetNumHeavyAtoms() if _mol(s) else 0) for _, (c, q, s) in sp] or [0]),
        "scored_gross_charge": sum(abs(c * q) for _, (c, q, s) in sp),
        "solv_complete": float(complete),
    }
    if complete:
        a, x, c = tot["alpb"], tot["cpcmx"], tot["cosmo"]
        f["solv_d_cpcmx"] = abs(x - a)
        f["solv_d_cosmo"] = abs(c - a)
        f["solv_spread"] = max(a, x, c) - min(a, x, c)
    else:
        f["solv_d_cpcmx"] = f["solv_d_cosmo"] = f["solv_spread"] = None
    return f


def build(results_dir, input_path, cache, labelled):
    inputs = json.load(open(input_path))
    rows = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        rec = json.load(open(path))
        rid = rec["reaction"]
        if rec.get("dG") is None or rid not in inputs:
            continue
        row = {"dG": rec["dG"], "sigma_pred_frozen": rec.get("sigma_pred"),
               "ci95_frozen": rec.get("ci95"), "class_frozen": rec.get("class")}
        if labelled:
            row["exp"] = rec["exp"]
            row["err"] = rec["dG"] - rec["exp"]
        row.update(structure_features(inputs[rid]["species"]))
        row.update(scored_features(rec, cache))
        rows[rid] = row
    return rows


def main():
    cache = species_cache()
    for name, res, inp, lab in (("tecrdb", TECRDB_RESULTS, TECRDB_INPUT, True),
                                ("modelseed", MODELSEED_RESULTS, MODELSEED_INPUT, False)):
        rows = build(res, inp, cache, lab)
        out = os.path.join(HERE, f"features_{name}.json")
        json.dump(rows, open(out, "w"), indent=1)
        n_solv = sum(r["solv_complete"] for r in rows.values())
        print(f"{name}: {len(rows)} reactions, solvent spread available for {int(n_solv)}")


if __name__ == "__main__":
    main()
