"""CBH Phase 2+3: compute the small-molecule library energies and validate the correction.

For each library fragment: G_implicit (UMA + ALPB/cosmo continuum) and, for CHARGED fragments,
G_micro (explicit-water cluster-continuum microsolvation). δ(frag) = G_micro − G_implicit.
Then Δcorr(reaction) = Σ_frag n_frag·δ(frag);  ΔG_corr = ΔG_UMA_logged + Δcorr.

VALIDATION built in:
  - deamination cluster: err should collapse toward 0 (Δcorr ≈ +43 needed).
  - alcohol-DH GUARD: must STAY ≈ its small raw error (the NAD-ring fragments must not inject a
    spurious shift). If the guard breaks, the aromatic ring fragments need excluding.
Experiment-free: δ comes only from UMA±microsolvation, never from TECRDB ΔG.
"""
import os, sys, json, re, importlib.util
from collections import Counter
import numpy as np
from rdkit import Chem

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.join(HERE, "..", "scripts")
sys.path.insert(0, SCR)
import unified_pipeline as up            # loads its imports (UMA, implicit_G, explicit_G, water_ref_G)

def _load(n, f):
    s = importlib.util.spec_from_file_location(n, os.path.join(SCR, f)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
cbh = _load("cbh", "cbh_correct.py")

RINGCO = json.load(open(os.path.join(SCR, "reactions_ringcofactor_all.json")))
TEC = json.load(open(os.path.join(SCR, "reactions_tecrdb_all.json")))
LIBJSON = os.path.join(HERE, "..", "artifacts", "cbh_library.json")

# validation reactions: deamination targets + alcohol-DH guards (all redox -> ringcofactor route)
DEAM = ["rxn00184", "rxn00278", "rxn00804", "rxn02930", "rxn00182"]
ALC_GUARD = ["rxn01846", "rxn00741", "rxn01130"]   # malate / lactate / alcohol DH — small raw error

def charge_of(smi):
    m = Chem.MolFromSmiles(smi)
    return sum(a.GetFormalCharge() for a in m.GetAtoms())

def logged_dG(rid):
    p = os.path.join(SCR, "..", "logs", "ringcofactor", f"{rid}.log")
    if not os.path.exists(p):
        p = os.path.join(SCR, "..", "logs", "ph0_sweep", f"{rid}.log")
    m = re.search(r"ΔG = ([+-]?\d+\.\d+)", open(p, errors="ignore").read())
    return float(m.group(1)) if m else None

def build_library():
    # union library over the validation reactions
    lib = Counter()
    for rid in DEAM + ALC_GUARD:
        src = RINGCO if rid in RINGCO else TEC
        for fr in cbh.reaction_reference(src[rid]["species"]):
            lib[fr] += 1
    frags = sorted(lib)
    print(f"library: {len(frags)} unique fragments over {len(DEAM+ALC_GUARD)} validation reactions")
    pu = up.load_uma()
    L = json.load(open(LIBJSON)) if os.path.exists(LIBJSON) else {}   # resume completed frags
    for i, smi in enumerate(frags):
        if smi in L and L[smi].get("G_implicit") is not None:
            print(f"[{i+1}/{len(frags)}] cached {smi}", flush=True); continue
        q = charge_of(smi)
        print(f"[{i+1}/{len(frags)}] computing {smi} q{q:+d} ...", flush=True)
        gi, _ = up.implicit_G(pu, q, smi, [1], 10, 48, lambda s: print(s, flush=True), smi[:9])
        entry = {"charge": q, "G_implicit": gi, "G_micro": None}
        if q != 0 and gi is not None:
            try:
                gm, _ = up.explicit_G(pu, q, smi, [1], lambda s: print(s, flush=True), smi[:9])
                entry["G_micro"] = gm
            except Exception as e:
                print(f"   micro FAILED {smi}: {e}", flush=True)
        entry["delta"] = (entry["G_micro"] - gi) if (entry["G_micro"] is not None and gi is not None) else 0.0
        L[smi] = entry
        json.dump(L, open(LIBJSON, "w"), indent=1)                    # incremental save
        print(f"  δ {smi:24s} q{q:+d}  Gimp {gi}  Gmicro {entry['G_micro']}  δ={entry['delta']:.1f}", flush=True)
    return L

def dcorr(rid, L):
    src = RINGCO if rid in RINGCO else TEC
    net = cbh.reaction_reference(src[rid]["species"])
    return sum(n * L[smi]["delta"] for smi, n in net.items())

def validate(L):
    print("\n=== VALIDATION ===")
    for label, rids in [("DEAMINATION (target: err -> ~0)", DEAM), ("ALCOHOL-DH GUARD (must stay small)", ALC_GUARD)]:
        print(f"\n{label}")
        for rid in rids:
            src = RINGCO if rid in RINGCO else TEC
            exp = src[rid]["exp"][0]; dg = logged_dG(rid)
            if dg is None: print(f"  {rid}: no log"); continue
            dc = dcorr(rid, L)
            print(f"  {rid}  ΔG_UMA {dg:+7.1f}  Δcorr {dc:+6.1f}  ->  ΔG_corr {dg+dc:+7.1f}  exp {exp:+6.1f}  "
                  f"| err {dg-exp:+6.1f} -> {dg+dc-exp:+6.1f}")

if __name__ == "__main__":
    if os.path.exists(LIBJSON) and "--rebuild" not in sys.argv:
        L = json.load(open(LIBJSON)); print(f"loaded cached library ({len(L)} frags)")
    else:
        L = build_library()
    validate(L)
