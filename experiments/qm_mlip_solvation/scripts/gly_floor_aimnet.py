"""Glycosyl-floor localization, step 2 (aimnet2 env): re-score the UMA-relaxed geometries
with AIMNet2 (independent wB97M-D3-trained potential). Compare reaction ΔE_elec to UMA."""
import os, sys, json, glob
import numpy as np

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "gly_floor")
EV2KJ = 96.485

def read_xyz(p):
    lines = open(p).read().splitlines()
    n = int(lines[0]); comment = lines[1]
    meta = dict(kv.split("=") for kv in comment.split() if "=" in kv)
    q = int(meta["charge"]); coeff = int(float(meta["coeff"]))
    sym, pos = [], []
    for ln in lines[2:2 + n]:
        s, x, y, z = ln.split()
        sym.append(s); pos.append([float(x), float(y), float(z)])
    return sym, np.array(pos), q, coeff

def main():
    from aimnet2calc import AIMNet2ASE
    from ase import Atoms
    rows = {}
    calc = None
    for p in sorted(glob.glob(os.path.join(ART, "*.xyz"))):
        name = os.path.splitext(os.path.basename(p))[0]
        sym, pos, q, coeff = read_xyz(p)
        atoms = Atoms(symbols=sym, positions=pos)
        ase_calc = AIMNet2ASE("aimnet2", charge=q)
        atoms.calc = ase_calc
        E_kj = float(atoms.get_potential_energy()) * EV2KJ
        rows[name] = dict(coeff=coeff, charge=q, aimnet_E_kj=round(E_kj, 3))
        print(f"{name:10s} q{q:+d} AIMNet2_E {E_kj:.1f} kJ", flush=True)
    dE = sum(r["coeff"] * r["aimnet_E_kj"] for r in rows.values())
    print(f"\nAIMNet2 gas ΔE_elec(reaction) = {dE:+.1f} kJ/mol", flush=True)
    json.dump(rows, open(os.path.join(ART, "aimnet.json"), "w"), indent=2)
    # side-by-side vs UMA
    uma = json.load(open(os.path.join(ART, "uma.json")))
    dE_uma = sum(r["coeff"] * r["uma_E_kj"] for r in uma.values())
    print(f"\n=== ΔE_elec(reaction):  UMA {dE_uma:+.1f}   AIMNet2 {dE:+.1f}   "
          f"diff {dE - dE_uma:+.1f} kJ/mol ===", flush=True)
    print("(exp ΔG'° = -9.51; if UMA~AIMNet2 the +11 floor is NOT electronic)", flush=True)

if __name__ == "__main__":
    main()
