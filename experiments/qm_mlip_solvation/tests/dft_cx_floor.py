"""Attack the C-X electronic floor with DFT at OMol25's OWN level of theory (wB97M-V/def2-TZVPD)
on the TRUNCATED reactive core (small enough thanks to truncation). Since UMA is trained to
approximate wB97M-V, computing wB97M-V directly gives the reference UMA is (poorly) approximating
on the anomeric/glycosidic bond. Reads the saved UMA-relaxed truncated rxn00605 core geometries.
Compares ΔE_elec(DFT) to UMA's +60.5 gas ΔE_elec. CPU; density-fitted for speed.
Truncated rxn00605: glycoside + Pi - donorCap - G6Pt  (glucosyl transfer, exp ΔG -9.51)."""
import os, glob, numpy as np
from pyscf import gto, dft
H2KJ = 2625.4996

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "gly_floor")
SPECIES = {"donorCap": -1, "G6Pt": -1, "glycoside": +1, "Pi": +1}   # coeff by role

def read_xyz(p):
    L = open(p).read().splitlines(); n = int(L[0])
    meta = dict(kv.split("=") for kv in L[1].split() if "=" in kv)
    q = int(meta["charge"]); umaE = float(meta.get("uma_E_kj", "nan"))
    atoms = []
    for ln in L[2:2 + n]:
        s, x, y, z = ln.split(); atoms.append((s, (float(x), float(y), float(z))))
    return atoms, q, umaE

def dft_energy(atoms, q):
    mol = gto.M(atom=[(s, xyz) for s, xyz in atoms], basis="def2-tzvpd",
                charge=q, spin=0, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "wb97m-v"                      # OMol25 functional (VV10 NLC auto-enabled)
    mf = mf.density_fit()                  # RI for speed
    e = mf.kernel()
    return float(e) * H2KJ, mf.converged

def main():
    dft_E, uma_E = {}, {}
    for name, coeff in SPECIES.items():
        p = os.path.join(ART, name + ".xyz")
        atoms, q, umaE = read_xyz(p)
        e, conv = dft_energy(atoms, q)
        dft_E[name] = e; uma_E[name] = umaE
        print(f"{name:10s} q{q:+d} natom={len(atoms)}  DFT {e:.1f} kJ (conv={conv})  UMA {umaE:.1f}", flush=True)
    dE_dft = sum(SPECIES[n] * dft_E[n] for n in SPECIES)
    dE_uma = sum(SPECIES[n] * uma_E[n] for n in SPECIES)
    print(f"\n=== gas ΔE_elec(reaction) ===")
    print(f"  DFT wB97M-V/def2-TZVPD : {dE_dft:+.1f} kJ/mol")
    print(f"  UMA (approx of same)   : {dE_uma:+.1f} kJ/mol")
    print(f"  DFT - UMA              : {dE_dft - dE_uma:+.1f} kJ/mol  (UMA's electronic error)")
    print(f"\n  NOTE: full ΔG = ΔE_elec + thermal(UMA) + ΔΔGsolv(xtb). UMA-composite gave ΔG +12.6")
    print(f"  (err vs exp -9.51). If DFT ΔE_elec shifts by ~-X, the corrected ΔG moves by ~-X.")

if __name__ == "__main__":
    main()
