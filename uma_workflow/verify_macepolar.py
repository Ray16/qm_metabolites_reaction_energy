#!/usr/bin/env python
"""Verify MACE-POLAR-1 is actually working before spending GPU time on it.

Four checks, in increasing strength:

  1. imports          -- mace exposes PolarMACE, graph_longrange present
  2. model loads      -- the released .model file deserialises onto the GPU
  3. charge coupling  -- the SAME geometry at different total charge must give
                         different energies. If it does not, the charge is not
                         reaching the model and every number afterwards would be
                         silently wrong. This is the check that actually matters
                         for us, because our species run from 0 to -4.
  4. DFT ground truth -- reproduce the isodesmic anchor reaction
                            mNA+ + 2 MeSH -> mNAH + MeSSMe
                         on the same xtb geometries used before, where we have
                         wB97M-V/def2-TZVPD reference: UMA 964.7, DFT 965.7
                         kJ/mol. A working model should land near 965; a number
                         far from that means the charge/spin convention or the
                         energy reference is being mishandled.

Run:  /homes/rzhu/miniforge3/envs/macepolar/bin/python verify_macepolar.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
BENCH = os.path.join(THERMO, "large_dGPredictor_error")
ANCHORS = os.path.join(BENCH, "geometries_anchors")
REF = os.path.join(os.path.dirname(THERMO), "backup", "thermo_superseded",
                   "data", "redox_anchor_correction.json")
MODEL = os.environ.get("MACE_POLAR_MODEL",
                       os.path.join(THERMO, "models", "MACE-POLAR-1-L.model"))
EV_TO_KJ = 96.48533212
CHARGES = {"mNAplus": 1, "mNAH": 0, "MeSH": 0, "MeSSMe": 0}


def ok(msg):
    print(f"  [ok]   {msg}")


def bad(msg):
    print(f"  [FAIL] {msg}")


def main():
    print("=== 1. imports ===")
    import mace
    print(f"  mace {mace.__version__}")
    have_polar = False
    try:
        from mace.modules.models import PolarMACE   # noqa: F401
        have_polar = True
        ok("PolarMACE importable from mace.modules.models")
    except Exception as e:
        try:
            import mace.modules as M
            names = [n for n in dir(M) if "olar" in n]
            if names:
                have_polar = True
                ok(f"PolarMACE found in mace.modules: {names}")
            else:
                bad(f"no PolarMACE symbol ({type(e).__name__})")
        except Exception as e2:
            bad(f"mace.modules import failed: {e2}")
    try:
        import graph_longrange  # noqa: F401
        ok("graph_longrange present")
    except Exception as e:
        bad(f"graph_longrange missing ({type(e).__name__}) -- PolarMACE needs it")
    if not have_polar:
        print("\nPolarMACE is not available in this build; the released model "
              "cannot be loaded. Stop here and pin a MACE revision that has it.")
        return 1

    print("\n=== 2. model loads ===")
    if not os.path.isfile(MODEL):
        bad(f"model file not found: {MODEL}")
        return 1
    # mace_polar sets model_type="PolarMACE"; a plain MACECalculator would
    # mis-load the polar checkpoint. It accepts a local path or a key
    # ("polar-1-s"/"-m"/"-l") and auto-downloads in the latter case.
    from mace.calculators import mace_polar
    calc = mace_polar(model=MODEL, device="cuda", default_dtype="float64")
    ok(f"loaded {os.path.basename(MODEL)}")

    print("\n=== 3. charge coupling (same geometry, different total charge) ===")
    from ase import Atoms

    def read(path):
        lines = open(path).read().splitlines()
        n = int(lines[0].split()[0])
        s, p = [], []
        for ln in lines[2:2 + n]:
            t = ln.split()
            s.append(t[0]); p.append([float(t[1]), float(t[2]), float(t[3])])
        return Atoms(symbols=s, positions=p)

    def energy(atoms, q):
        a = atoms.copy()
        a.info["charge"] = int(q)
        a.info["spin"] = 1
        a.calc = calc
        return float(a.get_potential_energy()) * EV_TO_KJ

    probe = os.path.join(ANCHORS, "MeSH", "xtbopt.xyz")
    if not os.path.isfile(probe):
        cand = [os.path.join(ANCHORS, "MeSH", f)
                for f in os.listdir(os.path.join(ANCHORS, "MeSH"))
                if f.endswith(".xyz")]
        probe = cand[0]
    at = read(probe)
    e0, e1 = energy(at, 0), energy(at, -1)
    d = e1 - e0
    print(f"  MeSH geometry: E(q=0) = {e0:.1f}   E(q=-1) = {e1:.1f}   diff = {d:+.1f} kJ/mol")
    if abs(d) < 1.0:
        bad("energy is INSENSITIVE to total charge -- charge is not reaching the "
            "model; do not trust any downstream number")
        return 1
    ok("energy responds to total charge")

    print("\n=== 4. isodesmic anchor vs wB97M-V ground truth ===")
    ref = json.load(open(REF))
    stoich = ref["R0_stoich"]
    E = {}
    for name, q in CHARGES.items():
        d_ = os.path.join(ANCHORS, name)
        xyz = os.path.join(d_, "xtbopt.xyz")
        if not os.path.isfile(xyz):
            xyz = os.path.join(d_, sorted(f for f in os.listdir(d_)
                                          if f.endswith(".xyz"))[0])
        E[name] = energy(read(xyz), q)
        print(f"  {name:9} q={q:+d}  E = {E[name]:14.1f} kJ/mol")
    dE = sum(c * E[n] for n, c in stoich.items())
    uma, dft = ref["dE_R0_uma_kJ"], ref["dE_R0_dft_kJ"]
    print(f"\n  dE(R0)   MACE-POLAR = {dE:8.1f}")
    print(f"           UMA        = {uma:8.1f}   (vs DFT {uma - dft:+.1f})")
    print(f"           wB97M-V    = {dft:8.1f}   <- ground truth")
    print(f"  MACE-POLAR - DFT = {dE - dft:+.1f} kJ/mol")
    if abs(dE - dft) < 20:
        ok("anchor reaction reproduced within 20 kJ/mol of hybrid DFT")
        return 0
    bad("anchor reaction far from DFT -- suspect charge/spin convention or "
        "energy reference (e.g. atomic E0s / isolated-atom offsets)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
