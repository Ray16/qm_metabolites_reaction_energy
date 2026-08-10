#!/usr/bin/env python
"""Arm 3 ensemble builder: solvent-consistent ranking + per-conformer thermostatistics.

Fixes the root cause of the MMFF path: conformers were selected for the expensive
GFN2/ALPB tier using GAS-PHASE MMFF energies, which mis-rank a solvated ensemble
(gas-phase overstabilizes intramolecular H-bonds ALPB screens out) and have no
parameters for phosphates/sulfates. Here the ranking layer is GFN-FF/ALPB -- same
solvent as the target, no parameter gaps, ~100x cheaper than GFN2.

  ETKDGv3 embed (MMFF only as geometry cleanup, NOT selection: NSTART=EMBED)
    -> xtb --gfnff --alpb water --opt loose   [rank in the target solvent]
    -> dedup (energy+RMSD) + gate: keep everything within GATE_KJ (~6 kcal/mol=25)
    -> xtb --gfn2 --alpb water --opt tight     [only survivors, not a fixed count]
    -> re-dedup + gate on GFN2 energy
    -> xtb --gfn2 --alpb water --bhess PER CONFORMER  [populations/entropy]

Reuses tested helpers from build_ensembles_fast. Env: same FAST_* vars, plus
FAST_GATE_KJ (25). Set FAST_NSTART=FAST_EMBED so MMFF does no selection.
Writes the same schema (+H_RRHO_kJ, TS_kJ per conformer) so downstream reads unchanged.
Run:  XTB_BIN=... FAST_*=... python build_ensembles_gfnff.py [cpd ...]
"""
import os, re, sys, json, shutil, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_ensembles_fast as bef      # reuses _xtb, xtb_opt, embed_mmff, write_xyz, dedup, etc.

GATE_KJ = float(os.environ.get("FAST_GATE_KJ", "25"))   # ~6 kcal/mol
# Per-conformer Hessians capture conformational entropy but cost ~10x. For a fast
# "does solvent-consistent SELECTION help?" pass, set FAST_PER_CONF_HESS=0 to use one
# shared Hessian on the lowest conformer (like the MMFF path). Turn on later for the
# entropy refinement on the ~20 species where it matters.
PER_CONF_HESS = os.environ.get("FAST_PER_CONF_HESS", "1") == "1"
T = 298.15
H2KJ = bef.HARTREE_TO_KJ


def _energy(stdout, wd):
    m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", stdout)
    if m:
        return float(m.group(1))
    xo = os.path.join(wd, "xtbopt.xyz")           # fallback: energy in comment line
    if os.path.isfile(xo):
        for tok in open(xo).readlines()[1].split():
            try:
                return float(tok)
            except ValueError:
                continue
    return None


def gfnff_opt(atoms, chg, wd):
    """GFN-FF/ALPB loose opt -> (opt_xyz_path, energy_Eh). Cheap ranking layer."""
    os.makedirs(wd, exist_ok=True)
    bef.write_xyz(atoms, os.path.join(wd, "in.xyz"), "etkdg")
    r = bef._xtb(["in.xyz", "--gfnff", "--alpb", "water", "--opt", "loose",
                  "--chrg", str(chg), "--uhf", "0"], wd)
    xo = os.path.join(wd, "xtbopt.xyz")
    e = _energy(r.stdout, wd)
    if e is None or not os.path.isfile(xo):
        return None
    return xo, e


def bhess(xtbopt_xyz, chg, wd):
    """Per-conformer GFN2/ALPB bhess -> (G_RRHO_kJ, H_RRHO_kJ, TS_kJ, n_imag, imag_cm)."""
    os.makedirs(wd, exist_ok=True)
    shutil.copy(xtbopt_xyz, os.path.join(wd, "in.xyz"))
    r = bef._xtb(["in.xyz", "--gfn", "2", "--alpb", "water", "--bhess",
                  "--chrg", str(chg), "--uhf", "0"], wd)
    def g(tag):
        m = re.search(tag + r"\s+(-?\d+\.\d+)", r.stdout)
        return float(m.group(1)) if m else None
    E, H, G = g("TOTAL ENERGY"), g("TOTAL ENTHALPY"), g("TOTAL FREE ENERGY")
    if None in (E, H, G):
        return None
    n_imag, imag_cm = bef._imag_modes(wd)
    return ((G - E) * H2KJ, (H - E) * H2KJ, (H - G) * H2KJ, n_imag, imag_cm)


def atoms_from_xyz(path):
    out = []
    for ln in open(path).read().splitlines()[2:]:
        p = ln.split()
        if len(p) >= 4:
            out.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return out


def process(meta):
    work = tempfile.mkdtemp(prefix=f"gfnff_{meta.cpd_id}_", dir=bef.SCRATCH)
    try:
        mol, cids = bef.embed_mmff(meta)            # NSTART=EMBED -> no MMFF selection
        # 1) GFN-FF/ALPB rank
        ranked = []
        with ThreadPoolExecutor(max_workers=bef.OPT_WORKERS) as ex:
            futs = {ex.submit(gfnff_opt, bef.conf_to_atoms(mol, c), meta.charge,
                              os.path.join(work, f"ff{k:03d}")): k for k, c in enumerate(cids)}
            for fut in as_completed(futs):
                res = fut.result()
                if res:
                    ranked.append(res)
        if not ranked:
            raise RuntimeError(f"{meta.cpd_id}: all GFN-FF opts failed")
        ranked.sort(key=lambda t: t[1])
        emin = ranked[0][1]
        gated = [(xo, e) for xo, e in ranked if (e - emin) * H2KJ <= GATE_KJ]
        # 2) GFN2/ALPB tight opt on survivors
        gfn2 = []
        with ThreadPoolExecutor(max_workers=bef.OPT_WORKERS) as ex:
            futs = [ex.submit(bef.xtb_opt, atoms_from_xyz(xo), meta.charge,
                              os.path.join(work, f"g2_{i:03d}")) for i, (xo, _e) in enumerate(gated)]
            for fut in as_completed(futs):
                res = fut.result()
                if res:
                    gfn2.append(res)
        if not gfn2:
            raise RuntimeError(f"{meta.cpd_id}: all GFN2 opts failed")
        # 3) dedup + gate on GFN2 aqueous energy
        gfn2.sort(key=lambda r: r["e_alpb_Eh"])
        e0 = gfn2[0]["e_alpb_Eh"]
        kept = []
        for r in gfn2:
            rel = (r["e_alpb_Eh"] - e0) * H2KJ
            if rel > GATE_KJ:
                break
            geom = bef.read_xyz_heavy(r["xtbopt"])
            if any(abs(rel - k["rel_kJ"]) < bef.DEDUP_KJ
                   and bef.kabsch_rmsd(geom, k["geom"]) < bef.DEDUP_RMSD for k in kept):
                continue
            r["rel_kJ"] = rel; r["geom"] = geom; kept.append(r)
        # 4) thermostatistics
        cdir = os.path.join(bef.ENS_DIR, meta.cpd_id)
        os.makedirs(cdir, exist_ok=True)
        records = []
        if PER_CONF_HESS:
            with ThreadPoolExecutor(max_workers=bef.OPT_WORKERS) as ex:
                futs = {ex.submit(bhess, r["xtbopt"], meta.charge,
                                  os.path.join(work, f"hess{i:03d}")): (i, r) for i, r in enumerate(kept)}
                hres = {}
                for fut in as_completed(futs):
                    i, r = futs[fut]; hres[i] = (r, fut.result())
            for i in sorted(hres):
                r, hb = hres[i]
                if hb is None:
                    continue
                g_rrho, h_rrho, tS, n_imag, imag_cm = hb
                dest = os.path.join(cdir, f"conf_{len(records):03d}.xyz")
                shutil.copy(r["xtbopt"], dest)
                records.append(dict(conf=len(records), xyz=dest, dGsolv_kJ=r["dGsolv_kJ"],
                                    G_RRHO_kJ=g_rrho, H_RRHO_kJ=h_rrho, TS_kJ=tS,
                                    S_kJ_per_K=tS / T, g_tot_kJ=None, rel_kJ=r["rel_kJ"],
                                    n_imag=n_imag, imag_cm=imag_cm))
        else:
            # one shared Hessian on the lowest conformer (fast; isolates selection effect)
            grrho, n_imag, imag_cm = bef.xtb_ohess_thermal(kept[0]["xtbopt"], meta.charge,
                                                           os.path.join(work, "hess"))
            for r in kept:
                dest = os.path.join(cdir, f"conf_{len(records):03d}.xyz")
                shutil.copy(r["xtbopt"], dest)
                records.append(dict(conf=len(records), xyz=dest, dGsolv_kJ=r["dGsolv_kJ"],
                                    G_RRHO_kJ=grrho, g_tot_kJ=None, rel_kJ=r["rel_kJ"],
                                    n_imag=n_imag if len(records) == 0 else -1,
                                    imag_cm=imag_cm if len(records) == 0 else 0.0))
        return records
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    os.makedirs(bef.ENS_DIR, exist_ok=True); os.makedirs(bef.SCRATCH, exist_ok=True)
    mets = bef.load_metabolites(bef.MET_JSON)
    only = set(sys.argv[1:])
    if only:
        mets = [m for m in mets if m.cpd_id in only]
    ens = json.load(open(bef.ENS_JSON)) if os.path.isfile(bef.ENS_JSON) else {}
    todo = [m for m in mets if bef.REDO or not ens.get(m.cpd_id)]
    print(f"=== GFN-FF/ALPB-ranked ensembles | embed={bef.EMBED} gate={GATE_KJ}kJ | "
          f"{len(todo)} to run ===", flush=True)
    with ThreadPoolExecutor(max_workers=bef.JOBS) as ex:
        futs = {ex.submit(process, m): m for m in todo}
        for fut in as_completed(futs):
            m = futs[fut]
            try:
                kept = fut.result()
                bef._checkpoint(ens, m.cpd_id, kept)
                print(f"  {m.cpd_id:9s} {m.name[:24]:24s} kept={len(kept):2d} "
                      f"(gate {GATE_KJ:.0f}kJ, per-conf bhess) [done]", flush=True)
            except Exception as e:
                print(f"  {m.cpd_id} FAILED: {e}", flush=True)


if __name__ == "__main__":
    main()
