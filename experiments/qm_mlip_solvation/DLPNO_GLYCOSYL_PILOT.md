# DLPNO-CCSD(T) glycosyl-ceiling pilot (rxn01362 orotate PRTase) — 2026-08-17

Feasibility probe: does DLPNO-CCSD(T) move the glycosyl-core reaction energy *beyond* the DFT/UMA
plateau? (Prior probe: wB97M-V ≈ UMA to +0.7 kJ → DFT is not enough; only DLPNO could move it.)
CPU/ORCA only, no GPU touched, nothing committed.

## (a) Does ORCA DLPNO run in this env? — YES, cleanly
- `source thermodynamic_calc/env.sh` puts ORCA 6.1.1 + OpenMPI 4.1.8 on PATH; call by absolute
  `$ORCA_ROOT/orca`. Trivial DLPNO-CCSD(T)/cc-pVDZ + CPCM(water) on H₂O: 4 s, −76.25002 Eh, normal term.
- All 8 production single points (4 cores × {DLPNO, DFT}) TERMINATED NORMALLY.

## (b) Does DLPNO move the glycosyl core vs DFT? — YES, ~13 kJ in the correcting direction
Core reaction (truncated radius-2, neutralised, xtb/GFN2-alpb-opt geometries, single conformer):
`PPi + Orotidylate → PRPP + Orotate`, both methods + CPCM(water) on the SAME geometry (so solvation cancels):

| method | core ΔE (kJ/mol) |
|---|---|
| wB97X-D3 / def2-SVP + CPCM | **+89.8** |
| DLPNO-CCSD(T) / cc-pVDZ + CPCM | **+77.2** |
| **DLPNO − DFT** | **−12.6** |

- UMA (pH-0 route) over-predicts this reaction by **+49.4** (ΔG +44.1 vs exp −5.3). DFT ≈ UMA (prior).
  DLPNO is **12.6 kJ LESS endergonic** than DFT → it moves in the **error-correcting direction** and would
  cut the +49 error to ~+36 on this single-conformer estimate. So the ceiling is REAL and DLPNO-addressable
  (unlike DFT, which sat on top of UMA). A correlation-level effect, not a solvation artifact (CPCM cancels).
- **Caveat — this is a LOWER bound.** cc-pVDZ badly under-converges the CCSD(T) correlation energy; the
  DLPNO−DFT gap should GROW with cc-pVTZ / CBS extrapolation. Single geometry + single conformer here; the
  production number needs the conformer ensemble the UMA pipeline uses.

## (c) Cost & go/no-go
- DLPNO-CCSD(T)/cc-pVDZ + CPCM, PAL8, per core: PPi(13 at) 2 min, orotate(15) 3.4 min, PRPP(25) 7.7 min,
  orotidylate(27) 13.4 min. DFT reference: <1 min each.
- Glycosyl class ≈ 15 reactions × ~4 core species ≈ 60 single points. 8-wide parallel on this 80-core node
  → a few hours wall. With cc-pVTZ, multiply ~3–5×; still a single-overnight job. **Tractable.**
- **GO** for a proper glycosyl-class DLPNO run, with two upgrades over this pilot: (1) cc-pVTZ or
  cc-pVDZ→cc-pVTZ 2-point CBS for the correlation energy; (2) score the SAME conformer ensemble UMA uses
  (Boltzmann), not one geometry. Expectation: DLPNO partially-to-substantially closes the glycosyl error;
  this pilot shows ~13 kJ of the ~24–49 kJ is recoverable from correlation at the *smallest* basis.

## Method notes / reproducibility
- Geometries: RDKit ETKDG → MMFF → xtb `--opt --alpb water --gfn 2` (neutral protonation state, matching
  the pH-0 route). Files in `/tmp/qm_thermo_scratch/dlpno_pilot/*_opt.xyz` (scratch — not persisted).
- ORCA keywords: `! DLPNO-CCSD(T) cc-pVDZ cc-pVDZ/C RIJCOSX def2/J TightPNO CPCM(water) PAL8` and
  `! wB97X-D3 def2-SVP def2/J RIJCOSX CPCM(water) PAL8`, `%maxcore 3000`.
- Closed-shell singlets throughout (charge 0, mult 1). Solvation identical between the two methods → the
  DLPNO−DFT difference is purely the electronic-correlation upgrade.
