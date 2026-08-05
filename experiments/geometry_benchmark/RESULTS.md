# Does the geometry method matter? g-xTB vs GFN2 vs r2SCAN-3c

Run 2026-08-05 on pyrophosphate (PPi, q = -3) — the P–O–P motif that carries
the +64 kJ/mol per-phosphoanhydride bias found in the 130-reaction set.

Protocol: one ETKDG seed (0xC0FFEE), optimised three ways, then scored with
**identical** downstream terms (MACE-POLAR-1 gas single point + xtb-GFN2/ALPB
dGsolv single points at the given geometry). Geometry is the only variable.
G_RRHO held fixed; thermal corrections are insensitive at this scale.

## Geometry

| method | P–O_br | P–O–P | P···P | P–O_term | heavy RMSD vs DFT |
|---|---|---:|---:|---:|---:|
| GFN2/ALPB | 1.697 / 1.620 | 122.6° | 2.910 | 1.527 | **0.085 Å** |
| g-xTB/ddCOSMO | 1.680 / 1.607 | 118.6° | 2.826 | 1.509 | **0.193 Å** |
| r2SCAN-3c/CPCM | 1.716 / 1.642 | 125.8° | 2.989 | 1.547 | — |

The three are monotonic: g-xTB is more compact than GFN2, which is more compact
than DFT. **GFN2 is 2.3x closer to the reference than g-xTB.**

## Composite impact — the terms cancel

Relative to the r2SCAN-3c geometry:

| geometry | ΔE_MACE(gas) | Δ dGsolv | **net** |
|---|---:|---:|---:|
| GFN2/ALPB | +9.7 | −10.5 | **−0.8** |
| g-xTB/ddCOSMO | +47.1 | −44.0 | **+3.1** |

A more compact structure raises the gas energy (unscreened internal repulsion)
and lowers dGsolv (smaller cavity) by nearly the same amount. The composite is
therefore **geometry-robust**: GFN2 costs under 1 kJ/mol against a DFT
reference, despite a visible structural difference.

## Two conclusions

1. **The P–O–P geometry hypothesis is refuted.** Geometry contributes <1 kJ/mol
   to PPi's free energy. The +64 kJ/mol phosphoanhydride bias is electronic or
   solvation error in the bond change itself, not a distorted structure. Note
   the sign also points the wrong way: GFN2's dGsolv is 10.5 kJ/mol *too
   negative*, so correcting the geometry would make that bias marginally worse.
2. **g-xTB is not an upgrade here.** Worse polyanion geometries than GFN2, no
   usable solvation (see below), and 2–25x slower. Do not adopt for this role.

## g-xTB solvation, measured

`--gbe` and `--cosmo` exist but are electrostatic-only and the upstream repo
calls the parameterisation "under development"; the ddCOSMO gradient is
documented as inconsistent. Measured on HPO4(2-): ddCOSMO gives −716 kJ/mol
where xtb-ALPB's *total* dGsolv is −1032. It cannot replace the dGsolv term.

Both optimisations did converge ("GEOMETRY OPTIMIZATION CONVERGED"), so the
inconsistent gradient did not prevent convergence here.

## Caveats

n = 1 species. The cancellation should be confirmed on ATP and a neutral before
being treated as general. The reference is r2SCAN-3c/CPCM, not experiment. And
because g-xTB's own solvation is the weak part, this tests g-xTB *as usable
today for solvated metabolite geometries*, not the g-xTB Hamiltonian in
isolation. Its H–Lr coverage and transition-metal claims remain untested and
are still the most plausible place it could earn a role — the ~390
metal-containing ModelSEED compounds where MLIPs fail silently.
