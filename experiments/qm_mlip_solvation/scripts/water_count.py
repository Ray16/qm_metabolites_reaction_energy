#!/usr/bin/env python
"""Production water-count rule for explicit-solvation clusters.

REPLACES the abandoned occupancy self-selection (former step7/step7c grand potential
+ step8 calibration). Those tried to READ a self-selected occupancy PEAK and calibrate
it. We abandoned that because the peak is:
  - a NOISY, method-dependent observable (fast/UMA-Hessian pins at the cap; even
    GFN2 --ohess bounces 4/6/4 for a -2 phosphate across conformer seeds), and
  - IRRELEVANT to the goal: the reaction ΔG is insensitive to the exact water count
    (step7c gave ΔG within 3 kJ at n=9 vs n=3 — 3x different occupancy — because the
    waters cancel across the reaction; step7b converged at WPC 2->3).

What the reaction ΔG actually needs from the water count:
  (a) ENOUGH — above the threshold where under-solvation biases ΔG, i.e. saturate the
      FIRST shell of the compact anionic site, and
  (b) CONSISTENT — the SAME species always gets the SAME n, so the explicit waters
      cancel across the reaction. A noisy self-selected peak breaks (b) and thereby
      breaks the cancellation ΔG relies on. A deterministic rule delivers both.

NOT-monotone warning: MORE water is NOT safer. The insensitivity window is bounded —
too few → biased (under-solvated); first-shell saturation → accurate; TOO MANY →
high-variance + broken cancellation. Extra waters past the first shell are floppy
appendages that (1) re-explode conformer noise (the project's core enemy), (2) wander
during relaxation so they no longer cancel across the reaction, (3) model bulk worse
than the continuum they replace, (4) add near-zero modes that degrade the thermal
Hessian. So target first-shell coordination and STOP; let `converged_enough` (which
catches BOTH under- and over-watering) be the safety net, not padding.

Rule — waters per H-bond site (coordination numbers; generous by design):
  hard anionic O (carboxylate / phosphate / sulfonate / sulfate O-) : 2
  soft anionic S-                                                   : 1
  cationic N-H donor (per N-H)                                      : 1

Sufficiency is VERIFIED per reaction by a cheap ΔG(n) vs ΔG(n+1/site) convergence
probe (see `converged_enough` docstring), NOT by reading an occupancy peak. This is
step7b's WPC-ladder logic generalized and tied directly to ΔG.
"""
from rdkit import Chem

WATERS_PER_SITE = {"hardO": 2, "softS": 1, "cationNH": 1}

_SMARTS = {
    "hardO": Chem.MolFromSmarts("[O-]"),      # carboxylate / phosphate / sulfonate / sulfate O-
    "softS": Chem.MolFromSmarts("[#16-]"),    # thiolate / other soft S-
}
_NPLUS = Chem.MolFromSmarts("[N+,n+]")


def count_sites(smiles):
    """H-bond site inventory {hardO, softS, cationNH} from SMILES (None mol -> zeros)."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return {"hardO": 0, "softS": 0, "cationNH": 0}
    hardO = len(m.GetSubstructMatches(_SMARTS["hardO"]))
    softS = len(m.GetSubstructMatches(_SMARTS["softS"]))
    nh = 0
    for (idx,) in m.GetSubstructMatches(_NPLUS):
        nh += m.GetAtomWithIdx(idx).GetTotalNumHs()
    return {"hardO": hardO, "softS": softS, "cationNH": nh}


def water_count(smiles, per_site=None):
    """Deterministic production water count n for a species. Returns (n, sites)."""
    per_site = per_site or WATERS_PER_SITE
    sites = count_sites(smiles)
    n = sum(per_site[k] * sites[k] for k in per_site)
    return n, sites


def needs_explicit(smiles):
    """Heuristic gate: does this species carry a compact high-charge-density anionic
    site that continuum over-solvates? (carboxylate/phosphate/sulfonate O- or S-).
    Production still only routes a reaction to explicit water when such a site is
    CREATED/DESTROYED (not a spectator) — see triage. Soft/delocalized-only -> False."""
    s = count_sites(smiles)
    return (s["hardO"] > 0) or (s["softS"] > 0)


def converged_enough(dG_at_n, dG_at_n_plus, tol=4.0):
    """Sufficiency check that REPLACES the occupancy peak: the count is enough when
    adding one more water per site does not move the reaction ΔG beyond `tol` kJ/mol.
    Run once per representative reaction (cheap), not per production reaction."""
    return abs(dG_at_n_plus - dG_at_n) <= tol


if __name__ == "__main__":
    # quick self-check on the calibration reps
    for name, smi in [("acetate", "CC(=O)[O-]"),
                      ("methylphosphate", "COP(=O)([O-])[O-]"),
                      ("PPi", "O=P([O-])([O-])OP(=O)([O-])O"),
                      ("methanethiolate", "C[S-]"),
                      ("methylammonium", "C[NH3+]")]:
        n, sites = water_count(smi)
        print(f"  {name:16s} n_water={n:2d}  sites={sites}  explicit={needs_explicit(smi)}")
