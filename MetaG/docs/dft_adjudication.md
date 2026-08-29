# DFT adjudication of UMA vs eQuilibrator vs GC on ModelSEED reactions

**The problem:** ModelSEED reactions have **no experimental ΔrG′°** (0/4545 TECRDB substrate-level matches).
So when UMA disagrees with both incumbents (GC, eQuilibrator), "which is right?" cannot be answered by the
incumbents — and it is a *methodological error* to treat them as ground truth. GC and eQ share the **same
group-additivity blind spot**: they both systematically under-estimate multi-electron **redox**, so they can
cluster near each other *and both be wrong*. The only valid arbiters are **experiment** (small-molecule
analogues) or **higher-level QM (DFT)**.

## Method (small-molecule / gas-phase electronic reference)

- **Experimental anchors:** clean small-molecule reactions of the same *type* with known gas-phase ΔG,
  scored through the same UMA path (`analysis/o2ref.py`-style, `verify_adenylylate_physics.py`).
- **DFT anchors:** `pyscf` **PBE0/def2-TZVP** single-point on the **UMA-relaxed** geometry (xtb GFN2
  `--opt tight` → UMA `batched_fire` → DFT single-point on that geometry), gas-phase, RKS (closed-shell) /
  UKS (open-shell). Same-geometry protocol as `verify_anchor_physics.py` — isolates the *electronic* ΔE.
  Tools: `analysis/verify_adenylylate_physics.py`, `analysis/diag_phenoxazinone.py`.
- **Caveats:** single-point (not DFT-optimised), gas-phase, no ZPE/thermal, no dispersion. **PBE0 is not
  benchmark-quality for extended fused conjugation** (delocalisation error, tens of kJ) — UMA−PBE0 is a
  sanity check, NOT the true error. For a definitive number use ωB97M-V or DLPNO-CCSD(T) + optimised geoms.

## Results (validated so far)

| reference reaction | UMA | reference | err | verdict |
|---|---|---|---|---|
| 2H₂ + O₂ → 2H₂O | −450.2 | −457.2 (exp) | +7.0 | UMA accurate on O₂ redox |
| H₂O₂ → H₂O + ½O₂ | −121.2 | −116.7 (exp) | −4.5 | " |
| benzene + ½O₂ → phenol | −159.0 | −162.6 (exp) | +3.6 | UMA accurate on aromatic oxidation |
| hydroquinone + ½O₂ → benzoquinone + H₂O | −104.5 | ~−102 (exp) | −2.5 | " |
| ethylene oxide + H₂O → glycol | −55.8 | −81.5 (exp) | +25.7 | sign right, ~26 kJ under (ring strain) |
| pyrophosphate + AcOH → acetyl-P + H₃PO₄ | +9.0 | +7.9 (PBE0) | +1.1 | mixed-anhydride electronic is fine → adenylylate offset is *solvation* |
| **rxn00054**: 4 aminophenol + 3 O₂ → 2 phenoxazinone + 6 H₂O | **−981** | **−893 (PBE0)** | **−88** | **UMA ≈ DFT; GC −286 / eQ −237 are ~600 kJ WRONG** |

Phenoxazinone S–T gap: UMA −138, PBE0 −131 kJ (singlet ~135 kJ below triplet) → **clean closed-shell
singlet, no diradical/multireference character** — UMA reproduces it. So rxn00054 is not a multireference
failure; UMA is the most accurate of the three, off PBE0 by only ~88 kJ on a 12-electron oxidation.

## The lesson (why this doc exists)

rxn00054 was initially mislabelled a **"UMA error, ~700 kJ off"** — because the error was measured against
GC/eQ. DFT shows the incumbents are the ones ~600 kJ off; UMA is ~88 kJ off the (imperfect) DFT reference.
This is the SECOND time the GC/eQ-as-truth trap bit (the first was the EC1 O₂ oxidoreductases). The
heuristic *"incumbents agree → UMA suspect"* is UNRELIABLE on redox for the additivity-blind-spot reason
above. **Rule: adjudicate UMA only against experiment or DFT, never against GC/eQ.**

## Open (not yet DFT-checked) — "suspect" labels now doubtful

- **rxn00057** (dibenzodioxin dimerising oxygenase) — likely same pattern as rxn00054.
- **rxn02988** (quinolinate synthase), **rxn23024** (PLP synthase) — de-novo pyridine formation, flagged OPEN.

These need the same DFT check before any are called genuine UMA failures. To date, **every UMA/incumbent
divergence checked against a real reference has favoured UMA** (or been a ModelSEED data error, e.g.
rxn00486). No confirmed UMA failure yet on this set.
