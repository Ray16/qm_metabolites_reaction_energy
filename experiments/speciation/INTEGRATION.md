# Integration test: does QC speciation help reaction dG prediction?

Applied the carbonyl-hydration ensemble correction to the 367-reaction set using
curated experimental Khyd (isolates "does speciation correctness help" from "can
QC predict it", which is validated at ~10 kJ). 54 speciation-sensitive compounds,
29 reactions carry a hydration-correctable species.

## Result: decisive in its domain, swamped by the wall elsewhere

| subset | n | MAE before | MAE after | improved |
|---|--:|--:|--:|--:|
| low-charge \|z\|<=1 (no phosphate wall) | 5 | 16.9 | **10.8** | **5/5** |
| \|z\|>=2 (wall present) | 24 | 57.6 | 54.2 | 16/24 |
| whole 367-set | 367 | 36.7 | 36.4 | -- |

Individual low-charge rescues (err before -> after): glyoxylate rxn00562
+16.5 -> +2.4; glycolaldehyde rxn01306 +7.2 -> +0.4; glyceraldehyde rxn15750
+10.9 -> +6.5; acetaldehyde rxn00541 +8.9 -> +7.1.

## Verdict
Speciation correction is directionally correct (21/29 affected reactions improve)
and **decisive where speciation is the bottleneck** -- low-charge reactions with
a heavily-hydrated aldehyde drop from 16.9 to 10.8, all five improved, several
rescued to near experiment. But on the multiply-charged reactions the ~4-14 kJ
speciation correction is swamped by the ~40-50 kJ phosphate solvation wall, so
the aggregate barely moves (36.7 -> 36.4).

**So the module helps -- in its proper domain, not as a general dG booster.** Its
value is (a) rescuing the low-charge speciation-limited reactions, and (b) as a
structure/database-correction layer: it flags that ModelSEED stores GAP as the
free aldehyde (really ~96% hydrate), methylglyoxal/glyoxylate as free carbonyls
(essentially fully hydrated), and enol tautomers as keto (cpd02469, cpd01784) --
systematic microspecies errors that group methods and the database both carry and
that QC uniquely identifies.
