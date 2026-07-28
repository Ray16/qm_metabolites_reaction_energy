# MACE-POLAR cluster-continuum pKa gate

Run on 2026-07-28 using the archived valid xTB/ALPB clusters, rescored as:

    E_MACE-POLAR(cluster) + G_xTB,ALPB(cluster) - E_xTB,gas(cluster)

The proton reference was independently re-fitted on the three cationic control
pairs for every water count. The table therefore isolates the remaining anion
error rather than a common proton-reference offset.

| explicit waters | pKa pairs | anion pairs | anion MAE (kJ/mol) | phosphate MAE (kJ/mol) |
|---:|---:|---:|---:|---:|
| 0 | 26 | 23 | 31.4 | 25.2 |
| 1 | 26 | 23 | 23.6 | 32.4 |
| 2 | 20 | 17 | 14.3 | 30.0 |

The two-water clusters improve the aggregate anion score, but fail the
pre-specified gate for a reaction-level method: anion MAE is still above 10
kJ/mol, the phosphate subset gets worse, and six pairs are absent because no
valid cluster survived. These results rule out applying this static,
lowest-cluster correction to the metabolic reactions.

The next explicit-water approach must generate and Boltzmann-average solvent
clusters (or use explicit-solvent alchemical free energies), retain the full
cluster ensemble, and validate phosphate-rich held-out pairs before reaction
scoring.

## Charge-adaptive seeded-ensemble pilot

The follow-up implementation is `grand_canonical_clusters.py`.  A small xTB
pilot (acetic acid/acetate, H2PO4-/HPO4^2-, and the imidazolium calibration
control) confirms that the state-specific ladder is functioning: H2PO4- has
its largest sampled population at one water (56%), while HPO4^2- has its
largest population at two waters (64%).  The imidazolium pair is held at n=0
for the proton-reference fit because the placement model targets anionic donor
contacts.

That pilot still gives a 35.1 kJ/mol two-anion MAE (phosphate 39.7 kJ/mol), so
it is evidence for the hydration mechanism, not evidence that this seeded xTB
ensemble is accurate enough.  It must be expanded to the full pKa set and
compared with the fixed-n MACE-POLAR gate before any reaction-level use.
