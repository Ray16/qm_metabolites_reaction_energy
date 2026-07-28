# Explicit-water solvation gate

This experiment evaluates cluster-continuum microsolvation on an external pKa
set before applying it to metabolic reactions. It replaces the old xTB-only
cluster score with a consistent composite:

    G_cluster = E_MACE-POLAR(cluster) + [G_xTB,ALPB(cluster) - E_xTB,gas(cluster)]

The bracketed term retains xTB's ALPB and RRHO contribution while MACE-POLAR
supplies the electronic energy for the actual solute-water cluster. The xTB gas
single point is recomputed because the archived record stores an ALPB energy.
The pKa
benchmark re-fits the proton reference using cationic controls separately for
each water count, then reports anion error.

The archived clusters contain only the lowest valid seeded structure, so this is
a **model-substitution gate**, not yet an explicit-water ensemble free energy.
Proceed to regenerated solvent-cluster ensembles only if this test materially
improves anion pKa error with adequate coverage.

Example:

```bash
python score_macepolar_pka.py \
  --pairs /path/to/pka_pairs.json \
  --clusters /path/to/microsolv_n0.json /path/to/microsolv_n1.json /path/to/microsolv_n2.json \
  --geometry-root /path/to/geometries_microsolv \
  --model models/MACE-POLAR-1-L.model
```
# Charge-adaptive cluster ensembles

`grand_canonical_clusters.py` supersedes the fixed-water-count design for the
next validation step.  It keeps several valid, distinct xTB/ALPB cluster minima
at each water count, then assembles a grand-canonical free energy.  Thus a
phosphate at -3 can have a richer first shell than its -2 conjugate acid; water
does not have to cancel pairwise.

This is a seeded-minima approximation, not a converged liquid-water free-energy
calculation.  It must pass the pKa gate before it is considered for reaction
energies.

Example:

```bash
python experiments/explicit_water/grand_canonical_clusters.py build --pairs .../pka_pairs.json --source .../pka_xtb.json --out results/explicit_water/grand_clusters.json --waters-per-anionic-site 2 --max-water 8 --seeds 12 --resume
python experiments/explicit_water/grand_canonical_clusters.py score --pairs .../pka_pairs.json --ensemble results/explicit_water/grand_clusters.json --out results/explicit_water/grand_clusters_pka.json
```

The xTB-only score checks the sampling/standard-state assembly.  Use
`score_macepolar_grand.py` for the decision gate, because it uses the same
MACE-POLAR + xTB/ALPB composite as the reaction pipeline.
