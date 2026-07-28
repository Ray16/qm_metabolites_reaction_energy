# Periodic explicit-solvent phosphate gate

This is the replacement for the rejected finite water-cluster approach. It uses
periodic bulk water, 0.15 M salt, and alchemical free-energy sampling. The first
gate is deliberately two small phosphate increments—H2PO4- → HPO4^2- and methyl
phosphate—before any GSH, PPi, ATP, or reaction calculation.

The dedicated environment is:

```bash
/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/desd_fep/bin/python
```

It includes OpenMM, OpenMMTools, PyMBAR, OpenFF, AmberTools/GAFF2 and GPU
support. Prepare reproducible periodic systems with:

```bash
python experiments/bulk_fep/prepare_systems.py --out results/bulk_fep/systems
```

Before a production run, check each endpoint on the target GPU:

```bash
python experiments/bulk_fep/alchemical_preflight.py --system-dir results/bulk_fep/systems/H2PO4-
```

Preparation is not a result. A charged-solute hydration protocol must include:

- soft-core alchemical windows and MBAR overlap diagnostics;
- independent replicas and uncertainty estimates;
- a finite-size/Ewald charge correction (or an explicitly charge-neutral
  transformation);
- a gas-phase/reference leg before converting a hydration difference to pKa.

We will not use it on the ten reactions unless the two phosphate increments
agree with experiment to roughly 10 kJ/mol and have converged uncertainties.
