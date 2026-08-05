# Geometry-method benchmark

Does the choice of geometry optimiser change the composite free energy?

`optimizer_geometry_ab.py` optimises the same ETKDG structure with two methods
and scores both with identical downstream terms, so geometry is the only
variable. A DFT reference is added by hand (ORCA `r2SCAN-3c TightSCF Opt
CPCM(water)`); see `RESULTS.md` for the pyrophosphate case.

    XTB_BIN=<gfn2 xtb> python experiments/geometry_benchmark/optimizer_geometry_ab.py

g-xTB is obtained as a statically linked binary from `grimme-lab/g-xtb`
(checksum-verified) and invoked as `xtb --gxtb ...`. It is **not** a dependency
of the pipeline; nothing in the production path uses it.

`RESULTS.md` is the finding of record. Generated geometries land in the
git-ignored `results/geometry_benchmark/`.
