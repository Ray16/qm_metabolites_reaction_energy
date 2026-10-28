# Additional supported scientific workflows

These tools use the same installed `metag` package. Their command help gives
the accepted schemas and options; dated `analysis/` producers preserve the
actual experiments rather than defining a second production interface.

## Shared representations and networks

Use the same routing mode for preflight and scoring:

```bash
METAG_ROUTING_MODE=full_molecule metag audit-routes \
  reactions.json -o route_panel.json --require-consistent
METAG_ROUTING_MODE=full_molecule metag score --require-consistent-panel
```

The scoring command runs inside a GPU allocation. Its panel gate runs before
loading UMA. A passing structural gate does not establish energy-cycle closure
or accurate aqueous populations. The `compound_reference` preset adds checks
for owned acid/base models and proton bookkeeping. Unsupported states remain
visible under either preset. [Network audits](../src/metag/tools/audit_network.py)
retain unknown and failed metadata; no calibrated network covariance is supplied.

## Independent searches and chemical references

`metag audit-sampling` compares explicitly expected independent-seed results.
It checks chemistry, conditions, producer/configuration and nonoverlapping seed
windows. Missing and failed replicas remain in the denominator. Its optional
equilibrium-ratio tolerance is a declared downstream criterion; agreement of
two searches does not prove global convergence or predictive coverage. A
[worked contract](../analysis/physical_improvement_20261010/SAMPLING_AUDIT_USAGE.md)
preserves the corresponding study.

The [chemical-reference converter](../src/metag/validation/chemical_reference.py)
handles ionic chemical reactions, proton-count populations and molal/molar
standard states. The [AMP example](../analysis/chemical_reference_20261010/README.md)
derives a conditional reference and retains parameter sensitivities without
changing the default benchmark or protonation rules.

## Recorded minima and component comparisons

```bash
metag export-minima species_cache.json -o new_export
metag compare-states comparison_plan.json -o new_comparison.json
metag refine-ensemble species_cache.json --records smd_results -o new_refinement.json
```

`export-minima` verifies ordered coordinates and component-energy closure.
`compare-states` reconstructs declared exchanges and fixed-solvent ranges from
complete recorded states. See [state comparison plans](state-ensemble-comparisons.md).

`refine-ensemble` requires a strict replacement-solvent result for every
recorded accepted conformer. It checks geometry, charge/spin, SCF convergence,
CDS accounting, functional, basis, numerical protocol and producer. Missing,
failed, duplicate or incompatible records withhold the refinement. Original
uncertainty is not transferred to the reweighted result.

These operations describe the supplied sampled basis. They do not locate new
solvent minima, supply omitted states or change the default ALPB workflow.

## Thermochemistry from a supplied Hessian

Install `.[thermal]` for NumPy/ASE without loading Torch or fairchem.
`metag.energetics.thermal.gibbs_corr_from_hessian` accepts a Cartesian Hessian in
eV/Angstrom squared and an electronic energy in eV. With `return_info=True`, it
returns mode/minimum diagnostics and the thermodynamic decomposition: ideal-gas
enthalpy correction, entropy, Gibbs correction, ZPE and internal/floored-mode
counts. Preserve geometry, mass, symmetry, spin and reference-state metadata.

The documented imaginary-mode policy and 50 cm⁻¹ frequency floor
apply. qRRHO adjusts entropy while retaining harmonic enthalpy. Explicit
`temperature_K` and `pressure_Pa` arguments affect this supplied-Hessian gas
utility; the reaction pipeline remains at its fixed target conditions. Gas
references are not aqueous reaction labels.

## Experimental reference construction

```bash
python -m pip install -e '.[reference]'
metag build-reference \
  --source artifacts/benchmark/opentecr_source \
  --modelseed /path/to/ModelSEEDDatabase \
  --reactions src/metag/data/reactions_opentecr_std.json \
  --require-reported-I --temperature-K 298.15 \
  --output new_reference/reactions.json --report new_reference/report.json
```

Mapping and condition screening precede residual scoring. Total Mg salt
concentration is not free Mg activity. A supported free-activity declaration
records `p_mg_basis: "free_activity"` and its source. The historical mapping
preset remains available only as an explicitly identified reproduction regime.
Reference construction does not by itself validate stereo/pool definitions,
source corrections, reference covariance or estimator accuracy.
