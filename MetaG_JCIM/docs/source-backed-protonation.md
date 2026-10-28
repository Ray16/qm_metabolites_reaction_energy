# Source-backed molecular protonation models

`score_reaction()` accepts an optional `species_protonation_models` mapping,
keyed by the input species label. A model replaces that species' automatically
owned independent-site terms with a sequential macroscopic binding polynomial.
It never adds an empirical reaction offset. Existing defaults are unchanged.

The current bridge supports full neutral, fully protonated C/H/O carboxylic-acid
families. The source reference formula and isomeric SMILES must match the actual
scored molecule, and the ladder must cover every carboxylic acid group. Sources
must explicitly state temperature, ionic strength, water, absence of magnesium
binding and the 1M activity convention. Constants reported at another standard
state require an auditable conversion before use. Source locator, table/page,
artifact hash, uncertainty convention and development exposure are recorded.
The API checks declarations; callers remain responsible for source verification.

Parameters use `kind="sequential_macroscopic_acid_pka"`,
`reference_deprotonations=0` and
`reference_scope="fully_protonated_carboxylic_acid_family"`. The portable citrate
example is in `analysis/critical_state_improvement_20261013_01/models/` with its
NIST provenance. It is a development example, not an automatic compound lookup.

Intermediate macrostates cannot be equated to an arbitrary computed tautomer.
Consequently this bridge does not admit neutral AMP. Nitrogen/phosphorus species,
nonacid carbonyls, enols and other unresolved oxygen-state branches are also
outside its current scope. Oxaloacetate needs a joint protonation/tautomer/hydrate
model. A full-molecule source cannot be transferred to a truncated core merely
because reaction elemental balance is preserved. Invalid mapping or conditions
return a structured no-estimate before species-energy work starts.

Dev16 checks the same declarations during CPU preflight, including source
conditions, molecular ownership and complete site replacement. Use
`python -m metag.tools.audit_routes INPUT.json -o REPORT.json --require-consistent`
before reserving computation, or use `metag --require-consistent-panel` for the
scoring CLI's gate. Neither preflight path imports the optional QM runtime.
The standalone `metag.tools.preflight` command records blocked reactions in its
JSON report; use `audit_routes --require-consistent` when a failing exit status
is required.

Identical scored molecular states must use the same protonation assignment
within a reaction and across a panel. A source model applied to only one copy
of an identity reaction is rejected before QM. The panel audit compares
per-molecule assignments independently of species labels, reaction direction
and stoichiometric extent. Source identity remains part of each assignment;
coincidentally equal numbers do not establish a shared reference. Unavailable
ownership is reported separately from observed assignment conflicts. These
checks do not establish independently sampled energy-cycle closure.

The source-model fingerprint enters the reported estimator configuration.
Diagnostics distinguish the remaining independent correction from the signed
macroscopic contribution, record proton-count populations, local pH and
sequential-pKa derivatives, and retain source metadata. These derivatives use
the cumulative population beyond each dissociation step and the specified
reference macrostate; macroscopic ladders are not independent sites. They
measure local parameter sensitivity, not prediction uncertainty. Reported source
uncertainties are not
propagated without covariance; historical class widths do not become validated
intervals for this variant. Finite sampled reference conformers and vertical
solvation/gas thermal assumptions remain approximations requiring validation.
