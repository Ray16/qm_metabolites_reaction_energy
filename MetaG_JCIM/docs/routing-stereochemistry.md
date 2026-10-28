# Source stereochemistry through routing

Dev17 retains the source stereochemistry of cysteinyl alpha carbons when building
thiol/disulfide peptide cores. It edits the matched parent residues and caps bonds
outside each retained centre's neighbours. Reordering atoms or reversing the
input stereocentre changes the core consistently. An unspecified source centre
remains unspecified; a compound name never supplies an L/D assignment. Mixed or
unsupported disulfides keep their parent structures instead of acquiring a
homodimer core. Energetic cancellation of the removed environment remains an
approximation requiring separate validation.

Generic spectator truncation now rejects a proposed fragment when a previously
specified structure becomes stereochemically undefined. The planner retains its
pre-truncation representation, including any other admitted cofactor cores. It
records the rejected fragment under
`routes.stereochemistry_truncation_rejected`. This avoids unnecessarily restoring
an entire NAD molecule when only a substrate cut failed.

The original 364-input panel contains four such recoveries: two glutathione
redox reactions, testosterone dehydrogenation and quinate dehydrogenation. The
remaining 101 inputs with unresolved source stereochemistry and six other
chemical exclusions retain their no-estimate status. Numerical minimum/state
coverage, experimental accuracy and full/core transfer are separate checks.

Inspect a panel before reserving computation:

```bash
python -m metag.tools.audit_routes reactions.json -o routes.json --require-consistent
```

The regression fixtures and source-centre, atom-order, mixed-disulfide and
fallback tests are in `tests/unit/test_routing_stereo_recovery.py`. The complete
dev17 suite reports 967 passed and one opt-in GPU skip. Clean installed CPU
checks reproduce the four admissions without importing the scoring/QM runtime.

Numerical work remains versioned separately. The dev16 full-panel run is frozen
under `analysis/publication_sprint_20261018_01/`. The dev17 follow-up uses private,
hash-verified copies of each unchanged reaction's own accepted dev16 molecular
files and performs real dev17 API scoring; the four recovered cases get fresh
calculations. Model/reference/inference identity, unchanged physical settings,
unchanged raw-energy source, numerical replay parity and cache byte identity are
checked. Cold and cache-assisted costs retain their original scope. These are
not independent experimental validation or a promise of a particular error.
