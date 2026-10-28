# Explicit reducing-sugar solution pools

Use `species_state_pools` when the source identifies an equilibrating reducing
sugar parent. Ordinary undefined stereochemistry continues to require a source
assignment. Declaring a pool never supplies an unknown nonanomeric configuration.

```json
{
  "species": {
    "D-glucose": [-1, 0, "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"],
    "D-fructose": [1, 0, "OC[C@H]1OC(O)(CO)[C@@H](O)[C@@H]1O"]
  },
  "n_Hplus": 0,
  "species_state_pools": {
    "D-glucose": {
      "kind": "reducing_sugar_equilibrium",
      "source": "Input source defines the equilibrating D-glucose parent"
    },
    "D-fructose": {
      "kind": "reducing_sugar_equilibrium",
      "source": "Input source defines the equilibrating D-fructose parent"
    }
  }
}
```

The CPU route planner chooses an explicit reference drawing of the input's ring
size as a computational gauge and retains the original source depiction. It
constructs the open carbonyl and both configurations of each accessible five-
or six-membered O-hemiacetal ring. Fixed stereocentres, formula and formal charge
are preserved. Blocking an OH with phosphate or a glycosidic linkage prevents
that OH from closing a ring. Anomeric glycosides and glycosyl phosphates do not
acquire a mutarotation pool. Multiple reducing centres, unresolved other stereo,
isotopes and capped or failed constructions are refused.

Every required state receives the same configured electronic, thermal,
standard-state and complete carbonyl-hydration treatment. If one state cannot
be evaluated, the reaction returns no estimate. No partially successful state
list is renormalized. Generic spectator truncation is disabled for a declared
pool because it has no established mapping of the full solution-state family;
independently guarded cofactor-core treatment remains explicit in the route.

For raw state energies `g_i` and their owned independent-site protonation
factors `f_i`, the pool energy is `-RT log sum_i exp(-(g_i + f_i)/RT)`.
The reaction assembler already adds the reference state's factor once, so the
pooling layer subtracts that same reference factor before returning its species
energy. Acid-only routes may retain a protonated amine; they are followed without
introducing an unowned base correction. Different reference drawings therefore
give the same final transformed pool. Flat caller pKa terms and joint
macroscopic/ring-proton/ring-chain models without a compatible owned sugar
reference are refused.

Sampling identities use the complete state/factor family. A deterministic
reference-factor offset is used only to verify that two labels share the same
sampled estimate; it does not alter the reaction energy. Repeated physical
pools combine their signed stoichiometric coefficients before sampling
quadrature. The source definition and pool policy are included in the effective
configuration, so older uncertainty calibrations cannot silently apply.

The constructed basis is the declared five/six-ring/open subspace, including
hydration under the configured aqueous model. Septanoses, dimers, other
protonation or tautomer families and unsampled conformer basins are outside this
conditional model. Construction tests and population checks do not establish
universal population accuracy or calibrated prediction intervals. The 23-state
control on five sugar families and all workflow outcomes are retained in the
[review study](../analysis/remaining_review_20261022_01/WORK_STATUS.txt).
