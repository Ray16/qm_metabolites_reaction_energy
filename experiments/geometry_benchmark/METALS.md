# Transition metals: can any of these methods do them?

Run 2026-08-05. Short answer: no, and the pipeline was failing at it silently.

## Scale of the subset

302 transition-metal compounds appear in non-obsolete ModelSEED reactions
(Fe 133, Co 110, Mo 18, Cu 14, Ni 8, Fe+Ni 7, W 5, Mn 3, Cr 3). But:

- **136 of 302 (45%) contain a `*` placeholder** — no defined structure, so no
  method of any kind applies.
- **131 are large** (>60 heavy atoms): haem, cobalamin, Fe–S clusters.
- Only ~54 are small, and most of those are **bare ions** (`[Fe+2]`, `[Cu+2]`,
  `[Co+3]`, `[Mo+2]`, `[W]`, `[Cr+6]`).

## The silent failure

`structures._validate` refused the odd-electron ions cleanly — Fe(III) 23 e⁻,
Cu(II) 27, Co(II) 25, Mn(II) 23, Cr(III) 21. But it **accepted** the
even-electron ones and computed them as closed-shell singlets:

| species | electrons | parity | true ground state | old verdict |
|---|---:|---|---|---|
| Fe(II) | 24 | even | high-spin d6, **quintet** | accepted as singlet |
| Fe(0) | 26 | even | **5-D quintet** | accepted as singlet |
| Mo(II) | 40 | even | **quintet** | accepted as singlet |
| W(0) | 74 | even | **5-D quintet** | accepted as singlet |
| Cr(VI) | 18 | even | d0 singlet ✓ | accepted (correct) |

Cost of that assumption, singlet vs true spin (g-xTB): **Fe(II) 470, Fe(0) 470,
Mo(II) 245, W(0) 201 kJ/mol**. Three to eight times the worst error anywhere
else in this project, and produced without a warning.

Fixed: `_validate_spin_state` now refuses any d-block metal unless an explicit
`SPIN_MULTIPLICITY_OVERRIDES` entry exists.

## Method comparison on the bare ions

| | Fe(II) | Fe(0) | Mo(II) | W(0) |
|---|---|---|---|---|
| GFN2-xTB | **fails** (no SCF) | **fails** | runs, quintet +542 (wrong sign) | **fails** |
| g-xTB | runs, −470 | runs, −470 | runs, −245 | runs, −201 |
| MACE-POLAR | −184 | −65 | −417 | **+15 (wrong sign)** |

- **GFN2 cannot run Fe or W at all** — a loud failure, which is at least safe.
  Its one Mo answer violates Hund's rule.
- **g-xTB runs all four with the correct ordering.** This is the one place in
  the whole evaluation where g-xTB clearly wins, and it vindicates the H–Lr
  coverage claim.
- **MACE-POLAR does respond to spin** (so it is not structurally blind), but the
  pipeline hardcodes `spin=1` so it never uses it — and where checkable it
  disagrees with g-xTB by 171–405 kJ/mol and gets W(0) **backwards**.

## Conclusion

Computable is not the same as accurate. Even with the right multiplicity, a bare
aqueous transition-metal ion's thermodynamics is dominated by explicit
inner-shell hydration and ligand-field splitting, neither of which a continuum
composite represents. These compounds should be **flagged out of scope**, not
patched. The guard now makes that explicit rather than returning a confident
wrong number.

If transition metals ever do need to be covered, g-xTB is the only tested
electronic method that runs on them, and it would need explicit spin states,
explicit first-shell ligands, and its own validation set.
