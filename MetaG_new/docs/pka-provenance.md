# pKa provenance and applicability audit

The runtime table in `src/metag/routing/pka_transform.py` is hard-coded. It mixes
compound constants, rounded functional-group approximations, and assumptions
about coupled protonation sites. It is not a collection of measured microscopic
constants for every metabolite. The October 6 fixes change bookkeeping and
configuration fingerprints, not the numerical pKa values.

| Runtime parameter | Evidence checked | Applicability limitation |
|---|---|---|
| `AMMONIA_PKA = 9.25` | NIST's evaluated ammonia value is 9.245 at 298.15 K and zero ionic strength [1]. | A rounded compound constant for NH4+/NH3. |
| `PRIMARY_AMINE_PKA = 10.6` | NIST gives methylamine 10.645 under the same conditions [1]. | A methylamine-like approximation. The existing classifier also uses this fallback for other amines; that extension is not independently validated. |
| `AAA_AMINE_PKA = 9.60` | The table assumes a common alpha-amino-acid value. NIST lists glycine ionization measurements near 9.78 at zero ionic strength [1]. | Neither a universal amino-acid constant nor the amine microconstant of fully neutral glycine. |
| `THIOL_PKA = 8.7` | The code cites GSH/cysteine values without primary references. NMR studies explicitly resolve interacting thiol/amine sites in glutathione [2]. | A class approximation; a single number cannot represent all thiols or all glutathione microstates. |
| `PHENOL_PKA = 10.0` | The code cites phenol/tyrosine but gives no original measurement citation. | A parent-phenol approximation. The SMARTS includes substituted phenols, whose acidity can be very different; applicability is not established for all matches. |
| `POLYACID_PKA` | Code attributes the list to Martell & Smith, via a secondary reference table. NIST independently supports the succinate pair 4.207/5.636 [1]. | Default **off**. Not every entry was independently verified. Removing tetrahedral stereochemistry conflates stereoisomers, especially tartaric acid. |
| `CARBOXYL_PAIR_LADDER` | Uses parent-acid constants; the succinate example is supported by NIST [1]. | Default **off**. Transferring a parent constant to substituted analogues requires separate validation. |
| `CARBOXYL_PKA_ALPHA["amine_neutralized"] = 4.4` | Code derives it from an assumed glycine tautomer equilibrium. A published microscopic model distinguishes neutral-glycine acid/base constants and their coupling [3]. | The independent-site product is an approximation. A macroscopic amino-acid pKa cannot simply be substituted as a microscopic constant. |
| `CARBOXYL_PKA_ALPHA["oxo"] = 1.8` | Code cites Lopalco et al., whose NMR study distinguishes oxo/hydrated states at 25 °C, I = 0.15 M [4]. | Transferring to other alpha-keto acids and adjusting to I = 0 are additional model assumptions. |
| `P_N_LADDER = [2.70, 4.58]` | The source comment explicitly says exact citations need confirmation. | **Not source-validated by this audit.** Do not describe the first value as verified literature data or conclude the phosphagen error cannot involve this assignment. |

The fingerprint now includes ammonia, amino-acid amine, generic amine,
imidazole, and guanidinium constants as well as acid constants. Previously an
edit to a base constant could silently preserve the uncertainty-calibration
fingerprint. Adding a name to that fingerprint does not introduce a new pKa.

The neutral-microspecies routine now assigns its existing base term to neutral
NH3 and saturated aliphatic amines as well as their explicitly protonated input
drawings. This fixes representation dependence within that routine. It does
not validate the constants or remove reaction-dependent routing elsewhere.

## Reference strategy

The [IUPAC Digitized pKa Dataset](https://github.com/IUPAC/Dissociation-Constants)
is a useful primary collection for an eventual measured-data lookup. It has
aqueous measurements, structures, experimental conditions, reliability labels,
and original-reference codes, with about 10,600 distinct molecules. It is
accessible under CC BY-NC 4.0; a distributable package must retain its terms and
attribution. No copy of that dataset is bundled by this change.

A compound hit is not sufficient for automatic replacement: retain dissociation
type (pKa, pKaH, pKb), temperature, ionic strength, solvent, and the identity of
the protonation transition. In particular, a list of macroscopic pKas does not
identify all site microconstants of a polyprotic metabolite. Record an explicit
missing-data status when these cannot be established.

Marvin/ChemAxon's pKa plugin supplies predictions and microspecies distributions,
not a universal experimental table. Its [licensing documentation](https://docs.chemaxon.com/latest/calculators_licensing.html)
distinguishes interactive Marvin calculations from batch `cxcalc` use. Making
it mandatory would add an external license dependency to this standalone package.

## Sources checked

1. Goldberg, Kishore and Lennen (2002), *Thermodynamic Quantities for the
   Ionization Reactions of Buffers*, JPCRD 31, 231–370. The NIST compilation
   specifies the hypothetical ideal solution at unit molality; it is not a
   blanket specification of MetaG's 1 M standard-state convention.
   [NIST full text](https://www.nist.gov/system/files/documents/srd/jpcrd615.pdf).
2. Mirzahosseini, Somlyay and Noszál (2015), *The comprehensive acid–base
   characterization of glutathione*, Chemical Physics Letters 622, 50–56.
   [DOI](https://doi.org/10.1016/j.cplett.2015.01.020).
3. Borkovec, Koper and Spiess (2014), *The intrinsic view of ionization equilibria of
   polyprotic molecules*. Its glycine example is at I = 0.1 M and 25 °C.
   [DOI](https://doi.org/10.1039/C4NJ00655K).
4. Lopalco et al. (2016), *Determination of pKa and Hydration Constants for a
   Series of alpha-Keto-Carboxylic Acids Using Nuclear Magnetic Resonance
   Spectrometry*, J. Pharm. Sci. 105, 664–672.
   [PubMed](https://pubmed.ncbi.nlm.nih.gov/26149194/).

These sources support the distinctions above; they do not validate the entire
current table or its transfer to arbitrary ModelSEED compounds.
