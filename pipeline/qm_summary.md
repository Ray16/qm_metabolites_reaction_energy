# QM vs fine-tuned dGPredictor vs TECRDB — top-10 disagreements

- QM (UMA composite) MAE vs experiment: **62.2 kJ/mol**
- dGPredictor (fine-tuned) MAE vs experiment: **61.2 kJ/mol**
- QM closer to experiment than dGPredictor in **3/10** reactions
- QM within the combined uncertainty band in **0/10** reactions

| # | rxn | class | exp | dGP | QM | QM−exp | dGP−exp | QM closer |
|--:|---|---|--:|--:|--:|--:|--:|:--:|
| 1 | rxn00086 | GSH/NAD(P) redox | 11.9 | 101.6 | -78.6 | -90.5 | +89.8 | · |
| 2 | rxn32133 | GSH/NAD(P) redox | -11.9 | -101.6 | 78.6 | +90.5 | -89.8 | · |
| 3 | rxn00070 | GSH/NAD(P) redox | 18.0 | 104.9 | -83.2 | -101.2 | +86.9 | · |
| 4 | rxn34788 | GSH/NAD(P) redox | -18.0 | -104.9 | 83.2 | +101.2 | -86.9 | · |
| 5 | rxn00605 | glycosyltransfer | -9.5 | -56.8 | 7.1 | +16.6 | -47.3 | ✓ |
| 6 | rxn01713 | glycosyltransfer | 3.9 | -39.8 | 39.8 | +35.8 | -43.8 | ✓ |
| 7 | rxn01834 | glyoxalase (thioester) | 23.5 | -19.8 | 87.8 | +64.3 | -43.3 | · |
| 8 | rxn00579 | glycosyltransfer | -4.2 | -46.9 | 67.0 | +71.2 | -42.8 | · |
| 9 | rxn01675 | nucleotidyltransfer (PPi) | 1.0 | 42.9 | 44.9 | +43.9 | +41.9 | · |
| 10 | rxn01005 | nucleotidyltransfer (PPi) | 2.7 | 42.7 | 9.9 | +7.2 | +40.0 | ✓ |

## By chemistry class

- **GSH/NAD(P) redox** (n=4): QM MAE 95.8, dGPredictor MAE 88.3 kJ/mol
- **glycosyltransfer** (n=3): QM MAE 41.2, dGPredictor MAE 44.6 kJ/mol
- **glyoxalase (thioester)** (n=1): QM MAE 64.3, dGPredictor MAE 43.3 kJ/mol
- **nucleotidyltransfer (PPi)** (n=2): QM MAE 25.6, dGPredictor MAE 40.9 kJ/mol

## Caveats

- 7/10 reactions rest on a single TECRDB measurement.
- Ranks 9,10 (triphosphate → diphosphate + PPi) redistribute charge, so residual xtb-ALPB solvation error does not fully cancel; a COSMO-RS solvation upgrade is the next lever there.
- UMA composite is so far validated against experiment only on 5 small, low-charge reactions; treat these as independent estimates with ~10–20 kJ/mol uncertainty, not gold-standard values.
