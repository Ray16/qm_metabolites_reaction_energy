# Proposed release notices — author review pending

This is a staging document, not an adopted project license. The code owner,
contributing institutions, final author list and code license need author
confirmation before public release. Development wheels remain local.

MetaG source is separate from the terms governing its dependencies and input
assets. The pinned UMA checkpoint is obtained independently from the FAIR
model repository; its FAIR Chemistry License and acceptable-use terms apply
separately. No weights or model-access credentials are shipped with MetaG.
See https://huggingface.co/facebook/UMA/blob/main/LICENSE. xTB is also installed
separately and no executable is bundled.

The upstream openTECR measurement-data repository declares CC0-1.0:
https://raw.githubusercontent.com/opentecr/opentecr-data/main/LICENSE.
The pinned local workbook/source provenance is retained in
`artifacts/benchmark/opentecr_source/README.md`. MetaG's derived reaction file
also includes structure mappings, selected conditions and transformed values;
its component origins must remain identifiable.

ModelSEED data use CC BY 4.0 except where otherwise noted; their license
retains component-source conditions, including BiGG, KEGG and MetaCyc terms:
https://github.com/ModelSEED/ModelSEEDDatabase/blob/master/LICENSE.
A MetaG code license must not relicense those inputs or erase required attribution.
Record the exact source revision/snapshot for every released derived asset.

Four JSON assets are currently packaged: the derived reaction/reference panel,
class uncertainty calibration, fitted site-pKa model, and fitted uncertainty
feature model. Their byte identities and generation/provenance fields are in
`analysis/publication_hardening_20261009/distribution_inventory_dev13.json`.
The fitted artifacts retain their original training configuration and do not
establish coverage calibration or thermochemical accuracy for dev12. Their
input-source attribution must be approved alongside the derived data.

Primary article PDFs, full benchmark snapshots and copyrighted literature
copies remain local and are excluded from the wheel. Public reproducibility
materials should carry citations, lawful source links, hashes and the derived
numerical audit, rather than redistribute those PDFs by default.
