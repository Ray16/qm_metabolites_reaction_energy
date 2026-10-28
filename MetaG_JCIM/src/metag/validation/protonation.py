"""Conservative structural coverage diagnostics; no QM or experimental reaction energies.

A built neutral reference is not evidence of complete speciation. This diagnostic
identifies some known gaps, not every possible ionizable group.
"""
from rdkit import Chem
from collections import defaultdict
from metag.routing import pka_transform as pk


def audit_model_consistency(records):
    """Find shared scored states given different per-molecule protonation models.

    No energy calculation or network projection is involved. Missing ownership
    is reported separately and never counted as verified reference consistency.
    """
    uses = defaultdict(lambda: defaultdict(list))
    unavailable = []
    for rid, record in records.items():
        if record.get('status') not in ('review', 'no_known_gap_detected'):
            continue
        references = record.get('protonation_reference_models')
        if not references:
            unavailable.append(rid)
            continue
        for name, row in references.items():
            identity = (row['reference_smiles'], row['reference_charge'])
            uses[identity][row['assignment_sha256']].append({'reaction': rid, 'species': name})
    conflicts = [
        {'reference_smiles': smiles, 'reference_charge': charge,
         'assignments': [{'assignment_sha256': signature, 'uses': entries}
                         for signature, entries in sorted(groups.items())]}
        for (smiles, charge), groups in sorted(uses.items()) if len(groups) > 1
    ]
    return {'n_states_with_owned_models': len(uses), 'conflicts': conflicts,
            'unavailable_reactions': sorted(unavailable), 'passed': not conflicts,
            'complete_ownership': not unavailable,
            'scope': 'Consistency of available per-molecule protonation assignments; '
                     'not independent energy cycles, verified source content or validated speciation.'}


def inspect_species(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"status": "invalid", "issues": ["invalid_smiles"]}
    canonical = pk._canonicalize_maxanion(smiles)
    cmol = Chem.MolFromSmiles(canonical)
    issues = set()
    anions = pk._anion_sites(cmol)
    covered = {i for i, _ in anions}
    if any(a.GetFormalCharge() < 0 and a.GetIdx() not in covered for a in cmol.GetAtoms()):
        issues.add("unmodeled_anionic_site")
    if any(a.GetAtomicNum() not in {1, 6, 7, 8, 9, 15, 16, 17, 35, 53} for a in mol.GetAtoms()):
        issues.add("element_outside_protonation_rules")
    guan_c = {m[1] for m in cmol.GetSubstructMatches(pk._GUAN_C)}
    for a in cmol.GetAtoms():
        if a.GetSymbol() == "P" and any(n.GetSymbol() == "N" for n in a.GetNeighbors()):
            issues.add("phosphoramidate_pka_source_unverified")
        if a.GetSymbol() != "N":
            continue
        if a.GetIsAromatic():
            issues.add("aromatic_n_speciation_incomplete")
        if any(n.GetIdx() in guan_c for n in a.GetNeighbors()):
            issues.add("guanidine_amidine_speciation_incomplete")
        if a.GetFormalCharge() == 1 and a.GetTotalNumHs() > 0 and not a.GetIsAromatic():
            heavy = [n for n in a.GetNeighbors() if n.GetAtomicNum() != 1]
            if heavy and not all(n.GetAtomicNum() == 6 and n.GetHybridization() == Chem.HybridizationType.SP3
                                 for n in heavy):
                issues.add("protonated_n_uses_broad_class_approximation")
    if len(anions) > 1:
        issues.add("multiple_acid_sites_independent_site_approximation")
    try:
        reference, acids, bases, charge = pk._neutralize_v2(smiles)
        if reference is None:
            return {"status": "unsupported", "issues": sorted(issues | {"reference_build_failed"})}
    except (ValueError, RuntimeError) as exc:
        return {"status": "unsupported", "issues": sorted(issues), "error": str(exc)}
    if acids or bases:
        issues.add("class_pka_transfer_unvalidated" if pk.pka_model_config()["pka_model"] == "table"
                   else "site_model_transfer_unvalidated")
    if charge:
        issues.add("reference_retains_charge")
    return {"status": "review" if issues else "no_known_gap_detected", "issues": sorted(issues),
            "reference_smiles": reference, "reference_charge": charge,
            "acid_pkas": acids, "base_pkas": bases}
