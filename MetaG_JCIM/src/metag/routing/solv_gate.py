"""Gate for the SMD solute-solvation correction (metag.energetics.smd_solv).

WHY a gate (measured, full-367): a GLOBAL xtb-COSMO->SMD swap is net-WORSE. SMD is more accurate on the
absolute solvation of a CREATED/DESTROYED compact polar/charged group (the solvation wall: hydratase,
carbamoyl, glycosyl), but on reactions where solvation ~CANCELS (isomerase, transaminase -- the polar
groups are conserved), COSMO's error already cancels and SMD only injects single-geometry scatter. So
apply the correction ONLY where a polar/charged group is net created/destroyed -> a non-cancelling
solvation change. This is a physics criterion (does the solvation environment change?), not a class label.

Two stages:
  1. STRUCTURAL pre-filter (cheap, here): is a solvation-relevant group net created/destroyed? If not,
     skip SMD entirely (keeps isomerase/transaminase/group-conserving transfers on COSMO).
  2. MAGNITUDE floor (in the pipeline, after computing the SMD correction): apply only if the correction
     exceeds the SMD single-geometry scatter floor. Threshold calibrated from the full-367 solute-only
     sweep (analysis/smd_measure.py).
"""
from rdkit import Chem

# solvation-relevant groups whose CREATION/DESTRUCTION changes the aqueous environment non-trivially and
# which xtb-COSMO mis-solvates. Conserved (net 0) => the solvation cancels and COSMO is fine.
_GROUPS = {
    "hydroxyl":     Chem.MolFromSmarts("[OX2H]"),                        # alcohol / carboxyl / phenol OH
    "carboxylate":  Chem.MolFromSmarts("[CX3](=O)[OX1-,OX2H1]"),         # -COO(-)/-COOH
    "amide":        Chem.MolFromSmarts("[CX3](=[OX1])[NX3]"),            # amide C(=O)-N
    "amine":        Chem.MolFromSmarts("[$([NX3;!$(NC=O);!$(N=*);!$([nX3])]),$([NX4+])]"),  # amine/ammonium (non-amide)
    "phosphate":    Chem.MolFromSmarts("[PX4]"),                         # any phosphoryl
    "thiol":        Chem.MolFromSmarts("[SX2H,SX1-]"),                   # thiol / thiolate
    "anionO":       Chem.MolFromSmarts("[OX1-]"),                        # bare anionic O
    "carbonyl":     Chem.MolFromSmarts("[CX3]=[OX1]"),                   # aldehyde/ketone/acid C=O
    "guanidinium":  Chem.MolFromSmarts("[NX3,NX4+]C(=[NX3,NX2+])[NX3]"), # guanidine/guanidinium
}


def _net_group_change(species):
    """net = Σ coeff·(#matches) over all species, per group (coeff +product / −reactant). A group with
    net != 0 is created/destroyed. Returns {group: net} for groups that changed (nonzero)."""
    net = {g: 0 for g in _GROUPS}
    for coeff, q, smi in species.values():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        for g, patt in _GROUPS.items():
            if patt is not None:
                net[g] += coeff * len(m.GetSubstructMatches(patt))
    return {g: n for g, n in net.items() if n != 0}


_ALKENE = Chem.MolFromSmarts("[CX3]=[CX3]")
_HYDROXYL = Chem.MolFromSmarts("[OX2H]")
_PHOS = Chem.MolFromSmarts("[#15]")


def is_hydrolyase(species):
    """Hydro-lyase (net C=C + H2O <-> C-OH): a NEUTRAL hydration/dehydration where the ONLY large
    solvation change is the water reference (no created anion, no phosphate) -- so the water-reference
    correction is clean and doesn't fight a competing solute-under-solvation error. Requires: net water
    consumed/produced != 0, net alkene change != 0, net hydroxyl change != 0 (opposite sign to alkene),
    and NO phosphorus anywhere (excludes enolase/PEP). Excludes the aconitase isomer step (no net water).
    """
    net_w = net_ene = net_oh = 0
    for coeff, q, smi in species.values():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return False
        if m.HasSubstructMatch(_PHOS):
            return False                                      # phosphate involved -> not a clean hydro-lyase
        if smi in ("O", "[OH2]") and q == 0:
            net_w += coeff
        net_ene += coeff * len(m.GetSubstructMatches(_ALKENE))
        net_oh += coeff * len(m.GetSubstructMatches(_HYDROXYL))
    return net_w != 0 and net_ene != 0 and net_oh != 0 and (net_ene * net_oh < 0)


def needs_smd(species):
    """Structural pre-filter: True iff a solvation-relevant polar/charged group is net created/destroyed,
    so the reaction has a NON-CANCELLING solvation change worth the SMD correction. Water is ignored
    (it stays on the pipeline's water_ref_G). Group-conserving reactions (isomerase, transaminase,
    symmetric transfer) return False -> keep COSMO."""
    changed = _net_group_change(species)
    return bool(changed), changed
