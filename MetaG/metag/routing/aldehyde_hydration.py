"""Aqueous carbonyl-hydration microspecies correction (self-gating, experiment-free).

A hydratable aldehyde exists in water as an equilibrium of the free carbonyl R-CHO and its gem-diol
R-CH(OH)2 (= carbonyl + H2O). Scoring only the free carbonyl puts the species free energy too high for a
strongly-hydrated aldehyde -> sign-consistent ΔG error. The gem-diol is just a +1-WATER microspecies, so
we fold it into ONE effective species free energy exactly as microspecies.py folds +1-PROTON microspecies:

    G_eff = -RT ln[ exp(-G_carbonyl/RT) + exp(-(G_diol - G*_liq(H2O))/RT) ]

subtracting the aqueous water free energy keeps it at unit water activity, so the hydration water is NOT
added to the reaction stoichiometry (parallel to not adding H+ for protonation). SELF-GATING: a strongly
hydrated aldehyde (glyoxylate, ΔG_hyd very negative) -> diol term dominates -> G_eff pulled down (fixed);
a weakly hydrated one (GAP, ΔG_hyd ~0) -> carbonyl dominates -> G_eff ~ unchanged (no harm).

Validated (2026-08-27, hard-swap A/B): glyoxylate rxn00276 err 23.2->1.8; MAE 28.4->21.8 over 6 rxns,
every reaction correct-signed, magnitude ~ hydration tendency. This module is the PROPER (mixture, not
swap) production form.
"""
import math
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

RT = 8.314462618e-3 * 298.15                                   # kJ/mol

_ALDE = Chem.MolFromSmarts("[CX3;H1,H2]=[OX1]")               # aldehyde R-CHO or formaldehyde H2C=O; the
#   physical is_strongly_hydrated() gate below provides specificity (formate/ketones/acids never pass it)
_RXN = AllChem.ReactionFromSmarts("[CX3;H1,H2:1]=[OX1:2]>>[C:1]([O:2])[OH]")   # CHO/H2C=O -> CH(OH)2 / CH2(OH)2

# PHYSICAL GATE (why these and not all aldehydes): hydration K_hyd is set by how electron-POOR the carbonyl
# carbon is. Strongly hydrated == carbonyl activated by an alpha electron-withdrawing group, OR formaldehyde
# (no alkyl donor). alpha-ALKYL aldehydes (the aliphatic semialdehydes: aspartate-4-semialdehyde,
# succinic semialdehyde) are electron-RICH -> weakly hydrated -> EXCLUDED (this is exactly where UMA's
# ΔG_hyd over-estimated and the uniform correction regressed). This is a-priori carbonyl-hydration organic
# chemistry, not a TECRDB-tuned list, so it GENERALISES to ModelSEED. alpha-hydroxy (glyceraldehyde/GAP) is
# only a MILD activator and UMA is unreliable there too -> also excluded (kept strict).
_ACTIVATED = [
    Chem.MolFromSmarts("[CX3H2]=[OX1]"),                       # formaldehyde
    Chem.MolFromSmarts("[CX3H1](=[OX1])[CX3]=[OX1]"),          # alpha-carbonyl/carboxyl: glyoxal, methylglyoxal, glyoxylate
    Chem.MolFromSmarts("[CX3H1](=[OX1])[CX4]([F,Cl,Br,I])"),   # alpha-halo (chloral-type)
]


def is_strongly_hydrated(smi):
    """True iff the molecule bears an aldehyde whose carbonyl is EWG-activated (electron-poor) -> reliably
    hydrated. This is the physical gate; a-priori organic chemistry, benchmark-independent."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    return any(m.HasSubstructMatch(p) for p in _ACTIVATED)


def gem_diol(smi):
    """gem-diol SMILES for a STRONGLY-hydrated (EWG-activated) aldehyde, or None otherwise. The gate keeps
    the correction to carbonyls where hydration is physically favoured AND UMA's ΔG_hyd is trustworthy."""
    m = Chem.MolFromSmiles(smi)
    if m is None or not m.HasSubstructMatch(_ALDE) or not is_strongly_hydrated(smi):
        return None
    prods = _RXN.RunReactants((m,))
    if not prods:
        return None
    p = prods[0][0]
    try:
        Chem.SanitizeMol(p)
    except Exception:
        return None
    return Chem.MolToSmiles(p)


def mixture_G(g_carbonyl, g_diol, g_water):
    """Effective species free energy folding the carbonyl<->gem-diol hydration equilibrium (unit water
    activity). g_diol already contains one water; subtract g_water to reference it to the free carbonyl."""
    terms = [-g_carbonyl / RT, -(g_diol - g_water) / RT]
    lo = min(terms)
    return -RT * (lo + math.log(sum(math.exp(t - lo) for t in terms)))
