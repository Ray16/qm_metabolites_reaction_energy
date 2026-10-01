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
    # hydrate the ACTIVATED aldehyde carbon (first atom of each _ACTIVATED pattern), not whichever CHO
    # RunReactants happens to list first: in O=CCCC(=O)C=O that was the alkyl CHO, leaving the alpha-keto
    # CHO -- the one the gate fired on -- unhydrated. Lowest index = deterministic.
    c = min(match[0] for p in _ACTIVATED for match in m.GetSubstructMatches(p))
    o = next(n.GetIdx() for n in m.GetAtomWithIdx(c).GetNeighbors()
             if n.GetSymbol() == "O" and m.GetBondBetweenAtoms(c, n.GetIdx()).GetBondTypeAsDouble() == 2)
    rw = Chem.RWMol(m)
    rw.GetBondBetweenAtoms(c, o).SetBondType(Chem.BondType.SINGLE)
    rw.AddBond(c, rw.AddAtom(Chem.Atom(8)), Chem.BondType.SINGLE)
    p = rw.GetMol()
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


# ---------------------------------------------------------------------------------------------------------
# GENERAL carbonyl hydration (CARBONYL_HYDRATION_ALL). The α-EWG gate above existed because the hydration
# free energy computed with xtb-COSMO was unreliable. Validated against experimental hydration constants
# (16 aldehydes/ketones, Guthrie/Bell; analysis/sweep_20261001/khyd_set.json, independent of TECRDB):
# UMA + xtb-COSMO MAE 3.6 log K (bias -3.6, i.e. ~20 kJ under-hydration -- the gem-diol's two OH groups
# carry COSMO's missing H-bond term twice) vs UMA + xtb-ALPB MAE 1.1 log K (r = 0.96; acetaldehyde 0.45 vs
# 0.03, acetone -3.3 vs -2.9, glyceraldehyde 1.8 vs 1.3, dihydroxyacetone -0.6 vs -1.0). With ALPB every
# aldehyde and ketone is hydrated through the same self-gating mixture; weakly hydrated ones contribute ~0.
_HYDRATABLE = Chem.MolFromSmarts("[CX3;$([CH2]=O),$([CH1](=O)[#6]),$(C(=O)([#6])[#6])]=[OX1]")
# alpha-KETO ACIDS (ketone carbon bonded to a carboxyl carbon) are excluded: as pH-7 anions they are only
# weakly hydrated (pyruvate ~6%, oxaloacetate 7.8% at pH 7.4, 2-oxoglutarate <10% -> <= 0.3 kJ), while
# UMA/ALPB over-hydrates them even after the K_hyd calibration (pyruvate 29%, oxaloacetate 82%) -- the
# gem-diol's O-H...O=C(OH) contact is over-stabilised by the continuum, as for the inter-acid H-bonds.
# Leaving them unhydrated is the more accurate choice. alpha-oxo ALDEHYDES (glyoxylate, 99% hydrated,
# reproduced) stay in.
_KETO_ACID = Chem.MolFromSmarts("[CX3;!H1;!H2](=[OX1])[CX3](=O)[OX2H1,OX1-]")
MAX_HYDRATION_SITES = 2
# Calibration of the computed hydration free energy against the SAME independent K_hyd set (16 carbonyls):
# log K_exp = 0.67 log K_calc - 0.34  <=>  ΔG_hyd = 0.67 ΔG_hyd,calc + 1.94 kJ/mol. Leave-one-out MAE
# 0.61 log units (3.5 kJ) vs 1.10 (6.3 kJ) raw; UMA/ALPB exaggerates both strong (glyoxylic acid,
# hexafluoroacetone, pyruvic acid) and weak hydration. HYDRATION_CAL=0 uses the raw value.
HYDRATION_CAL = (0.67, 1.94)


def calibrated_dg_hyd(dg):
    a, b = HYDRATION_CAL
    return a * dg + b


def hydration_sites(smi):
    """[(carbon_idx, gem-diol SMILES)] for each aldehyde/ketone carbonyl (at most MAX_HYDRATION_SITES)."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return []
    out = []
    keto_acid_c = {match[0] for match in m.GetSubstructMatches(_KETO_ACID)}
    for c, o in [x for x in m.GetSubstructMatches(_HYDRATABLE) if x[0] not in keto_acid_c][:MAX_HYDRATION_SITES]:
        rw = Chem.RWMol(m)
        rw.GetBondBetweenAtoms(c, o).SetBondType(Chem.BondType.SINGLE)
        rw.AddBond(c, rw.AddAtom(Chem.Atom(8)), Chem.BondType.SINGLE)
        p = rw.GetMol()
        try:
            Chem.SanitizeMol(p)
        except Exception:
            continue
        out.append((c, Chem.MolToSmiles(p)))
    return out


def mixture_G_sites(g_carbonyl, g_diols, g_water):
    """Effective G for independent hydration sites: G_c - RT Σ_i ln(1 + exp(-ΔG_hyd,i/RT)), ΔG_hyd,i =
    g_diol_i - g_water - g_carbonyl. (Same as mixture_G for one site.)"""
    g = g_carbonyl
    for gd in g_diols:
        x = -(gd - g_water - g_carbonyl) / RT
        g -= RT * (x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x)))
    return g
