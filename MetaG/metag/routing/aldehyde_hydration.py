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
# free energy computed with xtb-COSMO was unreliable. Validation against CITED experimental hydration constants
# (16 carbonyls, analysis/sweep_20261001/khyd_verified.json: recommended 298 K values of the acp-2021-58 review
# supplement, Tables S3/S4; script khyd_validation.py): log K MAE xtb-COSMO 3.02 (bias -2.8: the gem-diol's two
# OH groups carry COSMO's missing H-bond term twice) vs xtb-ALPB 1.70 raw (bias +1.6), 0.64 after calibration.
# With ALPB every aldehyde and ketone (except alpha-keto acids, below) is hydrated through the same self-gating
# mixture; weakly hydrated ones contribute ~0.
_HYDRATABLE = Chem.MolFromSmarts("[CX3;$([CH2]=O),$([CH1](=O)[#6]),$(C(=O)([#6])[#6])]=[OX1]")
# alpha-KETO ACIDS (ketone carbon bonded to a carboxyl carbon) are excluded: as pH-6/7 anions they are only
# weakly hydrated (K_hyd pyruvate 0.08, oxaloacetate dianion 0.06, 2-oxoglutarate dianion 0.12; acp-2021-58
# Table S4 -> <= 0.3 kJ), while UMA/ALPB over-hydrates them even after calibration (raw neutral-acid log K error
# +2.7 on 5 cited alpha-keto acids; pH-7 pyruvate 29%, oxaloacetate 82% predicted) -- the
# gem-diol's O-H...O=C(OH) contact is over-stabilised by the continuum, as for the inter-acid H-bonds.
# Leaving them unhydrated is the more accurate choice. alpha-oxo ALDEHYDES (glyoxylate, 99% hydrated,
# reproduced) stay in.
_KETO_ACID = Chem.MolFromSmarts("[CX3;!H1;!H2](=[OX1])[CX3](=O)[OX2H1,OX1-]")
MAX_HYDRATION_SITES = 3          # exact enumeration of all 2^n hydration states up to n = 3 sites
# Calibration of the computed hydration free energy against CITED experimental K_hyd (analysis/sweep_20261001/
# khyd_verified.json: recommended 298 K values from the acp-2021-58 review supplement, Tables S3/S4, with their
# primary references; fit and LOO in khyd_validation.py / khyd_validation.json). Fitted on the APPLICATION
# domain (11 aldehydes, ketones and glyoxylic acid; alpha-keto acids are not hydrated by the pipeline):
# log K_exp = 0.639 log K_calc - 0.411  <=>  ΔG_hyd = 0.639 ΔG_hyd,calc + 2.35 kJ/mol per hydration event.
# LOO MAE 0.46 log units (2.6 kJ) vs 1.25 raw (raw ALPB over-hydrates, bias +1.15). HYDRATION_CAL=0: raw.
HYDRATION_CAL = (0.639, 2.35)


def calibrated_dg_hyd(dg):
    a, b = HYDRATION_CAL
    return a * dg + b


def _site_carbons(m):
    """Hydratable carbonyl carbons (alpha-keto acids excluded) in CANONICAL-rank order, so the selected sites
    do not depend on the input atom ordering."""
    keto_acid_c = {match[0] for match in m.GetSubstructMatches(_KETO_ACID)}
    sites = {c: o for c, o in m.GetSubstructMatches(_HYDRATABLE) if c not in keto_acid_c}
    rank = list(Chem.CanonicalRankAtoms(m, breakTies=True))
    return sorted(sites.items(), key=lambda co: rank[co[0]])


def _hydrate(m, pairs):
    rw = Chem.RWMol(m)
    for c, o in pairs:
        rw.GetBondBetweenAtoms(c, o).SetBondType(Chem.BondType.SINGLE)
        rw.AddBond(c, rw.AddAtom(Chem.Atom(8)), Chem.BondType.SINGLE)
    p = rw.GetMol()
    try:
        Chem.SanitizeMol(p)
    except Exception:
        return None
    return Chem.MolToSmiles(p)


def hydration_states(smi):
    """Every hydrated microspecies of `smi`: [(n_hydrated_sites, SMILES)] for all non-empty subsets of the
    hydratable sites (complete enumeration of at most MAX_HYDRATION_SITES sites, chosen in canonical order --
    a molecule with more sites is truncated to the first MAX_HYDRATION_SITES and reported by the caller)."""
    from itertools import combinations
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return []
    sites = _site_carbons(m)[:MAX_HYDRATION_SITES]
    out = []
    for k in range(1, len(sites) + 1):
        for subset in combinations(sites, k):
            h = _hydrate(m, subset)
            if h is not None:
                out.append((k, h))
    return out


def n_hydration_sites(smi):
    m = Chem.MolFromSmiles(smi)
    return 0 if m is None else len(_site_carbons(m))


def hydration_sites(smi):
    """[(carbon_idx, mono-hydrate SMILES)] for each hydratable site (canonical order)."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return []
    return [(c, _hydrate(m, [(c, o)])) for c, o in _site_carbons(m)[:MAX_HYDRATION_SITES]
            if _hydrate(m, [(c, o)]) is not None]


def mixture_G_states(g_carbonyl, states, g_water):
    """Microspecies fold over the COMPUTED states only (exact over the successfully computed hydration states of at
    most MAX_HYDRATION_SITES canonically selected sites; omitted / failed states are reported by the caller):
    G_eff = -RT ln[ exp(-G_c/RT) + Σ_S exp(-(G_S - n_S·G_water)/RT) ],  states = [(n_S, G_S)].
    No state enters whose energy was not computed (the earlier independent-site product implicitly
    included doubly hydrated states)."""
    terms = [-g_carbonyl / RT] + [-(g - n * g_water) / RT for n, g in states]
    lo = max(terms)
    return -RT * (lo + math.log(sum(math.exp(t - lo) for t in terms)))
