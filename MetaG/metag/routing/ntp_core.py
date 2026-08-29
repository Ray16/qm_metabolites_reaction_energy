"""NTP (nucleoside-polyphosphate) core reduction — isodesmic truncation of the adenosine/nucleoside
scaffold on ATP/ADP/AMP-type species.

Same family as COFACTOR_RING (nicotinamide ring) and coa_core (pantetheine cap). In a phosphoryl-
transfer reaction the nucleoside-ribose (adenine+ribose) rides ALONG unchanged: ATP -> ADP keeps the
whole adenosine-5'-diphosphate identical, only the gamma-phosphate moves. That scaffold cancels
analytically in ΔG, but it is huge and floppy -> its conformer noise does NOT cancel in full-molecule
QM (the phosphagen kinases run with ATP/ADP conformer-CAPPED, sigma~0, undersampled -> +44..+77 error),
and the generic MCS AUTO_TRUNCATE REFUSES these (phosphoryl -> guanidinium N is not a clean ester cut).

Fix: replace the nucleoside-ribose with a METHYL cap on the 5'-oxygen, keeping the reactive
polyphosphate intact:  (adenosine)-5'-O-P(=O)(O)O-...  ->  CH3-O-P(=O)(O)O-...  (methyl polyphosphate).
The O-methyl is a spectator that cancels between both sides; the phosphoryl-transfer chemistry
(triphosphate vs diphosphate, and the acceptor P-N/P-O bond) is preserved at small scale. Isodesmic,
experiment-free -- the nucleotide analogue of the 1-methylnicotinamide cap.

GATE (self-validating, NOT a curated list): cap every nucleoside-5'-phosphate moiety, then KEEP the
substitution only if it PRESERVES the reaction mass+charge balance (same guard coa_core/truncate use).
Fires when the nucleoside is a true spectator; NO-OPs when the nucleoside/ribose itself is transformed
(e.g. NAD biosynthesis, nucleoside kinases that phosphorylate the sugar, adenylate-forming ligases).
"""
from rdkit import Chem
from collections import Counter

# ribose 5'-carbon: a CH2 bonded to (an ester O that bonds P) AND to a ring carbon (ribose C4').
# match atoms = (C5', O5', P, C4'). We cut C5'-C4', discard the ribose+base, cap C5' with H -> CH3.
_RIBOSE5 = Chem.MolFromSmarts("[CH2X4]([OX2][PX4])[CX4;R]")
# a nucleobase must be present so we only cap genuine nucleosides (an aromatic n-heterocycle attached
# to the ribose), never a bare sugar-phosphate (those are the phosphatase/kinase-sugar substrates).
_NUCLEOBASE = Chem.MolFromSmarts("[$([nX3,nX2]);R]")


def _ntp_core_species(smi):
    """(nucleoside)-5'-O-P... -> CH3-O-P...  (methyl polyphosphate). Returns core SMILES or None if
    there is no nucleoside-5'-phosphate (no ribose-5'-CH2-O-P bearing an attached nucleobase)."""
    m = Chem.MolFromSmiles(smi)
    if m is None or not m.HasSubstructMatch(_NUCLEOBASE):
        return None
    matches = m.GetSubstructMatches(_RIBOSE5)
    if not matches:
        return None
    rw = Chem.RWMol(m)
    # collect the (C5', C4') bonds to cut; process on the original indices then prune fragments
    cuts = []
    keep_seeds = []
    for c5, o5, p, c4 in matches:
        cuts.append((c5, c4)); keep_seeds.append(c5)
    for c5, c4 in cuts:
        if rw.GetBondBetweenAtoms(c5, c4) is not None:
            rw.RemoveBond(c5, c4)
    frag_lists = Chem.GetMolFrags(rw.GetMol())
    # keep every fragment that contains a C5' seed (the phosphate side); drop the ribose+base fragments
    keep = set()
    for f in frag_lists:
        if any(s in f for s in keep_seeds):
            keep.update(f)
    if not keep:
        return None
    for idx in sorted([a for a in range(rw.GetNumAtoms()) if a not in keep], reverse=True):
        rw.RemoveAtom(idx)
    mm = rw.GetMol()
    try:
        Chem.SanitizeMol(mm)                                 # H auto-fills C5' -> CH3-O-P...
    except Exception:
        return None
    return Chem.MolToSmiles(mm)


def _atoms(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None
    c = Counter()
    for a in m.GetAtoms():
        c[a.GetSymbol()] += 1
        c["H"] += a.GetTotalNumHs()
    return c, Chem.GetFormalCharge(m)


def _reaction_balance(species):
    net = Counter(); q = 0
    for _n, (coeff, _q, smi) in species.items():
        ac, aq = _atoms(smi)
        if ac is None:
            return None, None
        for k, v in ac.items():
            net[k] += int(coeff) * v
        q += int(coeff) * aq
    return {k: v for k, v in net.items() if v}, q


def ntp_core(species):
    """species: {name:(coeff,charge,SMILES)} -> new dict with nucleoside-5'-phosphate moieties replaced
    by their methyl-capped polyphosphate core, IFF the substitution preserves reaction mass+charge
    balance (else the ORIGINAL dict, identity, unchanged). Each core's stored charge is the ACTUAL
    formal charge of the capped SMILES (the polyphosphate keeps its anionic charge; only the neutral
    nucleoside-ribose is removed)."""
    has_ntp = any(_ntp_core_species(s) for _n, (_c, _q, s) in species.items())
    if not has_ntp:
        return species
    new = {}
    for n, (coeff, q, smi) in species.items():
        core = _ntp_core_species(smi)
        if core:
            core_q = Chem.GetFormalCharge(Chem.MolFromSmiles(core))
            new[n] = (coeff, core_q, core)
        else:
            new[n] = (coeff, q, smi)
    if _reaction_balance(species) != _reaction_balance(new):    # nucleoside not a spectator -> don't cap
        return species
    return new
