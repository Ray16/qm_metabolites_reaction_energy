"""Structural reaction-family detectors (pure classification, no correction).

Each family is a net bond/group transformation detected by SMARTS on created/destroyed groups, in ONE
canonical direction (with the reverse recognized and reported as direction -1) and with a stoichiometric
extent. These were the detectors behind the legacy anchor sub-classes; they live here so that RECOGNIZING
a family never implies CORRECTING it. Consumers:
  - metag.routing.anchor   -- legacy numerical offsets (ANCHOR_CORRECT, off since 2026-10-01; see
                              docs/anchor-audit.md for why most offsets are retired);
  - metag.uncertainty      -- structure-first sigma-class assignment;
  - diagnostics / limitation flags.
Detection is anion-pattern/bond based, never by reaction id, annotation, or observed error.
"""
from rdkit import Chem

from metag.chem import is_water

_PHOSPHORAMIDATE = Chem.MolFromSmarts("[#7]-[PX4](=O)")     # N-P(=O): phosphagen P-N bond
_PYRO = Chem.MolFromSmarts("[PX4]-O-[PX4]")                 # P-O-P: pyrophosphate / NTP anhydride
# C-O-P phosphate MONOESTER: the P carries exactly one ester O plus two terminal OH/O-. The looser
# "[#6]-[OX2]-[PX4](=O)" also matched both C-O-P links of a DIESTER (cAMP, glycerophosphoinositol), so
# phosphodiesterases (diester -> monoester, net monoester +1 per link) were mis-detected as phosphatases.
_MONOESTER = Chem.MolFromSmarts("[#6]-[OX2]-[PX4](=[OX1])(-[OX1-,OX2H1])-[OX1-,OX2H1]")
_THIOESTER = Chem.MolFromSmarts("[#6X3](=O)[SX2]")          # C(=O)-S: thioester (acyl-CoA)
_MIXEDANHYDRIDE = Chem.MolFromSmarts("[#6X3](=O)[OX2][PX4]")  # C(=O)-O-P: acyl/aminoacyl-adenylate anhydride
_ACYCLIC_AMIDE = Chem.MolFromSmarts("[CX3;!R](=[OX1])[NX3]")  # acyl/carbamoyl amide, carbonyl NOT in a ring
_CATION = Chem.MolFromSmarts("[N+]")                         # permanent/protonated cationic nitrogen
_ALPHA_AMINO_ACID = Chem.MolFromSmarts("[NX3,NX4+][CX4][CX3](=O)[OX1-,OX2H1]")  # alpha-amino acid backbone
_CARBOXYL = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")     # carboxylic acid / carboxylate


def _is_ppi(m):
    """True if the molecule is pyrophosphate (any protonation state): all heavy atoms P or O, exactly 2 P.
    Distinguishes free PPi (adenylyl-transfer product) from a nucleotide's internal P-O-P (ATP/ADP)."""
    zs = [a.GetAtomicNum() for a in m.GetAtoms()]
    return zs.count(15) == 2 and all(z in (8, 15) for z in zs)


def _mols(species):
    ms = []
    for coeff, q, smi in species.values():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        ms.append((float(coeff), smi, m))            # fractional coefficients (ModelSEED 0.5 O2) kept
    return ms


def _net_count(ms, patt):
    return sum(c * len(m.GetSubstructMatches(patt)) for c, _, m in ms)


def _produces_ppi(ms):
    """Net free pyrophosphate produced (coeff>0) minus consumed -- pins adenylyl-transfer direction."""
    return sum(c for c, _, m in ms if _is_ppi(m))


def _is_co2_like(m):
    """CO2 / carbonic acid / bicarbonate cosubstrate: <=4 heavy atoms, exactly 1 C, only C/O, >=2 O.
    Matches O=C=O, O=C(O)O, O=C([O-])O regardless of protonation."""
    zs = [a.GetAtomicNum() for a in m.GetAtoms()]
    return m.GetNumHeavyAtoms() <= 4 and zs.count(6) == 1 and zs.count(8) >= 2 and all(z in (6, 8) for z in zs)


def _net_skeleton_carboxyl(ms):
    """Net carboxyl groups created on CARBON-SKELETON molecules (>4 heavy atoms). Excludes the CO2/
    bicarbonate cosubstrate, which itself matches the carboxyl SMARTS -- counting it would cancel the
    carboxyl created on the acceptor to a spurious net 0 (the bug that silently zeroed a naive gate)."""
    return sum(c * len(m.GetSubstructMatches(_CARBOXYL)) for c, _, m in ms if m.GetNumHeavyAtoms() > 4)


def _detect(ms):
    """Sub-class of a reaction written in the class's CANONICAL direction, or None.

    Every class is defined in ONE direction -- the direction all its TECRDB anchor members are written
    in (P-N created, thioester created, P-O-P consumed, monoester / amide hydrolysed, PPi released) -- so
    the offset has one sign. The reverse reaction is handled by subclass_dir(), which flips the sign.
    (Detecting on "net change != 0" fired in BOTH directions with the same-signed offset: reversed
    creatine kinase got -56 kJ instead of +56, a 112 kJ error, and the one-directional classes left
    reversed reactions uncorrected, so ΔG(reverse) != -ΔG(forward).)"""
    ppi = _produces_ppi(ms)
    # adenylyl-transfer (acyl/aminoacyl-adenylate synthetases): free PPi is PRODUCED and a mixed
    # anhydride C(=O)-O-P is FORMED (ATP + carboxylate -> acyl-AMP + PPi). CHECKED BEFORE phosphagen:
    # an N-adenylate's P-N would otherwise be mis-caught by the phosphagen SMARTS and given the
    # creatine-kinase offset (wrong). SPLIT by whether the activated acid carries its own alpha-amino
    # group (amino acid) vs a plain aliphatic/aromatic carboxylate -- see ANCHORS docstring, item 4.
    if ppi > 0 and _net_count(ms, _MIXEDANHYDRIDE) > 0:
        is_aminoacid = any(c < 0 and m.HasSubstructMatch(_ALPHA_AMINO_ACID) for c, _, m in ms)
        return "adenylylate_aminoacid" if is_aminoacid else "adenylylate_aliphatic"
    # phosphagen: a P-N phosphoramidate is CREATED (ATP + guanidine -> ADP + phosphagen) -- but NOT an
    # adenylyl-transfer (which produces PPi, not ADP). The PPi guard stops N-adenylates being
    # mis-corrected as creatine kinase.
    if _net_count(ms, _PHOSPHORAMIDATE) > 0 and ppi <= 0:
        return "phosphagen"
    # thioester (acyl-CoA ligation): a C(=O)-S thioester is CREATED AND a phosphoanhydride (NTP) is
    # consumed -> an ATP-driven acyl-CoA ligase. The P-O-P requirement excludes the redox-acylating
    # dehydrogenase (no NTP), whose error is a different (redox) mechanism. SPLIT by whether free PPi is
    # released (acyl-adenylate intermediate) or ADP/GDP+Pi (direct acyl-phosphate) -- see ANCHORS
    # docstring, item 3.
    if _net_count(ms, _THIOESTER) > 0 and _net_count(ms, _PYRO) < 0:
        return "thioester_ppi" if ppi > 0 else "thioester_pi"
    # carboxy-phosphate (biotin/ATP carboxylase): CO2/bicarbonate consumed + ATP P-O-P consumed + a
    # carboxyl CREATED on a carbon skeleton. The mixed anhydride is an intermediate (not in the net eqn)
    # so gate on the NET transformation. Carboxyls counted only on >4-heavy mols so the consumed CO2/
    # bicarbonate cosubstrate (which itself matches the carboxyl SMARTS) does NOT cancel the created
    # carboxyl to a spurious net 0. Mutually exclusive with the classes above: adenylylate releases PPi
    # (ppi>0) not Pi; thioester needs a net thioester change (0 for propanoyl-CoA carboxylase, thioester
    # on both sides); phosphagen needs a P-N. Verified to hit EXACTLY {rxn00250, rxn51768} in TECRDB-367.
    if (any(c < 0 and _is_co2_like(m) for c, _, m in ms) and _net_count(ms, _PYRO) < 0
            and _net_skeleton_carboxyl(ms) > 0):
        return "carboxyP"
    # amide hydrolysis (acyclic acyl/carbamoyl amide + H2O -> carboxylic acid + amine): a SOLVATION offset
    # on the created COOH + amine (PHYSICS VERIFIED UMA-DFT +2.9). Only ACYCLIC amides ([CX3;!R]): cyclic
    # hydantoinases are a separate near-zero population and aromatic-ring deaminases don't match. Checked
    # AFTER the phosphate/thioester/carboxyP classes (all require a phosphate this class lacks) so those win
    # any overlap; the _PYRO guard is belt-and-suspenders. Water-consuming + net acyclic amide destroyed.
    # NOTE: the pool includes carbamate hydrolysis (rxn45677, carbamate -> bicarbonate + NH4+), i.e.
    # "carbamoyl amide" is in scope by design, not by accident.
    water_consumed = any(c < 0 and is_water(smi) for c, smi, _ in ms)
    has_pop = any(m.HasSubstructMatch(_PYRO) for _, _, m in ms)
    if water_consumed and _net_count(ms, _ACYCLIC_AMIDE) < 0 and not has_pop:
        return "amide_hydrolysis"
    # phosphatase monoester: hydrolysis (water consumed) that DESTROYS a C-O-P monoester, NO P-O-P present.
    # A monoester bearing its own adjacent permanent/protonated cation (phosphocholine, phosphoserine)
    # is a DIFFERENT, unresolved sub-case -- detected but excluded from ANCHORS (see docstring item 2):
    # its raw pipeline output is already near-correct, and the majority offset makes it worse.
    if water_consumed and not has_pop and _net_count(ms, _MONOESTER) < 0:
        cationic = any(c < 0 and m.HasSubstructMatch(_MONOESTER) and m.HasSubstructMatch(_CATION)
                       for c, _, m in ms)
        return "phosphatase_monoester_cationic" if cationic else "phosphatase_monoester"
    return None


def _extent(sc, ms):
    """Number of times the class's defining transformation occurs in the (canonical-direction) reaction:
    the anchor offset is per transformation, so a reaction written with doubled coefficients (or hydrolysing
    two monoesters) gets twice the correction -- the anchored ΔG stays EXTENSIVE."""
    if sc.startswith("adenylylate"):
        return _net_count(ms, _MIXEDANHYDRIDE)
    if sc == "phosphagen":
        return _net_count(ms, _PHOSPHORAMIDATE)
    if sc.startswith("thioester"):
        return _net_count(ms, _THIOESTER)
    if sc == "carboxyP":
        return _net_skeleton_carboxyl(ms)
    if sc == "amide_hydrolysis":
        return -_net_count(ms, _ACYCLIC_AMIDE)
    return -_net_count(ms, _MONOESTER)                   # phosphatase monoester (both sub-cases)


def subclass_dir(species):
    """(sub-class, direction) for a reaction, or (None, 0). direction = +1 if the reaction is written in
    the class's canonical direction (see _detect), -1 if it is the reverse. The forward reading wins if
    both readings match (never observed on TECRDB)."""
    sc, direction, _ = subclass_extent(species)
    return sc, direction


def subclass_extent(species):
    """(sub-class, direction, extent) -- extent = times the defining transformation occurs (see _extent)."""
    ms = _mols(species)
    if ms is None:
        return None, 0, 0.0
    sc = _detect(ms)
    if sc is not None:
        return sc, +1, float(_extent(sc, ms))
    rev = [(-c, smi, m) for c, smi, m in ms]
    sc = _detect(rev)
    if sc is not None:
        return sc, -1, float(_extent(sc, rev))
    return None, 0, 0.0


def subclass(species):
    """Structural detection of a systematic anion sub-class (either direction), or None. Anion-pattern
    based (not by error). Returns a name that may or may not be a key in ANCHORS -- a detected-but-excluded
    case (e.g. a cationic-adjacent phosphatase monoester) returns its own name so callers/diagnostics can
    see it was recognized, but anchor_correct() will not apply a correction unless the name is in ANCHORS."""
    return subclass_dir(species)[0]
