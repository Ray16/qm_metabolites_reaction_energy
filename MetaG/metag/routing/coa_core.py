"""CoA thioester core reduction — isodesmic truncation of the CoA pantetheine-ADP scaffold.

The NEXT isodesmic core-reduction (same family as COFACTOR_RING nicotinamide/GSH). CoA rides along
a reaction like NAD does: acyl-S-CoA on one side, CoA-SH (or another acyl-S-CoA) on the other, with
the huge pantetheine-3'-phospho-ADP scaffold IDENTICAL on both sides -> it cancels analytically in
ΔG but its floppy-tail conformer noise does NOT cancel numerically in full-molecule QM (CoA-thioester
is 55% of the >20 kJ tail; the acetaldehyde-DH/citrate-synthase/CoA-ligase family sits at +40..+56).

Fix: replace the S-CoA scaffold with a METHYL cap on the thioester/thiol S, keeping the reactive
acyl-C(=O)-S intact:  acyl-S-CoA -> acyl-S-CH3 ;  CoA-SH -> CH3-SH.  The S-methyl is a spectator that
cancels between the two sides; the acyl-S (thioester) vs H-S (thiol) difference — the real chemistry —
is preserved at small scale. Second-order-cap-cancellation, exactly like the 1-methylnicotinamide cap.
Experiment-free.

GATE (self-validating, NOT a curated list): apply the substitution to every CoA moiety, then KEEP it
only if it PRESERVES the reaction's mass+charge balance (same guard truncation uses). This fires on
the 28 genuine acyl-transfer / thioester-hydratase / CoA-transferase reactions (scaffold is a true
spectator) and NO-OPs on the 2 where the scaffold itself is transformed (dephospho-CoA kinase
phosphorylates the scaffold; glutathione-CoA adduct) — where truncating it would be wrong.

Dry-validated (balance): 28/30 TECRDB CoA reactions apply cleanly + preserve balance; 2 correctly
no-op. ΔG validation pending GPU run. Integrate into cofactor_truncate.py as a core-reduction step.
"""
from rdkit import Chem
from collections import Counter

_PANTE = Chem.MolFromSmarts("SCCNC(=O)CCNC(=O)")             # pantetheine linker = CoA signature
_THIOESTER = Chem.MolFromSmarts("[CX3](=O)[SX2][CH2]CNC(=O)")  # acyl-S-CoA thioester
_FREETHIOL = Chem.MolFromSmarts("[SX2H1,SX1H0-][CH2]CNC(=O)")  # free CoA-SH / CoA-S-


def _coa_core_species(smi):
    """acyl-S-CoA -> acyl-S-CH3 (keep acyl, methyl-cap S); CoA-SH -> CH3SH. Returns core SMILES or None."""
    m = Chem.MolFromSmiles(smi)
    if m is None or not m.HasSubstructMatch(_PANTE):
        return None
    te = m.GetSubstructMatch(_THIOESTER)
    if te:
        cC, _o, s, cpant = te[0], te[1], te[2], te[3]
        rw = Chem.RWMol(m)
        rw.RemoveBond(s, cpant)
        me = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(s, me, Chem.BondType.SINGLE)
        keep = [f for f in Chem.GetMolFrags(rw.GetMol()) if cC in f][0]
        for idx in sorted([a for a in range(rw.GetNumAtoms()) if a not in keep], reverse=True):
            rw.RemoveAtom(idx)
        mm = rw.GetMol()
        try:
            Chem.SanitizeMol(mm)
        except Exception:
            return None
        return Chem.MolToSmiles(mm)
    if m.HasSubstructMatch(_FREETHIOL):
        return "CS"
    return None


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
            net[k] += coeff * v
        q += coeff * aq
    return {k: v for k, v in net.items() if v}, q


def coa_core(species):
    """species: {name:(coeff,charge,SMILES)} -> new dict with CoA moieties replaced by their methyl-
    capped core, IFF the substitution preserves reaction mass+charge balance (else the ORIGINAL dict,
    identity, unchanged). The core's stored charge is the ACTUAL formal charge of the capped SMILES:
    truncation removes only the phosphate scaffold's charge, but a dicarboxylic/oxo-acyl core (succinyl,
    methylmalonyl, malyl, oxalyl) KEEPS its own carboxylate (-1) -- storing 0 there would feed UMA the
    wrong charge state. Downstream pH-0 may still neutralize that carboxylate; here we only guarantee
    the intermediate species dict is charge-self-consistent with its SMILES."""
    has_coa = any(_coa_core_species(s) for _n, (_c, _q, s) in species.items())
    if not has_coa:
        return species
    new = {}
    for n, (coeff, q, smi) in species.items():
        core = _coa_core_species(smi)
        if core:
            core_q = Chem.GetFormalCharge(Chem.MolFromSmiles(core))
            new[n] = (coeff, core_q, core)
        else:
            new[n] = (coeff, q, smi)
    b0 = _reaction_balance(species)
    b1 = _reaction_balance(new)
    if b0 != b1:                                             # scaffold not a spectator -> don't truncate
        return species
    return new
