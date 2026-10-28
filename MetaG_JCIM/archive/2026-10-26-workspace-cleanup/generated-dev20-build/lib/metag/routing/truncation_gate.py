"""Physics-grounded routing gate: prefer the FULL molecule over truncation.

Full-molecule scoring is the safe, correct-chemistry choice EXACTLY when none of the truncation-motivating
error sources are present:
  (1) no created/destroyed charged species   -> no anion-solvation wall exposed by running full
  (2) reaction is NOT an isomerization        -> conserved scaffold does not reorganize (no strain)
  (3) reaction center is ring-embedded        -> where truncation mis-cuts (ring-opening); full keeps chemistry
  (4) NO floppy large-fragment linker         -> no acyclic rotatable bond splitting a species into two
                                                 >=5-heavy-atom fragments; such a tether fails to cancel
                                                 conformationally when sampled independently (the disaccharide/
                                                 bisphosphate/SAH conformer-non-cancellation blow-ups).

When all four hold, run the full molecule (skip AUTO_TRUNCATE). Otherwise fall through to the normal
truncation logic. Validated: +1.05 kJ MAE vs current-pipeline baseline on the ring-fix candidate set,
with every large regression (disaccharides, bisphosphate, SAH, G6P) correctly kept at baseline.
"""
from rdkit import Chem
from metag.routing import pka_transform as pfa
from metag.routing import truncate as T
from metag.routing.cofactor_cores import cofactor_ring

_MIN_FRAG = 5           # heavy atoms; two fragments this size on a rotatable tether -> non-cancelling motion
_MIN_SUB = 12          # ignore small cofactor cores / water when scanning for the linker / ring center


def _n_ion_change(species):
    cr = cp = 0
    for _, (c, q, s) in species.items():
        if q != 0:
            cr += (-c if c < 0 else 0)
            cp += (c if c > 0 else 0)
    return abs(cp - cr)


def _rc_in_ring(species):
    sp = cofactor_ring(species)
    subs = [(c, q, s) for _, (c, q, s) in sp.items()]
    R = [s for c, q, s in subs if c < 0]
    P = [s for c, q, s in subs if c > 0]
    for rs in R:
        for ps in P:
            a = Chem.MolFromSmiles(rs); b = Chem.MolFromSmiles(ps)
            if a is None or b is None or a.GetNumHeavyAtoms() < 8:
                continue
            amap, _ = T.mcs_atom_map(a, b)
            if amap and any(a.GetAtomWithIdx(i).IsInRing() for i in T.reaction_center(a, amap, b)):
                return True
    return False


def _floppy_linker(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    for bond in m.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        try:
            em = Chem.FragmentOnBonds(m, [bond.GetIdx()], addDummies=False)
            frags = Chem.GetMolFrags(em, asMols=True, sanitizeFrags=False)
        except Exception:
            continue
        if len(frags) == 2 and min(f.GetNumHeavyAtoms() for f in frags) >= _MIN_FRAG:
            return True
    return False


def _any_floppy_linker(species):
    sp = cofactor_ring(species)
    for _, (c, q, s) in sp.items():
        m = Chem.MolFromSmiles(s)
        if m is not None and m.GetNumHeavyAtoms() > _MIN_SUB and _floppy_linker(s):
            return True
    return False


def prefer_full(species, anchor_correct=False):
    """True -> route to the full molecule (skip truncation). Physics gate, no benchmark-fitted thresholds.

    `anchor_correct`: only when the legacy anchor offsets are APPLIED does a recognized reaction family
    need to keep the route its offset was calibrated on (baseline truncation) -- otherwise the offset
    would be subtracted from a differently-scored ΔG. With anchors off (production since 2026-10-01) the
    veto has no purpose and is not consulted, so family recognition cannot change a point-estimate route.
    On the frozen panels the veto changed 0/364 TECRDB and 0/300 ModelSEED routes."""
    if anchor_correct:
        from metag.routing.reaction_families import subclass
        if subclass(species) is not None:
            return False
    return (_n_ion_change(species) == 0
            and not pfa.is_isomerization(species)
            and _rc_in_ring(species)
            and not _any_floppy_linker(species))
