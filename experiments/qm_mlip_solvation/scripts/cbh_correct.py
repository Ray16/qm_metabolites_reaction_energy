"""Generic isodesmic (CBH-2 / isoatomic) correction for reaction ΔG — EXPERIMENT-FREE.

Cancels UMA's shared per-motif reference errors so the pipeline generalizes to unseen reactions,
WITHOUT fitting to the benchmark. See CBH_PLAN.md.

CBH-2 identity, per molecule M (all heavy atoms/bonds/H/charge conserved by construction):
    M + Σ_bonds bond-frag  ==  Σ_atoms atom-frag
  atom-frag(a) = heavy atom a + ALL heavy neighbours, H-capped, charges & bond orders kept.
  bond-frag(i,j) = the two atoms of each heavy-heavy bond, H-capped (removes the double count).
Define  δ(M) := Σ_atoms atom-frag − Σ_bonds bond-frag   (a signed multiset of small molecules).
Then  δ(M) − M  is a balanced reference set, and for a balanced reaction Σ c_s M_s = 0,
      Δcorr := Σ_s c_s δ(M_s)   is itself balanced.  The energy correction is
      ΔG_corrected = ΔG_UMA + Σ_frag n_frag · [G_high(frag) − G_UMA(frag)].
CBH-2 (not -1) keeps carboxylate / phosphate / guanidinium / ammonium intact as whole reference
molecules (acetate, methyl phosphate, methylguanidinium, methylammonium), so charge & resonance are
never split. Spectator scaffolds (adenine, ribose) cancel in the net → the surviving library is small.

Phase 1 here = the DECOMPOSITION + strict balance gate only (no energies yet).
"""
from collections import Counter
from rdkit import Chem


def _submol(mol, atom_ids):
    """Build the H-capped fragment on the induced subgraph of atom_ids. mol must be Kekulized
    (no aromatic bond types) so fragments off a ring are valid molecules. Formal charges kept;
    open valences filled with implicit H by sanitize."""
    aid = list(atom_ids)
    idx = {}
    em = Chem.RWMol()
    for i in aid:
        a = mol.GetAtomWithIdx(i)
        na = Chem.Atom(a.GetAtomicNum())
        na.SetFormalCharge(a.GetFormalCharge())
        na.SetNoImplicit(False)                      # allow implicit H to cap
        idx[i] = em.AddAtom(na)
    for n, i in enumerate(aid):
        for j in aid[n + 1:]:
            b = mol.GetBondBetweenAtoms(i, j)
            if b is not None:
                em.AddBond(idx[i], idx[j], b.GetBondType())
    m = em.GetMol()
    Chem.SanitizeMol(m)
    return m


def _canon(m):
    return Chem.MolToSmiles(m)


def _formula(m):
    """(element Counter incl. H, total charge) for a fragment molecule."""
    mh = Chem.AddHs(m)
    c = Counter(a.GetSymbol() for a in mh.GetAtoms())
    q = sum(a.GetFormalCharge() for a in mh.GetAtoms())
    return c, q


def cbh2_fragments(smiles):
    """Return δ(M) as {canonical_smiles: signed_count}: +1 per atom-frag, −1 per bond-frag.
    Raises ValueError if the per-molecule CBH-2 identity does not balance (a construction bug)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles}")
    kek = Chem.Mol(mol)
    Chem.Kekulize(kek, clearAromaticFlags=True)
    heavy = [a.GetIdx() for a in kek.GetAtoms() if a.GetAtomicNum() > 1]
    frags = Counter()
    for a in heavy:
        nbrs = [n.GetIdx() for n in kek.GetAtomWithIdx(a).GetNeighbors() if n.GetAtomicNum() > 1]
        frags[_canon(_submol(kek, [a] + nbrs))] += 1
    for b in kek.GetBonds():
        i, j = b.GetBeginAtom(), b.GetEndAtom()
        if i.GetAtomicNum() > 1 and j.GetAtomicNum() > 1:
            frags[_canon(_submol(kek, [i.GetIdx(), j.GetIdx()]))] -= 1
    _assert_molecule_balance(mol, frags)
    return frags


def _assert_molecule_balance(mol, frags):
    """Check the CBH-2 identity  M + Σbond-frag == Σatom-frag  conserves every element, H and charge.
    i.e.  (Σ_frag signed · composition)  ==  composition(M)."""
    target_c, target_q = _formula(mol)
    acc_c, acc_q = Counter(), 0
    for smi, n in frags.items():
        c, q = _formula(Chem.MolFromSmiles(smi))
        for el, k in c.items():
            acc_c[el] += n * k
        acc_q += n * q
    for el in set(acc_c) | set(target_c):
        if acc_c[el] != target_c[el]:
            raise ValueError(f"CBH-2 imbalance for {Chem.MolToSmiles(mol)}: element {el} "
                             f"got {acc_c[el]} want {target_c[el]}")
    if acc_q != target_q:
        raise ValueError(f"CBH-2 charge imbalance for {Chem.MolToSmiles(mol)}: {acc_q} vs {target_q}")


def reaction_reference(species):
    """species: {name:(coeff,charge,SMILES)} -> net CBH-2 reference multiset {smiles: count}
    = Σ_species coeff · δ(M). Spectator fragments cancel. Asserts the net set is balanced against
    the (already-balanced) reaction, so the returned library is a valid isodesmic correction basis."""
    net = Counter()
    for name, (coeff, q, smi) in species.items():
        for fr, n in cbh2_fragments(smi).items():
            net[fr] += coeff * n
    net = Counter({k: v for k, v in net.items() if v != 0})
    _assert_reaction_balance(species, net)
    return net


def _assert_reaction_balance(species, net):
    """Net reference reaction must conserve every element and charge (H included)."""
    acc_c, acc_q = Counter(), 0
    for smi, n in net.items():
        c, q = _formula(Chem.MolFromSmiles(smi))
        for el, k in c.items():
            acc_c[el] += n * k
        acc_q += n * q
    # reaction species side (Σ coeff·composition) — must equal net (both are δ minus M cancels M=0)
    rc, rq = Counter(), 0
    for name, (coeff, q, smi) in species.items():
        c, cq = _formula(Chem.MolFromSmiles(smi))
        for el, k in c.items():
            rc[el] += coeff * k
        rq += coeff * cq
    for el in set(acc_c) | set(rc):
        if acc_c[el] != rc[el]:
            raise ValueError(f"reaction-ref imbalance: element {el} net {acc_c[el]} vs rxn {rc[el]}")
    if acc_q != rq:
        raise ValueError(f"reaction-ref charge imbalance: net {acc_q} vs rxn {rq}")


if __name__ == "__main__":
    # ---- Phase 1 self-test: balance gate on charged/zwitterionic building blocks ----
    tests = {
        "acetate": "CC(=O)[O-]", "alanine-zwitterion": "[NH3+][C@@H](C)C(=O)[O-]",
        "ammonium": "[NH4+]", "pyruvate": "CC(=O)C(=O)[O-]",
        "glutamate": "[NH3+][C@@H](CCC(=O)[O-])C(=O)[O-]",
        "2-oxoglutarate": "O=C([O-])CCC(=O)C(=O)[O-]",
        "methylguanidinium": "CNC(N)=[NH2+]", "methyl-phosphate": "COP(=O)([O-])[O-]",
        "water": "O", "nicotinamide": "NC(=O)c1cccnc1",
    }
    for name, smi in tests.items():
        fr = cbh2_fragments(smi)
        print(f"OK  {name:20s} {len(fr):2d} frag-types")
    print("\nper-molecule CBH-2 balance gate PASSED for all building blocks")
