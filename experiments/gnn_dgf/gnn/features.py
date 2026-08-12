"""Compound featurization (needs rdkit -> base env, used by prepare_* scripts).

Atom/bond graph features + QC node feature (xtb Mulliken charge) + graph-level
QC/descriptor block.  CompoundGraphs packs all compounds into one disconnected
graph.  Feature levels: none / solv / full / rich (rich adds CPCM-X solvation).
"""
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

ATOM_VOCAB = ["C", "N", "O", "P", "S", "H", "F", "Cl", "Br", "I", "Co", "*"]
HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
       Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
       Chem.HybridizationType.SP3D2]
BOND_VOCAB = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
              Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]


def _onehot(x, vocab):
    v = [0.0] * (len(vocab) + 1)
    if x in vocab:
        v[vocab.index(x)] = 1.0
    else:
        v[-1] = 1.0                 # unknown / out-of-vocab (e.g. metals)
    return v


def atom_features(atom):
    f = _onehot(atom.GetSymbol(), ATOM_VOCAB)
    f += _onehot(atom.GetHybridization(), HYB)
    f += [atom.GetDegree() / 4.0, atom.GetTotalNumHs() / 4.0,
          float(atom.GetFormalCharge()), float(atom.GetIsAromatic()),
          float(atom.IsInRing()), atom.GetTotalValence() / 6.0]
    return f


def bond_features(bond):
    return _onehot(bond.GetBondType(), BOND_VOCAB) + [
        float(bond.GetIsConjugated()), float(bond.IsInRing())]


def rdkit_descriptors(mol):
    """Physicochemical descriptors tied to solvation/thermo (raw; scaled by LayerNorm)."""
    return [Descriptors.MolWt(mol) / 100.0, rdMolDescriptors.CalcTPSA(mol) / 50.0,
            rdMolDescriptors.CalcNumHBD(mol) / 3.0, rdMolDescriptors.CalcNumHBA(mol) / 5.0,
            rdMolDescriptors.CalcNumRotatableBonds(mol) / 5.0,
            rdMolDescriptors.CalcNumAromaticRings(mol) / 2.0,
            rdMolDescriptors.CalcFractionCSP3(mol), Descriptors.MolLogP(mol) / 3.0,
            rdMolDescriptors.CalcLabuteASA(mol) / 100.0]


ATOM_DIM = len(atom_features(Chem.MolFromSmiles("CC").GetAtomWithIdx(0)))
BOND_DIM = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))

import torch  # noqa: E402


class CompoundGraphs:
    """All compounds packed into one disconnected graph for scatter MPNN.

    Levels: 'none' graph only; 'solv' +dGsolv; 'full' +xtb Mulliken node charge
    + HOMO/LUMO/gap + RDKit descriptors; 'rich' also swaps in CPCM-X solvation.
    """

    def __init__(self, mets, ensemble=None, qmfeat=None, level="full", cpcmx_solv=None):
        self.ids = [m["id"] for m in mets]
        self.idx = {c: i for i, c in enumerate(self.ids)}
        ensemble = ensemble or {}; qmfeat = qmfeat or {}; cpcmx_solv = cpcmx_solv or {}
        node_feats, batch, src, dst, edge_feats, qm = [], [], [], [], [], []
        offset = 0
        use_atom_qm = level in ("full", "rich")
        for i, m in enumerate(mets):
            mol = Chem.MolFromSmiles(m["smiles"])
            n = mol.GetNumAtoms()
            qf = qmfeat.get(m["id"]) or {}
            mull = qf.get("mulliken") or []
            for a in mol.GetAtoms():
                af = atom_features(a)
                if use_atom_qm:
                    af = af + [mull[a.GetIdx()] if a.GetIdx() < len(mull) else 0.0]
                node_feats.append(af); batch.append(i)
            for b in mol.GetBonds():
                u, v = b.GetBeginAtomIdx() + offset, b.GetEndAtomIdx() + offset
                bf = bond_features(b)
                src += [u, v]; dst += [v, u]; edge_feats += [bf, bf]
            for a in range(n):                          # self-loops
                src.append(a + offset); dst.append(a + offset)
                edge_feats.append([0.0] * BOND_DIM)
            offset += n
            e = (ensemble.get(m["id"]) or [{}])[0]
            dgsolv = e.get("dGsolv_kJ")
            gvec = []
            if level in ("solv", "full", "rich"):
                gvec.append((dgsolv if dgsolv is not None else 0.0) / 100.0)
            if level in ("full", "rich"):
                gvec += [qf.get("homo") or 0.0, qf.get("lumo") or 0.0,
                         (qf.get("gap") or 0.0) / 5.0]
                gvec += rdkit_descriptors(mol)
            if level == "rich":
                cx = cpcmx_solv.get(m["id"])
                gvec.append((cx if cx is not None else (dgsolv or 0.0)) / 100.0)
            qm.append(gvec or [0.0])
        self.atom_dim = len(node_feats[0])
        self.x = torch.tensor(node_feats, dtype=torch.float32)
        self.batch = torch.tensor(batch, dtype=torch.long)
        self.edge_index = torch.tensor([src, dst], dtype=torch.long)
        self.edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        self.qm = torch.tensor(qm, dtype=torch.float32)
        self.n_comp = len(self.ids)

    def pack(self):
        return dict(x=self.x, edge_index=self.edge_index, edge_attr=self.edge_attr,
                    qm=self.qm, batch=self.batch, atom_dim=self.atom_dim,
                    n_comp=self.n_comp, bond_dim=BOND_DIM)


def compound_count_feats(mets):
    """Atom-type-count fingerprint per compound (group-contribution style baseline)."""
    rows = []
    for m in mets:
        mol = Chem.MolFromSmiles(m["smiles"])
        c = {s: 0 for s in ATOM_VOCAB}
        for a in mol.GetAtoms():
            s = a.GetSymbol()
            c[s if s in c else "*"] += 1
        rows.append([c[s] for s in ATOM_VOCAB] + [float(m.get("charge", 0)), 1.0])
    return torch.tensor(rows, dtype=torch.float32)
