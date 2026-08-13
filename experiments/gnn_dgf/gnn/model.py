"""The GNN: a message-passing network that maps a compound graph to a scalar
formation energy f; reaction dG = S @ f.  Pure torch (no rdkit) so it imports
in the GPU (uma) env.  Sum-readout (energy is extensive) + LayerNorm'd QC block.
"""
import torch
import torch.nn as nn

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Graph:
    """All compounds packed into one disconnected graph; moved to DEV."""

    def __init__(self, d):
        for k in ("x", "edge_index", "edge_attr", "qm", "batch"):
            setattr(self, k, d[k].to(DEV))
        self.atom_dim = d["atom_dim"]
        self.bond_dim = d["bond_dim"]
        self.n_comp = d["n_comp"]


class MPNN(nn.Module):
    """Message-passing net -> scalar per-compound f.

    qm_in_messages: if True, the (LayerNorm'd) graph-level QM vector is broadcast
    to each atom and concatenated into the node features BEFORE message passing,
    so QM informs the learned representation rather than only the final readout
    (variant #4). Default False reproduces the readout-only baseline exactly.
    """

    def __init__(self, atom_dim, bond_dim, qm_dim, hidden=96, layers=3, drop=0.1,
                 qm_in_messages=False):
        super().__init__()
        self.qm_in_messages = qm_in_messages and qm_dim > 1
        self.qm_norm = nn.LayerNorm(qm_dim) if qm_dim > 1 else nn.Identity()
        embed_in = atom_dim + (qm_dim if self.qm_in_messages else 0)
        self.embed = nn.Linear(embed_in, hidden)
        self.edge = nn.Linear(bond_dim, hidden)
        self.msg = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.upd = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                          nn.Dropout(drop), nn.Linear(hidden, hidden))
            for _ in range(layers))
        self.readout = nn.Sequential(
            nn.Linear(hidden + qm_dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, g):
        qm = self.qm_norm(g.qm)
        x = torch.cat([g.x, qm[g.batch]], dim=-1) if self.qm_in_messages else g.x
        h = torch.relu(self.embed(x))
        e = self.edge(g.edge_attr)
        src, dst = g.edge_index
        for msg, upd in zip(self.msg, self.upd):
            m = torch.relu(msg(h)[src] + e)                    # message per edge
            agg = torch.zeros_like(h).index_add_(0, dst, m)     # sum into targets
            h = h + upd(torch.cat([h, agg], dim=-1))            # residual update
        pooled = torch.zeros(g.n_comp, h.size(1), device=h.device).index_add_(0, g.batch, h)
        return self.readout(torch.cat([pooled, qm], dim=-1)).squeeze(-1)


class CondHead(nn.Module):
    """Reaction-level correction h(cond) added to S@f (variant #1). Small MLP on
    per-reaction measurement conditions [pH, ionic_strength, T, pMg]. Kept tiny
    (few params) so it corrects for condition-driven label spread without
    overpowering the compound model."""

    def __init__(self, cond_dim=4, hidden=16, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, 1))

    def forward(self, cond):
        return self.net(cond).squeeze(-1)
