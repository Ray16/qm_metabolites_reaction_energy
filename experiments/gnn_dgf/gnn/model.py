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
    def __init__(self, atom_dim, bond_dim, qm_dim, hidden=96, layers=3, drop=0.1):
        super().__init__()
        self.embed = nn.Linear(atom_dim, hidden)
        self.edge = nn.Linear(bond_dim, hidden)
        self.msg = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.upd = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                          nn.Dropout(drop), nn.Linear(hidden, hidden))
            for _ in range(layers))
        self.qm_norm = nn.LayerNorm(qm_dim) if qm_dim > 1 else nn.Identity()
        self.readout = nn.Sequential(
            nn.Linear(hidden + qm_dim, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, g):
        h = torch.relu(self.embed(g.x))
        e = self.edge(g.edge_attr)
        src, dst = g.edge_index
        for msg, upd in zip(self.msg, self.upd):
            m = torch.relu(msg(h)[src] + e)                    # message per edge
            agg = torch.zeros_like(h).index_add_(0, dst, m)     # sum into targets
            h = h + upd(torch.cat([h, agg], dim=-1))            # residual update
        pooled = torch.zeros(g.n_comp, h.size(1), device=h.device).index_add_(0, g.batch, h)
        return self.readout(torch.cat([pooled, self.qm_norm(g.qm)], dim=-1)).squeeze(-1)
