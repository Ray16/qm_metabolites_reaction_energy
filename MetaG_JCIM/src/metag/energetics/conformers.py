"""Conformer generation, molecular identity and unique-minimum bookkeeping.

Production energetics are assembled by pipeline.py, with solvation implemented
in thermal.py. Archived benchmark demonstrations remain in historical study
snapshots; this module contains no experimental target or alternate scorer.
"""
import os
import warnings

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms

EV2KJ = 96.485
# Retained gas-energy tolerance from the historical float32 implementation.
# The current backend returns float64 totals; this tolerance and its float32-step
# scaling remain heuristic deduplication criteria, not its reported precision.
DEDUP_E_TOL = 1.5
KT = 2.4789
XTB_WORKERS = max(1, int(os.environ.get("METAG_XTB_WORKERS", "8")))


def boltz(Gs):
    """-RT ln Σ exp(-G_i/RT). Each entry is ONE thermodynamic state, so Gs must be UNIQUE minima
    (see UniqueMinima): N copies of the same basin lower the result by RT ln N (6.9 kJ at N=16)."""
    Gs = np.asarray(Gs); ref = Gs.min()
    return float(ref - KT * np.log(np.exp(-(Gs - ref) / KT).sum()))


def principal_moments(atoms):
    """Sorted principal moments of inertia (amu Å²). Invariant to rotation, translation AND atom
    permutation, so it identifies symmetry-equivalent copies of a minimum without an atom mapping."""
    m = atoms.get_masses()
    x = atoms.get_positions() - atoms.get_center_of_mass()
    I = (np.eye(3) * (m[:, None] * x * x).sum() - np.einsum("i,ij,ik->jk", m, x, x))
    return np.sort(np.linalg.eigvalsh(I))


class UniqueMinima:
    """Relaxed conformers deduplicated to UNIQUE minima before Boltzmann weighting.

    Different ETKDG seeds (and conformers within one pool) routinely relax into the SAME basin; counting
    each copy as a separate state adds a spurious degeneracy -RT ln(N_copies) that depends on how many
    samples were drawn, so it does not cancel across a reaction whose species stop at different sample
    counts.

    A rediscovered basin retains the representative with the lower stored vertical
    free energy. This is not an unbiased average over noisy repeated evaluations:
    geometry and solvent-surface variation can favor a lower representative. The
    final minimum/thermal checks do not bound that selection effect. Sampling
    diagnostics must not be interpreted as a predictive error distribution.

    Criterion: same minimum iff |ΔE_UMA| < e_tol AND the all-atom, symmetry-aware RMSD after optimal
    PROPER rotation is < rmsd_tol (RDKit GetBestRMS over graph automorphisms: permuted methyl / phosphate
    / carboxylate atoms and rotated copies merge). Proper rotations only: mirror-image conformers of an
    achiral molecule (gauche+/gauche-) are distinct states and stay separate -- the rotational-constant
    shortcut used first merged them (34 pairs for citric acid), undercounting conformational entropy for
    achiral species only, a bias that does not cancel against chiral partners.

    `template` = the RDKit molecule (with H) whose atom order matches the Atoms (pool_confs builds from
    Chem.AddHs(MolFromSmiles(smi))). Without a template (explicit water clusters) it falls back to
    energy + principal moments (can merge mirror images; that path is off by default)."""

    def __init__(self, e_tol=DEDUP_E_TOL, rmsd_tol=0.25, template=None, rot_tol=0.01, max_matches=20000):
        self.e_tol, self.rmsd_tol, self.rot_tol = e_tol, rmsd_tol, rot_tol
        self.template, self.max_matches = template, max_matches
        self.E, self.G, self.ref = [], [], []
        self.n_seen = 0
        self.last_index = None
        self.last_replaced = False

    def _e_tol(self, e):
        """Energy tolerance at energy e (kJ): never below 2 float32 steps of the UMA energy in eV.
        A negative e_tol (legacy A/B: never merge) is returned unchanged."""
        if self.e_tol < 0:
            return self.e_tol
        return max(self.e_tol, 2.0 * float(np.spacing(np.float32(abs(e) / EV2KJ))) * EV2KJ)

    def _mol_at(self, atoms):
        from rdkit import Chem as _C
        from rdkit.Geometry import Point3D
        m = _C.Mol(self.template)
        m.RemoveAllConformers()
        conf = _C.Conformer(m.GetNumAtoms())
        for i, (x, y, z) in enumerate(atoms.get_positions()):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        m.AddConformer(conf, assignId=True)
        return m

    def _same(self, e, rep, j):
        if abs(e - self.E[j]) >= self._e_tol(e):
            return False
        if self.template is None:
            ref = np.maximum(np.maximum(np.abs(rep), np.abs(self.ref[j])), 1e-3)
            return bool(np.all(np.abs(rep - self.ref[j]) / ref < self.rot_tol))
        from rdkit.Chem import rdMolAlign
        # GetBestRMS aligns its probe in place. Energies were evaluated at the
        # original coordinates, so alignment must never change the candidate
        # that may become the stored representative (finite grids can depend
        # on orientation even when the underlying energy is invariant).
        return rdMolAlign.GetBestRMS(Chem.Mol(rep), self.ref[j], maxMatches=self.max_matches) < self.rmsd_tol

    def add(self, atoms, E_kJ, G_kJ):
        """Register a relaxed conformer; returns True if it is a NEW minimum. A duplicate keeps the
        representative with the lower G (its solvation may differ slightly by geometry noise)."""
        self.n_seen += 1
        self.last_index = None
        self.last_replaced = False
        rep = principal_moments(atoms) if self.template is None else self._mol_at(atoms)
        for j in range(len(self.E)):
            if self._same(E_kJ, rep, j):
                self.last_index = j
                if G_kJ < self.G[j]:                      # the representative IS the structure whose E/G is
                    self.G[j] = float(G_kJ)               # used: geometry, E and G are replaced together
                    self.E[j] = float(E_kJ)
                    self.ref[j] = rep
                    self.last_replaced = True
                return False
        self.E.append(float(E_kJ)); self.G.append(float(G_kJ)); self.ref.append(rep)
        self.last_index = len(self.G) - 1
        self.last_replaced = True
        return True

    def drop(self, indices):
        """Remove the minima at `indices` (e.g. saddle points found by the thermal validation)."""
        keep = [j for j in range(len(self.G)) if j not in set(indices)]
        self.E = [self.E[j] for j in keep]; self.G = [self.G[j] for j in keep]; self.ref = [self.ref[j] for j in keep]
        self.last_index = None
        self.last_replaced = False
        return keep

    def __len__(self):
        return len(self.G)


def bond_graph(template):
    """Set of bonded atom-index pairs of an RDKit molecule WITH explicit H (atom order = pool_confs)."""
    return {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in template.GetBonds()}


def perceived_graph(atoms, scale=1.25):
    """Bonded pairs perceived from 3D geometry: d_ij < scale * (r_cov,i + r_cov,j). With scale 1.25 an
    O-H covalent bond (0.97 A) is < 1.21 A but a hydrogen bond (>= 1.5 A) is not; P-O (1.6 A) < 2.16 A."""
    pt = Chem.GetPeriodicTable()
    r = np.array([pt.GetRcovalent(int(z)) for z in atoms.get_atomic_numbers()])
    x = atoms.get_positions()
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    i, j = np.where(np.triu(d < scale * (r[:, None] + r[None, :]), k=1))
    return set(zip(i.tolist(), j.tolist()))


def same_connectivity(atoms, ref_graph):
    """False if relaxation changed the bonding (e.g. a zwitterion's N-H proton moved to its O-, or a
    phosphate proton hopped): the relaxed structure is then a DIFFERENT species from the SMILES scored."""
    return perceived_graph(atoms) == ref_graph


def same_stereochemistry(atoms, template):
    """Check specified atom/bond stereochemistry against the actual 3D geometry.

    Atom order must match the explicit-H reference template. Unspecified stereo
    remains unspecified: this guard does not invent a stereoisomer or enumerate
    its alternatives. Connectivity is checked separately by the caller.
    """
    from rdkit.Geometry import Point3D
    positions = np.asarray(atoms.get_positions(), dtype=float)
    if len(atoms) != template.GetNumAtoms() or not np.isfinite(positions).all():
        return False
    if list(atoms.get_atomic_numbers()) != [a.GetAtomicNum() for a in template.GetAtoms()]:
        return False
    atom_tags = [(a.GetIdx(), a.GetChiralTag()) for a in template.GetAtoms()
                 if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED]
    bond_tags = [(b.GetIdx(), b.GetStereo()) for b in template.GetBonds()
                 if b.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)]
    if not atom_tags and not bond_tags:
        return True
    candidate = Chem.Mol(template)
    candidate.RemoveAllConformers()
    Chem.RemoveStereochemistry(candidate)
    conformer = Chem.Conformer(template.GetNumAtoms())
    conformer.Set3D(True)
    for i, xyz in enumerate(positions):
        conformer.SetAtomPosition(i, Point3D(*map(float, xyz)))
    candidate.AddConformer(conformer, assignId=True)
    Chem.AssignStereochemistryFrom3D(candidate, replaceExistingTags=True)
    if any(candidate.GetAtomWithIdx(i).GetChiralTag() != tag for i, tag in atom_tags):
        return False
    def bond_parity(tag):
        if tag in (Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOZ):
            return "cis"
        if tag in (Chem.BondStereo.STEREOTRANS, Chem.BondStereo.STEREOE):
            return "trans"
        return tag
    return all(bond_parity(candidate.GetBondWithIdx(i).GetStereo()) == bond_parity(tag)
               for i, tag in bond_tags)


TRANSITION_METALS = set(range(21, 31)) | set(range(39, 49)) | set(range(57, 81))


def spin_multiplicity(smiles, q):
    """Ground-state spin multiplicity (2S+1) for the UMA `spin` field. Metabolites are
    overwhelmingly closed-shell singlets (1), but a few open-shell species must be set
    explicitly or UMA computes the wrong electronic state:
      * O2 -- ground state is a TRIPLET (mult 3), not the singlet a bare `O=O` SMILES implies.
        Computed as singlet it is ~115 kJ/mol too high (measured), driving every O2-consuming
        oxygenase/oxidase reaction that-many-times too negative. (The common, high-impact case.)
      * Any odd-electron species -> doublet (mult 2): NO, superoxide O2*-, organic radicals.
    Deterministic in (smiles, q), so it needs no separate cache key for singlets."""
    from metag.chem import require_supported_isotopes
    require_supported_isotopes(smiles)
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return 1
    m = Chem.AddHs(m)
    # O2: two oxygens, no H, neutral -> triplet ground state
    zs = [a.GetAtomicNum() for a in m.GetAtoms()]
    if any(z in TRANSITION_METALS for z in zs):
        raise ValueError(f"{smiles}: transition-metal species -- spin state cannot be inferred from the SMILES "
                         "and UMA/xtb states are not validated here; not supported")
    if int(q) == 0 and sorted(zs) == [8, 8]:
        return 3
    n_rad = sum(a.GetNumRadicalElectrons() for a in m.GetAtoms())
    if n_rad:                                             # SMILES-specified radicals ([CH2], [O][O], ...):
        return n_rad + 1                                  # high-spin (all unpaired electrons parallel)
    n_elec = sum(zs) - int(q)
    return 2 if (n_elec % 2) else 1                       # odd electrons -> doublet, else singlet


def pool_confs(smiles, q, seed, pool, spin=1):
    from metag.chem import require_supported_isotopes
    require_supported_isotopes(smiles)
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3(); p.randomSeed = seed; p.pruneRmsThresh = 0.3
    cids = list(AllChem.EmbedMultipleConfs(m, numConfs=pool, params=p))
    if not cids:
        cid = AllChem.EmbedMolecule(m, randomSeed=seed, useRandomCoords=True)
        if cid < 0:
            raise ValueError(f"{smiles}: RDKit could not generate a 3D conformer (seed {seed})")
        cids = [cid]
    try:
        AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=200)
    except Exception as exc:
        warnings.warn(
            f"MMFF pre-optimization failed ({type(exc).__name__}: {exc}); "
            "continuing with the current coordinates for validated UMA relaxation.",
            RuntimeWarning, stacklevel=2)
    valid_cids = {conf.GetId() for conf in m.GetConformers()}
    cids = [cid for cid in cids if cid in valid_cids]
    if not cids:
        raise ValueError(f"{smiles}: RDKit generated no usable 3D conformers (seed {seed})")
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    return [Atoms(symbols=syms, positions=m.GetConformer(c).GetPositions(),
                  info={"charge": int(q), "spin": int(spin)}) for c in cids]
