"""Conformer deduplication before Boltzmann weighting (metag.energetics.conformers.UniqueMinima): repeated
samples of one basin must NOT add degeneracy (the old sum lowered G by RT ln N), symmetry-equivalent copies
(rotated, translated, atom-permuted) must merge, and genuinely different conformers must stay distinct."""
import numpy as np
import pytest
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from metag.energetics.conformers import boltz, pool_confs, UniqueMinima, principal_moments, KT


def _butane_confs():
    m = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    AllChem.EmbedMultipleConfs(m, numConfs=30, randomSeed=7)
    res = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=2000)
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    out = []
    for cid, (conv, e) in zip(range(m.GetNumConformers()), res):
        out.append((Atoms(syms, positions=m.GetConformer(cid).GetPositions()), e * 4.184))
    return out, m


def test_old_sum_had_sample_count_bias():
    # the defect being fixed: 16 identical copies lower the Boltzmann G by RT ln 16 = 6.87 kJ
    assert abs((boltz([0.0]) - boltz([0.0] * 16)) - KT * np.log(16)) < 1e-9


def test_pool_confs_reports_failed_fallback(monkeypatch):
    monkeypatch.setattr(AllChem, "EmbedMultipleConfs", lambda *args, **kwargs: [])
    monkeypatch.setattr(AllChem, "EmbedMolecule", lambda *args, **kwargs: -1)
    with pytest.raises(ValueError, match="could not generate a 3D conformer"):
        pool_confs("CC", 0, seed=1, pool=4)


def test_duplicates_do_not_add_degeneracy():
    a = Atoms("OH2", positions=[[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]])
    u = UniqueMinima(template=Chem.AddHs(Chem.MolFromSmiles("O")))
    rng = np.random.default_rng(0)
    for k in range(16):
        b = a.copy()
        b.rotate(rng.uniform(0, 360), rng.normal(size=3))
        b.translate(rng.normal(size=3))
        u.add(b, -1000.0 + rng.uniform(-0.1, 0.1), -1010.0)
    assert len(u) == 1 and u.n_seen == 16
    assert abs(boltz(u.G) - (-1010.0)) < 1e-9          # one state: no -RT ln 16


def test_permuted_atoms_merge():
    # symmetry-equivalent copy with the two H's swapped is the same minimum
    a = Atoms("OHH", positions=[[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]])
    b = Atoms("OHH", positions=[[0, 0, 0.1173], [0, -0.7572, -0.4692], [0, 0.7572, -0.4692]])
    assert np.allclose(principal_moments(a), principal_moments(b))
    u = UniqueMinima(template=Chem.AddHs(Chem.MolFromSmiles("O"))); u.add(a, 0.0, 0.0)
    assert u.add(b, 0.0, 0.0) is False


def test_mirror_image_conformers_stay_distinct():
    # achiral butane: gauche+ and gauche- are mirror images (same E, same moments) but DISTINCT states
    m = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    AllChem.EmbedMolecule(m, randomSeed=3)
    conf = m.GetConformer()
    from rdkit.Chem import rdMolTransforms as T
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    T.SetDihedralDeg(conf, 0, 1, 2, 3, 60.0)
    gp = Atoms(syms, positions=conf.GetPositions())
    gm = Atoms(syms, positions=conf.GetPositions() * np.array([1.0, 1.0, -1.0]))   # reflection
    assert np.allclose(principal_moments(gp), principal_moments(gm), rtol=1e-6)
    u = UniqueMinima(template=m)
    u.add(gp, 0.0, 0.0)
    assert u.add(gm, 0.0, 0.0) is True and len(u) == 2


def test_distinct_conformers_kept_and_count_is_sampling_invariant():
    confs, tmpl = _butane_confs()                      # 30 MMFF minima: anti + gauche+ + gauche- basins
    u = UniqueMinima(template=tmpl)
    for a, e in confs:
        u.add(a, e, e)
    assert len(u) == 3                                 # anti, g+, g- survive; the ~30 copies collapse
    # doubling the sample set (same basins again) leaves the ensemble free energy unchanged
    u2 = UniqueMinima(template=tmpl)
    for a, e in confs + confs:
        u2.add(a, e, e)
    assert len(u2) == len(u) and abs(boltz(u2.G) - boltz(u.G)) < 1e-9


def test_duplicate_keeps_lower_G():
    a = Atoms("OH2", positions=[[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]])
    u = UniqueMinima(template=Chem.AddHs(Chem.MolFromSmiles("O"))); u.add(a, 0.0, 5.0); u.add(a.copy(), 0.1, 3.0)
    assert u.G == [3.0]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  {name} OK")
    print("PASS: conformer dedup")
