"""Regression tests for failed minimum probes and mixed-geometry energy terms."""
import numpy as np
import pytest
from ase import Atoms
from metag import pipeline as P
from metag.energetics.conformers import UniqueMinima

@pytest.mark.parametrize('converged,energies,expected',[
    ([False,False],[-100.,-100.],'inconclusive'),
    ([True,False],[-100.,-100.],'inconclusive'),
    ([True,True],[-100.,float('nan')],'inconclusive'),
    ([True,True],[-100.,-99.],'artefact'),
    ([True,False],[-102.,-100.],'saddle'),
])
def test_mode_following_requires_usable_evidence(monkeypatch,converged,energies,expected):
    monkeypatch.setattr(P,'batched_energies',lambda *a,**k:np.array([-100./P.EV2KJ]))
    monkeypatch.setattr(P,'batched_fire',lambda pu,ats,**k:(ats,np.array(energies)/P.EV2KJ,np.array(converged)))
    monkeypatch.setattr(P,'same_connectivity',lambda *a:True)
    verdict,pos=P._follow_imaginary_mode(None,['H','H'],np.array([[0.,0.,0.],[0.,0.,.74]]),
        np.array([[0.,0.,.1],[0.,0.,-.1]]),0,1,set(),'test')
    assert verdict==expected
    assert (pos is not None)==(expected=='saddle')


def test_auxiliary_energy_follows_selected_primary_geometry():
    u=UniqueMinima();aux='cosmo' if P.SOLV_MODEL!='cosmo' else 'alpb';also={aux:[]}
    a=Atoms('H2',positions=[[0,0,0],[0,0,.74]])
    P._add_minimum(u,also,a,-10.,{P.SOLV_MODEL:-5.,aux:-20.})
    b=a.copy();b.positions[1,2]+=.0005
    P._add_minimum(u,also,b,-10.1,{P.SOLV_MODEL:-6.,aux:-1.})
    assert u.G==pytest.approx([-16.1])
    assert also[aux]==pytest.approx([-11.1])  # higher auxiliary energy, correct geometry
    P._add_minimum(u,also,a,-10.2,{P.SOLV_MODEL:-1.,aux:-50.})
    assert also[aux]==pytest.approx([-11.1])  # rejected primary geometry cannot replace it
    P._add_minimum(u,also,b,-10.3,{P.SOLV_MODEL:-7.,aux:None})
    assert also[aux]==[None]  # missing value must not be inherited from old geometry


def test_robustness_policy_has_distinct_cache_identity():
    assert P.PHYSICS_VERSION not in {'2026-10-01c','2026-10-07b','2026-10-10d'}
    assert P._IMPLICIT_SETTINGS['physics']==P.PHYSICS_VERSION


def embed(smi):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol=Chem.AddHs(Chem.MolFromSmiles(smi))
    assert AllChem.EmbedMolecule(mol,randomSeed=17)==0
    atoms=Atoms([a.GetSymbol() for a in mol.GetAtoms()],positions=mol.GetConformer().GetPositions())
    return mol,atoms


def test_mirror_geometry_cannot_keep_specified_chiral_identity():
    from metag.energetics.conformers import same_stereochemistry,bond_graph,same_connectivity
    mol,atoms=embed('C[C@H](O)C(=O)O')
    assert same_stereochemistry(atoms,mol)
    mirrored=atoms.copy();mirrored.positions[:,0]*=-1
    assert same_connectivity(mirrored,bond_graph(mol))
    assert not same_stereochemistry(mirrored,mol)


def test_double_bond_isomer_cannot_pass_connectivity_only():
    from metag.energetics.conformers import same_stereochemistry,bond_graph,same_connectivity
    trans,a=embed('C/C=C/C');cis,b=embed('C/C=C\\C')
    assert same_connectivity(b,bond_graph(trans))
    assert same_stereochemistry(a,trans) and same_stereochemistry(b,cis)
    assert not same_stereochemistry(b,trans)


def test_unspecified_stereochemistry_is_not_silently_assigned():
    from rdkit import Chem
    from metag.energetics.conformers import same_stereochemistry
    mol,atoms=embed('CC(O)C(=O)O')
    original=Chem.MolToSmiles(mol)
    assert same_stereochemistry(atoms,mol)
    atoms.positions[:,0]*=-1
    assert same_stereochemistry(atoms,mol)
    assert Chem.MolToSmiles(mol)==original


def test_stereo_check_accepts_proper_rotation_and_translation():
    from metag.energetics.conformers import same_stereochemistry
    mol,atoms=embed('C[C@H](O)C(=O)O')
    atoms.rotate(73,'z');atoms.translate([12,-8,3])
    assert same_stereochemistry(atoms,mol)


def test_final_requires_all_sampled_thermal_corrections():
    mol, atoms = embed('CCO')
    u = UniqueMinima(template=mol, e_tol=-1)
    u.add(atoms, -100, -100)
    u.add(atoms, -80, -80)  # outside provisional window, but entropy can reverse ordering
    from metag.energetics.thermal_ensemble import ThermalTrack
    tr = ThermalTrack(None, mol, 0, 1, correction=None, combine=P.boltz, window=15, maximum=1)
    with pytest.raises(ValueError, match='every sampled'):
        tr.final(u, {}, 0., 0, validated={0: 0.})
    total, aux, info = tr.final(u, {'aux': [-90., None]}, 0., 0, validated={0: 0., 1: -40.})
    assert total == pytest.approx(P.boltz([-100., -120.]))
    assert aux['aux'] == [-90., None]
    assert info['n_reference_thermal_approximation'] == 0


def test_validation_rebuild_merges_resolved_basins_and_aligns_aux(monkeypatch):
    mol, atoms = embed('CCO')
    u = UniqueMinima(template=mol, e_tol=-1)
    u.add(atoms, -100., -110.); u.add(atoms, -100.2, -111.)
    def validate(pu, one, *args):
        g = 5. if one.G[0] == -110. else 7.
        return g, {}, dict(index=0, kind='as_is', pos=one.ref[0].GetConformer().GetPositions(), rejected=[])
    monkeypatch.setattr(P, '_thermal_at_minimum', validate)
    out, aux, thermal, info = P._validated_thermal_minima(
        None, u, {'aux': [-120., -105.]}, mol, None, 0, 1, 'test', lambda *_: None)
    assert out.G == [-111.]
    assert aux == {'aux': [-105.]}
    assert thermal == [7.]
    assert info['n_merged_after_resolution'] == 1
    row=info['validated_minima'][0]
    assert row['total_kj']==-104.
    assert row['electronic_kj']==-100.2
    np.testing.assert_array_equal(row['geometry']['positions_angstrom'],
                                  out.ref[0].GetConformer().GetPositions())
    monkeypatch.setattr(P, 'THERMAL_VALIDATION_MAX', 1)
    assert P._validated_thermal_minima(None,u,{},mol,None,0,1,'test',print) is None


def test_truncation_noise_cannot_widen_acceptance(monkeypatch):
    def score(*args, **kw):
        return dict(routes={'truncated': kw.get('allow_truncate',True)},
                    dG_raw=0 if kw.get('trunc_radius') == 2 else 20,
                    U_samp=30, suspect=None)
    monkeypatch.setattr(P, 'score_reaction', score)
    result=P._score_trunc_validated(None,{},[],1,1,lambda *_:None,'test',2)
    assert not result['routes']['truncated']
    assert result['trunc_validation']['threshold'] == 5
    assert result['trunc_validation']['verdict'] == 'rejected: sampling unresolved'


def test_explicit_water_identity_rejects_proton_transfer_and_mirror():
    mol, solute = embed('C[C@H](O)C(=O)O')
    water = Atoms('OH2', positions=[[20,0,0],[20.96,0,0],[19.68,.9,0]])
    cluster = solute + water
    assert P._explicit_cluster_identity(cluster,mol,1)
    mirror=cluster.copy(); mirror.positions[:len(solute),0]*=-1
    assert not P._explicit_cluster_identity(mirror,mol,1)
    transferred=cluster.copy(); transferred.positions[-1]=solute.positions[0]+[0,0,1.0]
    assert not P._explicit_cluster_identity(transferred,mol,1)


def test_macro_ladder_recovers_independent_identical_sites_without_double_counting():
    import math
    from metag.energetics.protonation import macroscopic_transform, site_free_energy
    # Two independent identical microscopic pKas=7 give sequential macro pKas 7 +/- log10(2).
    ladder=[7-math.log10(2),7+math.log10(2)]
    for ph in (-100., 6., 7., 8., 100.):
        result=macroscopic_transform(ladder,ph,5.708,reference_deprotonations=0)
        assert result['G_transform']==pytest.approx(2*site_free_energy(7,'acid',ph,5.708))
        assert sum(result['populations'])==pytest.approx(1.)
    middle=macroscopic_transform(ladder,7,5.708,reference_deprotonations=1)
    assert middle['populations']==pytest.approx([.25,.5,.25])
    assert middle['G_transform']==pytest.approx(-5.708*math.log10(2))
    with pytest.raises(ValueError,match='reference_deprotonations'):
        macroscopic_transform(ladder,7,5.708,reference_deprotonations=3)


def test_unresolved_nonreference_candidate_fails_whole_ensemble(monkeypatch):
    mol, atoms = embed('CCO')
    u = UniqueMinima(template=mol, e_tol=-1)
    u.add(atoms,-100.,-110.); u.add(atoms,-80.,-90.)
    def validate(pu,one,*args,**kwargs):
        g = 5. if one.G[0] == -110. else None
        return g, {}, dict(index=0,kind='as_is',pos=None,rejected=[])
    monkeypatch.setattr(P,'_thermal_at_minimum',validate)
    assert P._validated_thermal_minima(None,u,{},mol,None,0,1,'test',print) is None
    assert u.G==[-110.,-90.]  # transactional: input survives a failed validation


def test_auxiliary_nan_cannot_be_cached():
    assert P._complete_auxiliary_values([-10.,float('nan')]) is None
