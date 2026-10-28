"""Input protonation drawings must not create or lose supported base states."""
import pytest
from metag.routing.pka_transform import _neutralize_v2, build_ph0_reaction
from metag.energetics.protonation import reaction_transform


@pytest.mark.parametrize('charged,neutral', [
    ('CNC(N)=[NH2+]', 'CNC(=N)N'),
    ('CC(N)=[NH2+]', 'CC(N)=N'),
    ('NC(=[NH2+])N', 'NC(=N)N'),
    ('C[NH+]=C(N)N', 'CN=C(N)N'),
    ('NC(=[NH2+])NCCC[NH3+]', 'NC(=N)NCCCN'),
])
def test_neutral_charged_drawings_have_identical_reference_and_terms(charged, neutral):
    a, b = _neutralize_v2(charged), _neutralize_v2(neutral)
    assert a[0] == b[0]
    assert sorted(a[1]) == sorted(b[1])
    assert sorted(a[2]) == sorted(b[2])
    assert a[3] == b[3] == 0


@pytest.mark.parametrize('smi,n', [('NC(=N)N',1), ('NC(=[NH2+])N',1),
                                  ('NC(=N)NCCNC(=N)N',2), ('CC(=N)C',0),
                                  ('CC(=O)N',0), ('c1ccncc1',0)])
def test_one_base_term_per_supported_group(smi, n):
    assert len(_neutralize_v2(smi)[2]) == n


def test_mixed_drawings_do_not_create_reaction_energy_and_reverse_exactly():
    species = {'drawn_cation': [-1,1,'CNC(N)=[NH2+]'],
               'drawn_neutral': [1,0,'CNC(=N)N']}
    ref, terms, nh = build_ph0_reaction(species, force_base=True)
    reverse = {n: [-c,q,s] for n,(c,q,s) in species.items()}
    rr, rt, rnh = build_ph0_reaction(reverse, force_base=True)
    assert nh == rnh == 0
    assert ref['drawn_cation'][2] == ref['drawn_neutral'][2]
    assert reaction_transform(terms, 7., 5.708) == 0
    assert reaction_transform(rt, 7., 5.708) == -reaction_transform(terms, 7., 5.708)


@pytest.mark.parametrize('smiles', ['CNC(N)=[NH+]P(=O)(O)O', 'CNC(N)=NP(=O)(O)O'])
def test_phosphoguanidine_requires_independent_phosphate_speciation(smiles):
    with pytest.raises(ValueError, match='phosphoramidate.*quarantined'):
        _neutralize_v2(smiles)
