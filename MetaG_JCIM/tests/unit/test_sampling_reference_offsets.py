"""Deterministic reference changes do not create spurious independent noise."""
import pytest
from metag.assembly import assemble_reaction


def assemble(offsets):
    return assemble_reaction({'a':[-1,0,'a'],'b':[1,0,'b']},{'a':2.,'b':3.},
        {'a':.5,'b':.5},n_hplus=0,proton_energy=0.,pka_sites=[],ph=7.,rt_ln10=5.7,
        sampling_groups={'a':'same','b':'same'},sampling_offsets=offsets)


def test_deterministic_reference_offsets_affect_only_sampling_identity_validation():
    result=assemble({'a':1.})
    assert result.total==1. and result.species_sum==1. and result.sampling_sigma==0.


def test_wrong_sampling_gauge_is_still_refused():
    with pytest.raises(ValueError,match='inconsistent'):assemble({'a':2.})


@pytest.mark.parametrize('offsets',[{'unknown':1.},{'a':float('nan')},{'a':float('inf')},{'a':True}])
def test_unknown_or_nonfinite_offsets_never_bypass_sampling_checks(offsets):
    with pytest.raises(ValueError):assemble(offsets)
