from copy import deepcopy
import pytest
from metag.validation.uncertainty_contract import prediction_uncertainty


def calibrated():
    return {'coverage_calibrated':True,'level_calibrated':True,'level':95,
            'config_mismatch':[],'ood_flags':[],'calibration_basis':'internal grouped CV'}


@pytest.mark.parametrize('change',[{'coverage_calibrated':False},{'level_calibrated':False},
    {'level':90},{'config_mismatch':['new Hamiltonian']},{'ood_flags':['rare_element']},
    {'extrapolated_features':['size']},{'coverage_calibrated':'true'}])
def test_inapplicable_calibration_cannot_publish_ci_or_sign(change):
    info={**calibrated(),**change}
    row=prediction_uncertainty(4.,-18.,-2.,-10.,info)
    assert row['ci95']==[None,None] and row['sigma_pred'] is None
    assert row['unresolved'] is None and row['sign_status']=='uncertainty_unavailable'
    assert row['ci_info']['asserted_coverage_percent'] is None
    assert row['uncertainty_diagnostic']['interval_kj']==[-18.,-2.]
    assert not row['uncertainty_diagnostic']['would_include_zero']


def test_matched_declared_calibration_retains_interval_without_mutation():
    info=calibrated();saved=deepcopy(info)
    row=prediction_uncertainty(4.,-18.,-2.,-10.,info)
    assert info==saved and row['ci95']==[-18.,-2.] and row['sigma_pred']==4.
    assert row['unresolved'] is False
    assert row['ci_info']['asserted_coverage_percent']==95


@pytest.mark.parametrize('values',[(float('nan'),-1.,1.,0.),(-1.,-1.,1.,0.),
    (1.,2.,1.,0.),(True,-1.,1.,0.)])
def test_nonfinite_or_malformed_diagnostics_are_not_admitted(values):
    with pytest.raises(ValueError):prediction_uncertainty(*values,calibrated())
