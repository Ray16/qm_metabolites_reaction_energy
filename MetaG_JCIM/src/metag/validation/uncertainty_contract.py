"""Publish intervals only when their declared calibration applies."""
import math


def prediction_uncertainty(sigma,lower,upper,center,info):
    """Keep nominal historical widths separate from a calibrated95% interval."""
    if any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
           for value in (sigma,lower,upper,center)) or sigma<0 or not lower<=center<=upper:
        raise ValueError('Uncertainty diagnostics must be finite, ordered and nonnegative')
    eligible=(info.get('coverage_calibrated') is True and info.get('level_calibrated') is True
              and info.get('level')==95 and not info.get('config_mismatch')
              and not info.get('ood_flags') and not info.get('extrapolated_features'))
    metadata=dict(info,coverage_calibrated=eligible,level_calibrated=eligible,
                  externally_calibrated=False, external_validation_available=False,
                  formal_coverage_guarantee=False,
                  interval_status='calibrated_internal_CV' if eligible else 'unavailable_uncalibrated',
                  asserted_coverage_percent=95 if eligible else None)
    diagnostic={'sigma_kj':sigma,'interval_kj':[lower,upper],'center_kj':center,
                'requested_level':info.get('level'),'asserted_coverage_percent':None,
                'would_include_zero':lower<=0<=upper,
                'scope':'Historical calibration-model width and sampling diagnostic; no coverage claim outside its calibrated configuration/domain.'}
    return {'sigma_pred':sigma if eligible else None,
            'ci95':[lower,upper] if eligible else [None,None],
            'ci_center':center if eligible else None,'ci_info':metadata,
            'uncertainty_diagnostic':diagnostic,
            'unresolved':lower<=0<=upper if eligible else None,
            'sign_status':('calibrated_interval_includes_zero' if lower<=0<=upper else 'calibrated_interval_excludes_zero')
                           if eligible else 'uncertainty_unavailable'}
