"""One calibration-only mixing coefficient; no model, data or network access."""
import numpy as np

from laya_decision_views import gated_equal_fusion, probability_margin

EPSILON = 1e-12
MIXING_L2 = .01


def positive_probability(prob):
    probability_margin(prob)
    if not len(prob):
        raise ValueError('Empty probability block')
    return np.clip(np.asarray(prob, dtype=np.float64)[:, 1], EPSILON, 1-EPSILON)


def binary_probability(positive):
    p = np.asarray(positive, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all():
        raise ValueError('Expected a nonempty finite probability vector')
    p = np.clip(p, EPSILON, 1-EPSILON)
    return np.column_stack((1-p, p))


def mix_probabilities(first, second, coefficient):
    if (not np.isscalar(coefficient) or not np.isfinite(coefficient)
            or not 0 <= coefficient <= 1):
        raise ValueError('Mixture coefficient must be in [0, 1]')
    a, b = positive_probability(first), positive_probability(second)
    if a.shape != b.shape:
        raise ValueError('Mixture rows must align')
    return binary_probability(coefficient*a+(1-coefficient)*b)


def _calibration_labels(labels, count):
    y = np.asarray(labels)
    if y.shape != (count,) or not np.isin(y, [0, 1]).all():
        raise ValueError('Expected aligned binary calibration labels')
    return y


def fit_mixing_coefficient(first, second, labels):
    """Convex probability-mixture NLL with fixed shrinkage to equal weights."""
    a, b = positive_probability(first), positive_probability(second)
    if a.shape != b.shape:
        raise ValueError('Calibration views must align')
    y = _calibration_labels(labels, len(a))
    difference = a-b

    def derivative(weight):
        p = weight*a+(1-weight)*b
        return float(np.mean((p-y)*difference/(p*(1-p)))
                     + 2*MIXING_L2*(weight-.5))

    lo, hi = 0., 1.
    if np.array_equal(a, b):
        weight = .5
    elif derivative(lo) >= 0:
        weight = lo
    elif derivative(hi) <= 0:
        weight = hi
    else:
        for _ in range(80):
            mid = (lo+hi)/2
            if derivative(mid) > 0:
                hi = mid
            else:
                lo = mid
        weight = (lo+hi)/2
    p = weight*a+(1-weight)*b
    nll = float(-np.mean(np.where(y == 1, np.log(p), np.log1p(-p))))
    return {'coefficient': weight, 'weights': [weight, 1-weight],
            'bounds': [0., 1.], 'l2_to_equal': MIXING_L2, 'probability_epsilon': EPSILON,
            'calibration_count': len(y), 'calibration_nll': nll,
            'penalized_objective': nll+MIXING_L2*(weight-.5)**2,
            'derivative_at_solution': derivative(weight),
            'method': 'convex_derivative_bisection_80_steps',
            'fit_split': 'original_calibration_only'}


def apply_gated_mixture(first_head, first_tail, second_head, second_tail, truncated, coefficient):
    first = gated_equal_fusion(first_head, first_tail, truncated)
    second = gated_equal_fusion(second_head, second_tail, truncated)
    return mix_probabilities(first, second, coefficient)


def fit_historical_ols(views, labels):
    """Explicit diagnostic interpretation, not recovery of an unspecified fit."""
    if len(views) != 8:
        raise ValueError('Historical diagnostic fixes exactly eight views')
    columns = [positive_probability(p) for p in views]
    if len({p.shape for p in columns}) != 1:
        raise ValueError('Historical views must align')
    x = np.column_stack(columns)
    y = _calibration_labels(labels, len(x))
    weights, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
    if not np.isfinite(weights).all():
        raise FloatingPointError('Nonfinite OLS weights')
    return {'weights': weights.tolist(), 'rank': int(rank), 'singular_values': singular.tolist(),
            'method': 'positive_probability_OLS_no_intercept_rcond_None',
            'calibration_count': len(y), 'fit_split': 'original_calibration_only',
            'diagnostic_only': True, 'original_method_fully_specified': False,
            'output_clip': [EPSILON, 1-EPSILON]}


def apply_historical_ols(views, fit):
    x = np.column_stack([positive_probability(p) for p in views])
    weights = np.asarray(fit['weights'], dtype=float)
    if x.shape[1] != 8 or weights.shape != (8,) or not np.isfinite(weights).all():
        raise ValueError('Invalid frozen eight-view OLS')
    raw = x@weights
    return binary_probability(raw), int(np.sum((raw < EPSILON) | (raw > 1-EPSILON)))
