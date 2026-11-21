"""Regression tests for linear errors-in-variables fits."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import curve_fit

import pap_toolbox as pap


def _sample():
    x = np.array([0.1, 0.9, 2.2, 2.8, 4.0, 5.2])
    y = np.array([0.7, 2.2, 4.8, 6.3, 8.4, 10.8])
    return x, y


def test_both_axis_fit_matches_analytic_deming_regression():
    """For constant independent uncertainties Deming has a closed form."""
    x, y = _sample()
    sx, sy = 0.3, 0.25
    fit = pap.linear_fit(x, y, xerr=sx, yerr=sy)
    cx, cy = x - np.mean(x), y - np.mean(y)
    xx, yy, xy = np.dot(cx, cx), np.dot(cy, cy), np.dot(cx, cy)
    ratio = (sy / sx) ** 2
    expected_slope = (
        yy - ratio * xx + np.sqrt((yy - ratio * xx) ** 2 + 4 * ratio * xy**2)
    ) / (2 * xy)
    expected_intercept = np.mean(y) - expected_slope * np.mean(x)
    assert fit.fit_method == "errors_in_variables"
    assert fit.slope == pytest.approx(expected_slope, rel=1e-9)
    assert fit.intercept == pytest.approx(expected_intercept, rel=1e-9)
    sigma_eff = np.hypot(sy, fit.slope * sx)
    np.testing.assert_allclose(fit.effective_y_err, sigma_eff)
    chi2 = np.sum(((y - fit.evaluate(x)) / sigma_eff) ** 2)
    assert fit.chi2 == pytest.approx(chi2)
    assert fit.chi2_red == pytest.approx(chi2 / (len(x) - 2))
    np.testing.assert_allclose(fit.pulls, fit.residuals / sigma_eff)


def test_large_horizontal_uncertainty_changes_slope_and_covariance():
    x, y = _sample()
    vertical = pap.linear_fit(x, y, yerr=0.25)
    both = pap.linear_fit(x, y, xerr=1.0, yerr=0.25)
    assert abs(both.slope - vertical.slope) > 1e-3
    assert both.slope_err > vertical.slope_err
    assert both.intercept_err > vertical.intercept_err
    assert np.all(np.linalg.eigvalsh(both.cov_matrix) > 0)


def test_heteroscedastic_both_axis_errors_and_range_filtering():
    x, y = _sample()
    dx = np.array([0.1, 0.2, 0.0, 0.35, 0.15, 0.7])
    dy = np.array([0.3, 0.5, 0.15, 0.25, 0.22, 0.4])
    df = pd.DataFrame({"x": x, "y": y, "dx": dx, "dy": dy})
    fit = pap.linear_fit(df, "x", "y", xerr="dx", yerr="dy", x_range=(0.9, 4.0))
    np.testing.assert_allclose(fit.x_data, x[1:5])
    np.testing.assert_allclose(fit.x_err, dx[1:5])
    np.testing.assert_allclose(fit.y_err, dy[1:5])
    scale = np.hypot(fit.y_err, fit.slope * fit.x_err)
    assert fit.chi2 == pytest.approx(np.sum((fit.residuals / scale)**2))
    assert fit.ndof == 2
    assert fit.cov_matrix.shape == (2, 2)


def test_only_horizontal_errors_matches_inverse_weighted_regression():
    """Exact y means minimizing horizontal distances, not an unweighted y-fit."""
    x, y = _sample()
    dx = np.array([0.3, 0.2, 0.5, 0.4, 0.35, 0.25])
    fit = pap.linear_fit(x, y, xerr=dx)
    (a, c), inverse_cov = curve_fit(
        lambda y, a, c: a * y + c, y, x, sigma=dx, absolute_sigma=True
    )
    expected = (1.0 / a, -c / a)
    transform = np.array([[-1.0/a**2, 0], [c/a**2, -1.0/a]])
    np.testing.assert_allclose(fit.params, expected, rtol=1e-8)
    np.testing.assert_allclose(
        fit.cov_matrix, transform @ inverse_cov @ transform.T, rtol=1e-8
    )
    assert fit.chi2 == pytest.approx(np.sum(((x - a*y-c)/dx)**2))
    np.testing.assert_allclose(
        fit.pulls, fit.residuals / (np.abs(fit.slope) * dx)
    )


def test_zero_horizontal_errors_leave_vertical_fit_unchanged():
    x, y = _sample()
    old = pap.linear_fit(x, y, yerr=0.2)
    zero = pap.linear_fit(x, y, xerr=0.0, yerr=0.2)
    np.testing.assert_allclose(old.params, zero.params, rtol=1e-12)
    np.testing.assert_allclose(old.cov_matrix, zero.cov_matrix, rtol=1e-12)
    assert zero.fit_method == "vertical"
    assert zero.chi2 == pytest.approx(old.chi2)


@pytest.mark.parametrize("errors", [0.0, [0.1, 0.0, 0.3, 0.2, 0.5, 0.2]])
def test_only_horizontal_errors_require_positive_nonzero_errors(errors):
    x, y = _sample()
    if np.isscalar(errors):
        # When xerr=0 exactly, it reduces to ordinary unweighted regression.
        fit = pap.linear_fit(x, y, xerr=errors)
        assert fit.fit_method == "vertical"
    else:
        with pytest.raises(ValueError, match="strikt positiv"):
            pap.linear_fit(x, y, xerr=errors)


def test_intersection_inherits_x_y_aware_covariance():
    x = np.arange(7, dtype=float)
    first = pap.linear_fit(x, 2*x + 1 + np.array([0.1,0,-.1,.1,0,-.1,.1]),
                           xerr=0.4, yerr=0.25)
    second = pap.linear_fit(x, -x + 12 + np.array([0, .1,-.1,.1,0,-.1,0]),
                            xerr=0.3, yerr=0.2)
    crossing = pap.intersect_lines(first, second)
    assert np.isfinite([crossing.x, crossing.y, crossing.x_err, crossing.y_err]).all()
    assert crossing.x_err > 0 and crossing.y_err > 0
    assert np.all(np.linalg.eigvalsh(crossing.cov_matrix) > 0)
