"""Range selection, constraints, origin fits and generic residual diagnostics."""
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import pap_toolbox as pap


def _quadratic(x, a, b):
    return a*x**2 + b


def test_nonlinear_union_selects_each_point_once_and_slices_errors():
    x = np.arange(9.)
    y = 2*x*x+3
    y[4] = 999
    df = pd.DataFrame(dict(x=x, y=y, dx=x/100, dy=0.2+x/100))
    fit = pap.fit_curve(_quadratic, df, "x", "y", x_err_col="dx",
                        y_err_col=df.dy.to_numpy(), fitranges=[(0, 3), (2, 3), (5, 8)])
    keep = x != 4
    np.testing.assert_array_equal(fit.x_data, x[keep])
    np.testing.assert_allclose(fit.x_err, df.dx[keep])
    np.testing.assert_allclose(fit.y_err, df.dy[keep])
    np.testing.assert_allclose(fit.params, [2, 3])
    assert fit.ndof == 6


def test_nonlinear_single_range_and_active_bounds():
    x = np.arange(6.)
    fit = pap.fit_curve(_quadratic, {"x": x, "y": -2*x*x+3}, "x", "y",
                        y_err_col=0.2, x_range=(1, 5), bounds=([0, -np.inf], [np.inf, np.inf]))
    assert 0 <= fit.params[0] < 1e-7
    assert len(fit.x_data) == 5


@pytest.mark.parametrize("kwargs", [
    {"x_range": (0, 2), "fitranges": [(0, 2)]},
    {"fitranges": []}, {"fitranges": [(2, 1)]}, {"fitranges": [(30, 40)]},
])
def test_invalid_ranges(kwargs):
    x = np.arange(5.)
    with pytest.raises(ValueError):
        pap.fit_curve(_quadratic, {"x": x, "y": x*x}, "x", "y", **kwargs)
    with pytest.raises(ValueError):
        pap.linear_fit(x, x, **kwargs)


def test_origin_fit_matches_weighted_closed_form():
    x = np.array([1., 2., 3., 4., 5.])
    y = np.array([2., 3.8, 6.1, 7.7, 10.4])
    dy = np.array([.3, .2, .5, .3, .2])
    fit = pap.linear_fit(x, y, yerr=dy, fit_intercept=False)
    w = 1/dy**2
    assert fit.slope == pytest.approx(np.sum(w*x*y)/np.sum(w*x*x))
    assert fit.slope_err == pytest.approx(1/np.sqrt(np.sum(w*x*x)), rel=1e-6)
    assert fit.intercept == fit.intercept_err == 0
    assert fit.params.shape == (1,)
    assert fit.cov_matrix.shape == (1,1)
    assert fit.ndof == len(x)-1
    np.testing.assert_array_equal(fit.line_cov_matrix[1], [0,0])


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("xerr", [None, .3])
def test_linear_bounds_enforced(fit_intercept, xerr):
    x = np.arange(1., 7.)
    y = 2*x + 0.2
    bounds = ([0, -10], [1, 10]) if fit_intercept else (0, 1)
    fit = pap.linear_fit(x, y, yerr=.4, xerr=xerr, fit_intercept=fit_intercept, bounds=bounds)
    assert fit.slope == pytest.approx(1, abs=1e-7)
    assert fit.ndof == len(x) - (2 if fit_intercept else 1)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_x_only_bounds_on_both_signs(sign, fit_intercept):
    x = np.arange(1., 7.)
    y = sign*2*x
    bounds = (0.5, 1.) if sign > 0 else (-1., -.5)
    if fit_intercept:
        bounds = ([bounds[0], -10], [bounds[1], 10])
    fit = pap.linear_fit(x, y, xerr=.2, fit_intercept=fit_intercept, bounds=bounds)
    assert fit.slope == pytest.approx(sign, abs=1e-8)
    assert np.isfinite(fit.chi2)


def test_origin_eiv_matches_uncentered_deming_and_union():
    x = np.arange(1., 8.)
    y = 2*x + np.array([.2,-.3,.1,99,-.2,.3,.1])
    sx, sy = .3, .2
    fit = pap.linear_fit(x, y, xerr=sx, yerr=sy, fit_intercept=False,
                         fitranges=[(1,3),(5,7)])
    sel = x != 4
    xx, yy, xy = np.dot(x[sel],x[sel]), np.dot(y[sel],y[sel]), np.dot(x[sel],y[sel])
    ratio = (sy/sx)**2
    expected = (yy-ratio*xx+np.sqrt((yy-ratio*xx)**2+4*ratio*xy**2))/(2*xy)
    assert fit.slope == pytest.approx(expected, rel=1e-7)
    assert fit.ndof == 5
    np.testing.assert_allclose(fit.pulls, fit.residuals/np.hypot(sy,fit.slope*sx))


def test_intersection_with_fixed_intercept():
    x = np.arange(1., 7.)
    a = pap.linear_fit(x, 2*x, yerr=.2, fit_intercept=False)
    b = pap.linear_fit(x, -x+9, yerr=.2)
    crossing = pap.intersect_lines(a, b)
    assert crossing.x == pytest.approx(3)
    assert crossing.y == pytest.approx(6)
    assert crossing.x_err > 0


def test_generic_residual_plot_and_report(monkeypatch, capsys):
    monkeypatch.setattr(plt, "show", lambda: None)
    x = np.arange(6.)
    y = 2*x*x+3+np.array([0,.1,0,-.2,.1,0])
    fit = pap.fit_curve(_quadratic, {"x":x,"y":y}, "x", "y",
                        y_err_col=.3, show=True, residuals=True, standardized=True,
                        plot_band=False)
    try:
        assert len(fit.axes) == 2
        assert fit.axes[1].get_ylabel() == "Pulls"
        np.testing.assert_allclose(fit.pulls, (y-fit.evaluate(x))/.3)
        assert "FIT ERGEBNISSE" in capsys.readouterr().out
        ax = fit.plot_residuals()
        plt.close(ax.figure)
        top,bottom = fit.plot(residuals=True, plot_band=False)
        assert bottom.get_ylabel() == "Residuen"
        plt.close(top.figure)
    finally:
        plt.close(fit.axes[0].figure)


def test_generic_pulls_without_errors_raise():
    x = np.arange(5.)
    fit = pap.fit_curve(_quadratic, {"x":x,"y":x*x}, "x","y")
    with pytest.raises(ValueError, match="yerr"):
        fit.plot_residuals(standardized=True)
