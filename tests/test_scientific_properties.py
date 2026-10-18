"""Independent analytic checks of propagation, least squares, and plotting."""

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pytest
from matplotlib.collections import PolyCollection

import pap_toolbox as pap


@pytest.mark.parametrize(
    ("formula", "values", "errors", "expected_value", "expected_error"),
    [
        ("x*y", {"x": 2.0, "y": 3.0}, {"x": 0.2, "y": 0.3}, 6.0, np.sqrt(0.6**2 + 0.6**2)),
        ("x/y", {"x": 8.0, "y": 4.0}, {"x": 0.4, "y": 0.2}, 2.0, np.sqrt(0.1**2 + 0.1**2)),
        ("x**3", {"x": 2.0}, {"x": 0.1}, 8.0, 1.2),
    ],
)
def test_symbolic_error_propagation_matches_derivatives(
    formula, values, errors, expected_value, expected_error
):
    propagator = pap.ErrorPropagator(formula)
    actual_value, actual_error = propagator.evaluate(values, errors)
    assert actual_value == pytest.approx(expected_value)
    assert actual_error == pytest.approx(expected_error)


def test_symbolic_propagation_broadcasts_arrays_and_relative_errors():
    x = np.array([2.0, 3.0, 4.0])
    propagator = pap.ErrorPropagator("x*y")
    values, errors = propagator.evaluate(
        {"x": x, "y": 5.0}, {"x": "10%", "y": 0.2}
    )
    np.testing.assert_allclose(values, 5 * x)
    np.testing.assert_allclose(errors, np.sqrt((5 * 0.1 * x) ** 2 + (0.2 * x) ** 2))


def test_missing_symbolic_input_fails_informatively():
    propagator = pap.ErrorPropagator("x+y")
    with pytest.raises(ValueError, match="Fehlender Wert oder Fehler"):
        propagator.evaluate({"x": 1.0}, {"x": 0.1})


def test_first_significant_digit_of_zero_is_undefined():
    with pytest.raises(ValueError, match="undefined for zero"):
        pap.first_digit_func(0)


def _linear(x, slope, intercept):
    return slope * x + intercept


def _sample_data():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 2.3, 3.1, 4.2, 5.6, 6.0])
    sigma = np.array([0.20, 0.15, 0.30, 0.25, 0.40, 0.22])
    return {"x": x, "y": y, "sigma": sigma}


def test_weighted_fit_agrees_with_independent_linear_least_squares():
    data = _sample_data()
    x, y, sigma = data["x"], data["y"], data["sigma"]
    design = np.column_stack((x, np.ones_like(x)))
    weighted_design = design / sigma[:, None]
    weighted_y = y / sigma
    expected_cov = np.linalg.inv(weighted_design.T @ weighted_design)
    expected_parameters = expected_cov @ (weighted_design.T @ weighted_y)
    expected_chi2 = np.sum(((y - design @ expected_parameters) / sigma) ** 2)

    fit = pap.fit_curve(_linear, data, "x", "y", y_err_col="sigma", p0=[1.0, 0.0])
    np.testing.assert_allclose(fit.params, expected_parameters, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(fit.cov_matrix, expected_cov, rtol=3e-6, atol=1e-7)
    np.testing.assert_allclose(fit.param_errs, np.sqrt(np.diag(expected_cov)), rtol=3e-6)
    assert fit.chi2 == pytest.approx(expected_chi2, rel=1e-8, abs=1e-10)
    assert fit.ndof == len(x) - 2
    assert fit.chi2_red == pytest.approx(expected_chi2 / fit.ndof, rel=1e-8)


def test_fit_accepts_constant_uncertainties_and_preserves_them():
    data = _sample_data()
    fit = pap.fit_curve(
        _linear, data, "x", "y", x_err_col=0.05, y_err_col=0.2, p0=[1.0, 0.0]
    )
    np.testing.assert_allclose(fit.x_err, np.full(6, 0.05))
    np.testing.assert_allclose(fit.y_err, np.full(6, 0.2))
    expected_chi2 = np.sum(((data["y"] - fit.evaluate(data["x"])) / 0.2) ** 2)
    assert fit.chi2 == pytest.approx(expected_chi2)


def test_unweighted_fit_does_not_report_misleading_chi_square():
    data = _sample_data()
    fit = pap.fit_curve(_linear, data, "x", "y", p0=[1.0, 0.0])
    assert np.isnan(fit.chi2)
    assert np.isnan(fit.ndof)
    assert np.isnan(fit.chi2_red)


@pytest.mark.parametrize("with_band", [False, True])
def test_fit_plot_runs_headlessly_and_draws_requested_band(with_band):
    fit = pap.fit_curve(
        _linear, _sample_data(), "x", "y", y_err_col="sigma", p0=[1.0, 0.0]
    )
    fig, ax = plt.subplots()
    try:
        returned_ax = fit.plot(
            ax=ax,
            title="Fit diagnostics",
            xlabel="time",
            ylabel="distance",
            plot_band=with_band,
            language="en",
        )
        assert returned_ax is ax
        assert ax.get_title() == "Fit diagnostics"
        assert ax.get_xlabel() == "time"
        assert ax.get_ylabel() == "distance"
        assert len(ax.lines) >= 1
        has_band = any(isinstance(collection, PolyCollection) for collection in ax.collections)
        assert has_band is with_band
    finally:
        plt.close(fig)


def test_plot_without_fit_only_draws_measurements():
    fit = pap.fit_curve(_linear, _sample_data(), "x", "y", p0=[1.0, 0.0])
    fig, ax = plt.subplots()
    try:
        returned = fit.plot(ax=ax, plot_fit=False, plot_band=True, x_lim=(-1, 6))
        assert returned is ax
        assert tuple(ax.get_xlim()) == (-1, 6)
        assert not any(isinstance(c, PolyCollection) for c in ax.collections)
    finally:
        plt.close(fig)
