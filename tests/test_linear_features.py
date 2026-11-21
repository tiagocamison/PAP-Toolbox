"""Independent tests for the linear-fit notebook helpers."""

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import pytest

import pap_toolbox as pap


def _data():
    x = np.linspace(0, 5, 11)
    y = 1.8 * x + 0.7 + np.array(
        [0.02, -0.03, 0.01, 0.05, -0.01, 0.02, 0.01, -0.02, 0.04, 0.0, -0.04]
    )
    return pd.DataFrame({"t": x, "y": y, "dy": np.full(len(x), 0.15)})


def test_linear_fit_matches_existing_curve_fit_and_reports_coefficients():
    df = _data()
    with pytest.warns(UserWarning, match="xerr"):
        fit = pap.linear_fit(df, "t", "y", xerr=0.05, yerr="dy")
    reference = pap.fit_curve(
        lambda x, m, b: m * x + b, df, "t", "y",
        x_err_col=0.05, y_err_col="dy",
    )
    np.testing.assert_allclose(fit.params, reference.params, rtol=1e-6)
    np.testing.assert_allclose(fit.cov_matrix, reference.cov_matrix, rtol=1e-5)
    assert fit.slope == pytest.approx(reference.params[0])
    assert fit.intercept == pytest.approx(reference.params[1])
    assert fit.slope_err == pytest.approx(reference.param_errs[0])
    assert fit.intercept_err == pytest.approx(reference.param_errs[1])
    np.testing.assert_allclose(fit.x_err, np.full(len(df), 0.05))
    np.testing.assert_allclose(fit.residuals, df["y"] - fit.evaluate(df["t"]))
    np.testing.assert_allclose(fit.pulls, fit.residuals / df["dy"])


def test_array_input_and_unweighted_fit():
    x = np.arange(8.0)
    y = 2 * x - 1 + np.array([0.1, 0, -0.1, 0, 0.1, 0, -0.1, 0])
    fit = pap.linear_fit(x, y)
    assert fit.slope == pytest.approx(2, abs=0.03)
    assert np.isnan(fit.chi2)
    with pytest.raises(ValueError, match="yerr"):
        _ = fit.pulls


def test_range_is_inclusive_and_slices_error_arrays_consistently():
    df = _data()
    err = np.linspace(0.10, 0.20, len(df))
    fit = pap.linear_fit(df, "t", "y", yerr=err, x_range=(1.0, 3.0))
    np.testing.assert_allclose(fit.x_data, [1, 1.5, 2, 2.5, 3])
    np.testing.assert_allclose(fit.y_err, err[2:7])
    assert fit.ndof == 3


def test_piecewise_linear_ranges_and_intersection():
    x = np.arange(11, dtype=float)
    y = np.where(x < 5, 2 * x + 1, -x + 16)
    data = {"x": x, "y": y}
    first = pap.linear_fit(data, "x", "y", yerr=0.2, x_range=(0, 4))
    second = pap.linear_fit(data, "x", "y", yerr=0.2, x_range=(6, 10))
    crossing = pap.intersect_lines(first, second)
    assert crossing.x == pytest.approx(5, abs=1e-8)
    assert crossing.y == pytest.approx(11, abs=1e-8)
    assert crossing.x_err > 0 and crossing.y_err > 0
    np.testing.assert_allclose(crossing.cov_matrix, crossing.cov_matrix.T, atol=1e-12)


def test_intersection_uncertainty_uses_full_parameter_covariance():
    # Finite differences of the *intersection itself* provide an independent
    # Jacobian reference; ignoring cov(m, b) would fail this assertion.
    d1 = {"x": np.arange(6.0), "y": 2 * np.arange(6.0) + 1}
    d2 = {"x": np.arange(6.0), "y": -1 * np.arange(6.0) + 16}
    a = pap.linear_fit(d1, "x", "y", yerr=0.3)
    b = pap.linear_fit(d2, "x", "y", yerr=0.4)
    result = pap.intersect_lines(a, b)
    theta = np.array([a.slope, a.intercept, b.slope, b.intercept])

    def intersect(p):
        x0 = (p[3] - p[1]) / (p[0] - p[2])
        return np.array([x0, p[0] * x0 + p[1]])

    delta = 1e-5
    jac = np.column_stack([
        (intersect(theta + np.eye(4)[i] * delta) -
         intersect(theta - np.eye(4)[i] * delta)) / (2 * delta)
        for i in range(4)
    ])
    combined_cov = np.zeros((4, 4))
    combined_cov[:2, :2] = a.cov_matrix
    combined_cov[2:, 2:] = b.cov_matrix
    expected_cov = jac @ combined_cov @ jac.T
    np.testing.assert_allclose(result.cov_matrix, expected_cov, rtol=2e-6, atol=1e-10)


def test_log_intersection_to_physical_frequency_and_invalid_base():
    data1 = {"x": np.arange(6.0), "y": 2 * np.arange(6.0) + 1}
    data2 = {"x": np.arange(6.0), "y": -1 * np.arange(6.0) + 7}
    a = pap.linear_fit(data1, "x", "y", yerr=0.2)
    b = pap.linear_fit(data2, "x", "y", yerr=0.2)
    crossing = pap.intersect_lines(a, b)
    f, df = crossing.as_frequency(base=10)
    assert f == pytest.approx(100)
    assert df == pytest.approx(np.log(10) * 100 * crossing.x_err)
    with pytest.raises(ValueError):
        crossing.as_frequency(base=1)


def test_parallel_lines_and_wrong_inputs_fail():
    x = np.linspace(0, 5, 8)
    a = pap.linear_fit(x, 2 * x + 1, yerr=0.1)
    b = pap.linear_fit(x, 2 * x + 2, yerr=0.1)
    with pytest.raises(ValueError, match="parallel"):
        pap.intersect_lines(a, b)
    with pytest.raises(TypeError):
        pap.intersect_lines(a, object())


@pytest.mark.parametrize(
    ("bad_kwargs", "message"),
    [
        ({"x_range": (10, 20)}, "drei Messpunkte"),
        ({"x_range": (3, 1)}, "aufsteigende Grenzen"),
        ({"yerr": 0}, "strikt positiv"),
        ({"yerr": -0.1}, "strikt positiv"),
        ({"xerr": -0.1}, "nichtnegativ"),
        ({"yerr": [0.1]}, "Einträge"),
    ],
)
def test_invalid_inputs_raise_descriptive_errors(bad_kwargs, message):
    with pytest.raises(ValueError, match=message):
        pap.linear_fit(_data(), "t", "y", **bad_kwargs)


def test_three_point_minimum_and_nonfinite_input():
    with pytest.raises(ValueError, match="drei Messpunkte"):
        pap.linear_fit([0, 1], [1, 2])
    with pytest.raises(ValueError, match="endlich"):
        pap.linear_fit([0, 1, 2, 3], [0, 1, np.nan, 3])


def test_show_shortcut_report_units_and_plot(monkeypatch, capsys):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    fit = pap.linear_fit(
        _data(), "t", "y", yerr="dy", show=True,
        x_unit="s", y_unit="V", title="Example",
    )
    out = capsys.readouterr().out
    assert "m =" in out and "b =" in out
    assert "[V/s]" in out and "[V]" in out
    assert "chi2/ndof" in out
    assert shown == [True]
    assert fit.axes[0].get_xlabel() == "t [s]"
    assert fit.axes[0].get_ylabel() == "y [V]"
    plt.close(fit.axes[0].figure)


def test_residual_panel_can_be_requested_without_report(monkeypatch, capsys):
    monkeypatch.setattr(plt, "show", lambda: None)
    fit = pap.linear_fit(_data(), "t", "y", yerr="dy", plot=True, report=False,
                         residuals=True, standardized=True)
    assert len(fit.axes) == 2
    assert fit.axes[1].get_ylabel() == "Pulls"
    assert len(fit.axes[1].lines) >= 1
    assert capsys.readouterr().out == ""
    plt.close(fit.axes[0].figure)


def test_plot_and_report_can_be_disabled_independently(monkeypatch, capsys):
    monkeypatch.setattr(plt, "show", lambda: (_ for _ in ()).throw(AssertionError("plot")))
    fit = pap.linear_fit(_data(), "t", "y", yerr="dy", show=True, plot=False)
    assert fit.axes is None
    assert "LINEARER FIT" in capsys.readouterr().out
    with pytest.raises(ValueError, match="Residuen"):
        pap.linear_fit(_data(), "t", "y", residuals=True)


def test_finite_nonzero_intersection_covariance():
    d1 = _data()
    d2 = _data().copy()
    d2["y"] = -0.5 * d2["t"] + 8 + np.linspace(-0.04, 0.05, len(d2))
    a = pap.linear_fit(d1, "t", "y", yerr="dy")
    b = pap.linear_fit(d2, "t", "y", yerr="dy")
    result = pap.intersect_lines(a, b)
    eigs = np.linalg.eigvalsh(result.cov_matrix)
    assert np.all(eigs >= -1e-12)
