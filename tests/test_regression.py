import numpy as np
import pandas as pd

import pap_toolbox as pap


def test_format_latex_and_sigma_comparison():
    assert pap.format_latex(12.345, 0.67) == r"12.3 $\pm$ 0.7"
    assert np.isclose(pap.compare_to_literature(10.0, 1.0, 12.0, 0.0), 2.0)


def test_error_propagator_evaluate():
    prop = pap.ErrorPropagator("x + y")
    value, error = prop.evaluate({"x": 3.0, "y": 4.0}, {"x": 0.3, "y": 0.4})
    assert np.isclose(value, 7.0)
    assert np.isclose(error, 0.5)


def test_export_pap_table_contains_expected_latex():
    df = pd.DataFrame({"x": [1.23], "dx": [0.04]})
    table = pap.export_pap_table(df, [("x", "dx", r"$x$")], caption="Test", label="tab:test")
    assert r"\begin{table}[H]" in table
    assert r"1.23 $\pm$ 0.04" in table
    assert r"\label{tab:test}" in table


def test_fit_curve_preserves_fit_result_api():
    def linear(x, a, b):
        return a * x + b

    data = {
        "x": np.arange(5.0),
        "y": 2.0 * np.arange(5.0) + 1.0,
        "dy": np.full(5, 0.1),
    }
    result = pap.fit_curve(linear, data, "x", "y", y_err_col="dy", p0=[1.0, 0.0])
    np.testing.assert_allclose(result.params, [2.0, 1.0], atol=1e-10)
    assert result.ndof == 3
    assert np.isclose(result.chi2, 0.0, atol=1e-20)
    assert np.isclose(result.evaluate(2.5), 6.0)


def test_min_max_slope_regression():
    m_opt, dm = pap.min_max_slope([0.0, 2.0], [1.0, 5.0], [0.2, 0.2])
    assert np.isclose(m_opt, 2.0)
    assert np.isclose(dm, 0.2)
