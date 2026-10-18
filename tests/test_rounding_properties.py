"""Contract and consistency tests for PAP rounding and its output formats."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import pap_toolbox as pap


@pytest.mark.parametrize(
    "uncertainty",
    [
        0.000001,
        0.00999,
        0.01,
        0.01999,
        0.02999,
        0.04000000000000001,
        0.09999,
        0.1,
        0.1999,
        0.2001,
        0.2999,
        0.3001,
        0.9999,
        1.0,
        1.999,
        2.999,
        9.999,
        4376.84604035785,
        1e8,
    ],
)
def test_uncertainty_is_never_rounded_down(uncertainty):
    _, rounded_error = pap.round_pap(98449.71337263903, uncertainty)
    assert Decimal(str(rounded_error)) >= Decimal(str(uncertainty))


def test_conservative_rounding_over_logarithmic_range():
    """A reproducible broad check, without making the test depend on randomness."""
    rng = np.random.default_rng(20261018)
    uncertainties = 10.0 ** rng.uniform(-8, 8, size=256)
    for uncertainty in uncertainties:
        original = float(uncertainty)
        _, rounded = pap.round_pap(-12345.6789, original)
        assert rounded >= original, (original, rounded)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (98449.71337263903, 4376.84604035785),
        (1.2345, 0.04376),
        (-1.2345, 0.2341),
        (123.456, 0.1999),
        (0.0, 0.02999),
        (1.2345, 0.04),
        (10.0, 0.9999),
    ],
)
def test_latex_formats_agree_with_numeric_rounding(value, error):
    rounded_value, rounded_error = pap.round_pap(value, error)
    value_part, error_part = pap.format_separate(value, error)
    assert pap.format_latex(value, error) == rf"{value_part} $\pm$ {error_part}"
    assert float(value_part) == rounded_value
    assert float(error_part) == rounded_error


def test_vectorized_rounding_and_formatting_agree_with_scalar_functions():
    values = np.array([1.2345, 98449.71337263903, -2.444])
    errors = np.array([0.04376, 4376.84604035785, 0.2341])
    rounded_values, rounded_errors = pap.v_round_pap(values, errors)
    latex_values = pap.v_format_latex(values, errors)
    for i in range(len(values)):
        assert (rounded_values[i], rounded_errors[i]) == pap.round_pap(values[i], errors[i])
        assert latex_values[i] == pap.format_latex(values[i], errors[i])


def test_table_output_uses_same_uncertainty_rounding():
    values = [98449.71337263903, 1.2345, -1.2345]
    errors = [4376.84604035785, 0.04376, 0.2341]
    df = pd.DataFrame({"value": values, "error": errors})
    latex = pap.export_pap_table(df, [("value", "error", r"$x$")])
    for value, error in zip(values, errors):
        assert pap.format_latex(value, error) in latex


@pytest.mark.parametrize(
    ("value", "error", "expected_error"),
    [
        (1.23, 0.1999, 0.20),  # leading 1: two significant digits
        (1.23, 0.2999, 0.30),  # leading 2: two significant digits
        (1.23, 0.3001, 0.4),   # leading 3: one significant digit
        (1.23, 0.0400, 0.04),  # exact already: no unnecessary inflation
    ],
)
def test_pap_significant_digit_boundaries(value, error, expected_error):
    _, rounded_error = pap.round_pap(value, error)
    assert rounded_error == pytest.approx(expected_error)
