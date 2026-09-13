import pap_toolbox as pap
from pap_toolbox import device_errors, fitting, latex, propagation, rounding, statistics


def test_public_api_reexports_expected_symbols():
    expected = {
        "err_digital",
        "err_analog",
        "series_stats",
        "weighted_mean",
        "first_digit_func",
        "round_pap",
        "compare_to_literature",
        "ErrorPropagator",
        "format_latex",
        "format_separate",
        "export_pap_table",
        "FitResult",
        "fit_curve",
        "min_max_slope",
    }
    assert expected.issubset(set(pap.__all__))


def test_functions_live_in_expected_modules():
    assert device_errors.err_digital is pap.err_digital
    assert statistics.series_stats is pap.series_stats
    assert rounding.round_pap is pap.round_pap
    assert propagation.ErrorPropagator is pap.ErrorPropagator
    assert fitting.fit_curve is pap.fit_curve
    assert latex.export_pap_table is pap.export_pap_table
