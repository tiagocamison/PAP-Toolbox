"""PAP Toolbox public API."""

from .device_errors import err_analog, err_digital
from .fitting import FitResult, fit_curve, min_max_slope
from .latex import export_pap_table, format_latex, format_separate, v_format_latex
from .propagation import ErrorPropagator
from .rounding import (
    compare_to_literature,
    first_digit_func,
    round_pap,
    v_compare_to_literature,
    v_round_pap,
)
from .statistics import series_stats, weighted_mean

__all__ = [
    "ErrorPropagator",
    "FitResult",
    "compare_to_literature",
    "err_analog",
    "err_digital",
    "export_pap_table",
    "first_digit_func",
    "fit_curve",
    "format_latex",
    "format_separate",
    "min_max_slope",
    "round_pap",
    "series_stats",
    "v_compare_to_literature",
    "v_format_latex",
    "v_round_pap",
    "weighted_mean",
]
