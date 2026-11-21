"""Public API checks for convenience fitting."""

import pap_toolbox as pap
from pap_toolbox.linear import LineIntersection, LinearFitResult, intersect_lines, linear_fit


def test_linear_symbols_are_exported():
    for name in ("linear_fit", "intersect_lines", "LinearFitResult", "LineIntersection"):
        assert name in pap.__all__
    assert pap.linear_fit is linear_fit
    assert pap.intersect_lines is intersect_lines
    assert pap.LinearFitResult is LinearFitResult
    assert pap.LineIntersection is LineIntersection
