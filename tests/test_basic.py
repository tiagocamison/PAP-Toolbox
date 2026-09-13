import numpy as np

import pap_toolbox as pap


def test_err_digital_scalar():
    assert np.isclose(pap.err_digital(10.0, percent=1.0, digits=2, resolution=0.01), 0.12)


def test_err_digital_array():
    result = pap.err_digital(np.array([-10.0, 5.0]), percent=2.0)
    np.testing.assert_allclose(result, [0.2, 0.1])


def test_err_analog_default_fraction():
    assert pap.err_analog(0.2) == 0.1


def test_series_stats():
    stats = pap.series_stats([1.0, 2.0, 3.0])
    assert stats["N"] == 3
    assert stats["mean"] == 2.0
    assert np.isclose(stats["std"], 1.0)
    assert np.isclose(stats["sem"], 1.0 / np.sqrt(3.0))


def test_weighted_mean_equal_errors():
    mean, err = pap.weighted_mean([1.0, 3.0], [1.0, 1.0])
    assert np.isclose(mean, 2.0)
    assert np.isclose(err, 1.0 / np.sqrt(2.0))


def test_round_pap():
    value, error = pap.round_pap(12.345, 0.67)
    assert value == 12.3
    assert error == 0.7


def test_public_error_propagator_importable():
    propagator = pap.ErrorPropagator("x + y")
    assert {symbol.name for symbol in propagator.vars} == {"x", "y"}
